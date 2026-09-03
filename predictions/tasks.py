"""
Celery background tasks for ChurnGuard AI.

Handles asynchronous processing of large CSV batch uploads to prevent
HTTP request timeouts and ensure horizontal scalability.
"""

import io
import pandas as pd
from celery import shared_task
from .models import BatchUploadJob, PredictionRecord
from .services import ChurnModelService


@shared_task(bind=True)
def process_batch_job(self, job_id: int):
    """
    Celery task to asynchronously process an uploaded CSV file.
    Updates job status and progress percentage throughout execution.
    """
    try:
        job = BatchUploadJob.objects.get(id=job_id)
    except BatchUploadJob.DoesNotExist:
        return {"error": f"Job {job_id} not found."}

    try:
        job.status = 'PROCESSING'
        job.progress_percentage = 10
        if self and hasattr(self, 'request') and self.request.id:
            job.celery_task_id = self.request.id
        job.save(update_fields=['status', 'progress_percentage', 'celery_task_id'])

        # 1. Read CSV file
        file_path = job.csv_file.path
        df = pd.read_csv(file_path)

        if df.empty:
            job.status = 'FAILED'
            job.error_message = 'Uploaded CSV file is empty.'
            job.save(update_fields=['status', 'error_message'])
            return {"error": "CSV file is empty."}

        df.columns = df.columns.str.strip()
        job.progress_percentage = 30
        job.save(update_fields=['progress_percentage'])

        # 2. Run batch prediction with ML pipeline
        results = ChurnModelService.predict_batch(df)
        job.progress_percentage = 65
        job.save(update_fields=['progress_percentage'])

        # 3. Associate with current active model version
        active_version = ChurnModelService._get_active_model_version()

        # 4. Bulk insert records into database
        records_to_create = [
            PredictionRecord(
                customer_id=r['customer_id'],
                tenure=r['tenure'],
                monthly_charges=r['monthly_charges'],
                contract_type=r['contract_type'],
                churn_probability=r['churn_probability'],
                risk_level=r['risk_level'],
                churn_predicted=r['churn_predicted'],
                model_version=active_version
            )
            for r in results
        ]
        PredictionRecord.objects.bulk_create(records_to_create, batch_size=1000)
        job.progress_percentage = 90
        job.save(update_fields=['progress_percentage'])

        # 5. Finalize job metrics
        churn_detected = sum(1 for r in results if r['churn_predicted'])
        job.status = 'COMPLETED'
        job.total_records = len(results)
        job.churn_detected_count = churn_detected
        job.progress_percentage = 100
        job.save(update_fields=['status', 'total_records', 'churn_detected_count', 'progress_percentage'])

        return {
            "status": "COMPLETED",
            "job_id": job.id,
            "total_records": len(results),
            "churn_detected": churn_detected
        }

    except Exception as exc:
        job.status = 'FAILED'
        job.error_message = str(exc)
        job.save(update_fields=['status', 'error_message'])
        raise exc
