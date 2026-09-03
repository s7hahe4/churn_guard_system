import io
import pandas as pd
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.views.generic import TemplateView
from django.db.models import Avg, Count, Q
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import (
    CustomerDataSerializer,
    PredictionRecordSerializer,
    ModelVersionSerializer,
    BatchJobStatusSerializer
)
from .services import ChurnModelService
from .models import PredictionRecord, BatchUploadJob, ModelVersion
from .tasks import process_batch_job
import socket


def is_celery_broker_reachable(timeout: float = 0.15) -> bool:
    """Fast check to verify if the Celery message broker (Redis) is actively listening."""
    from django.conf import settings
    if getattr(settings, 'CELERY_TASK_ALWAYS_EAGER', False):
        return True
    try:
        sock = socket.create_connection(('127.0.0.1', 6379), timeout=timeout)
        sock.close()
        return True
    except Exception:
        return False


class DashboardView(TemplateView):
    """Renders the interactive ChurnGuard AI dashboard with live KPI metrics."""
    template_name = 'predictions/dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        try:
            records = PredictionRecord.objects.order_by('-created_at')
            total_count = records.count()
            high_risk_count = records.filter(risk_level='High').count()
            medium_risk_count = records.filter(risk_level='Medium').count()
            low_risk_count = records.filter(risk_level='Low').count()
            churn_count = records.filter(churn_predicted=True).count()
            
            avg_tenure = records.aggregate(avg=Avg('tenure'))['avg'] or 0
            avg_charges = records.aggregate(avg=Avg('monthly_charges'))['avg'] or 0
            
            context['stats'] = {
                'total_predictions': total_count,
                'high_risk_count': high_risk_count,
                'medium_risk_count': medium_risk_count,
                'low_risk_count': low_risk_count,
                'churn_count': churn_count,
                'churn_rate': round((churn_count / total_count * 100), 1) if total_count > 0 else 0,
                'avg_tenure': round(avg_tenure, 1),
                'avg_charges': round(avg_charges, 2),
            }
            context['recent_predictions'] = records[:10]
            context['active_model'] = ChurnModelService.get_active_model_info()
        except Exception as e:
            print(f"Context initialization note: {e}")
            context['stats'] = {
                'total_predictions': 0,
                'high_risk_count': 0,
                'medium_risk_count': 0,
                'low_risk_count': 0,
                'churn_count': 0,
                'churn_rate': 0,
                'avg_tenure': 0,
                'avg_charges': 0,
            }
            context['recent_predictions'] = []
            context['active_model'] = {
                'name': 'RandomForest (Active)',
                'algorithm': 'RandomForest',
                'roc_auc': 0.838,
                'is_active': True
            }
        return context


class PredictChurnView(APIView):
    """API endpoint for single customer churn prediction with SHAP explainability."""
    def post(self, request):
        serializer = CustomerDataSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({"errors": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)
        
        input_data = serializer.validated_data
        
        try:
            prediction_result = ChurnModelService.predict(input_data)
            active_version = ChurnModelService._get_active_model_version()
            
            # Save record in database with model_version tracking
            record = PredictionRecord.objects.create(
                customer_id=input_data.get('customerID', 'Anonymous'),
                tenure=prediction_result['tenure'],
                monthly_charges=prediction_result['monthly_charges'],
                contract_type=prediction_result['contract'],
                churn_probability=prediction_result['churn_probability'],
                risk_level=prediction_result['risk_level'],
                churn_predicted=prediction_result['churn_predicted'],
                model_version=active_version
            )
            
            response_data = {
                "id": record.id,
                "customer_id": record.customer_id,
                "model_name": active_version.name if active_version else "RandomForest (Baseline)",
                **prediction_result
            }
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class BatchPredictView(APIView):
    """
    API endpoint to upload CSV and trigger asynchronous bulk churn predictions via Celery.
    Gracefully falls back to synchronous processing if Celery broker is unavailable.
    """
    def post(self, request):
        if 'file' not in request.FILES:
            return Response({"error": "No CSV file provided."}, status=status.HTTP_400_BAD_REQUEST)
        
        uploaded_file = request.FILES['file']
        if not uploaded_file.name.endswith(('.csv', '.CSV')):
            return Response({"error": "Invalid file type. Please upload a .csv file."}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Create the batch upload job record
            job = BatchUploadJob.objects.create(
                csv_file=uploaded_file,
                status='PENDING',
                progress_percentage=0
            )

            # Check if Celery message broker is active and reachable
            async_dispatched = False
            if is_celery_broker_reachable():
                try:
                    task = process_batch_job.delay(job.id)
                    job.celery_task_id = task.id
                    job.save(update_fields=['celery_task_id'])
                    async_dispatched = True
                except Exception as e:
                    print(f"Celery dispatch failed: {e}")

            if not async_dispatched:
                # Process synchronously when background worker broker is unavailable
                try:
                    process_batch_job(job.id)
                    job.refresh_from_db()
                except Exception as sync_err:
                    return Response({"error": f"Error processing batch: {str(sync_err)}"},
                                    status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            return Response({
                "job_id": job.id,
                "status": job.status,
                "progress_percentage": job.progress_percentage,
                "async": async_dispatched,
                "message": "Batch processing initiated via Celery." if async_dispatched else "Batch processed synchronously (broker offline)."
            }, status=status.HTTP_202_ACCEPTED if async_dispatched else status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": f"Error initiating batch: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class BatchJobStatusView(APIView):
    """
    Polling endpoint for batch prediction progress.
    Returns status, percentage, and results preview upon completion.
    """
    def get(self, request, job_id):
        job = get_object_or_404(BatchUploadJob, id=job_id)
        serializer = BatchJobStatusSerializer(job)
        data = serializer.data

        if job.status == 'COMPLETED':
            # Include summary metrics and recent preview rows
            recent_batch_records = PredictionRecord.objects.order_by('-created_at')[:job.total_records]
            preview = [
                {
                    "customer_id": r.customer_id,
                    "contract_type": r.contract_type,
                    "tenure": r.tenure,
                    "monthly_charges": r.monthly_charges,
                    "churn_probability": r.churn_probability,
                    "churn_percentage": round(r.churn_probability * 100, 1),
                    "risk_level": r.risk_level,
                    "churn_predicted": r.churn_predicted,
                }
                for r in recent_batch_records[:20]
            ]
            churn_rate = round((job.churn_detected_count / job.total_records * 100), 1) if job.total_records > 0 else 0
            high_count = sum(1 for r in preview if r['risk_level'] == 'High')
            med_count = sum(1 for r in preview if r['risk_level'] == 'Medium')
            low_count = sum(1 for r in preview if r['risk_level'] == 'Low')

            data.update({
                "churn_rate": churn_rate,
                "high_risk_count": high_count,
                "medium_risk_count": med_count,
                "low_risk_count": low_count,
                "preview": preview
            })

        return Response(data, status=status.HTTP_200_OK)


class ModelRegistryView(APIView):
    """Lists all registered models with their evaluation metrics."""
    def get(self, request):
        models = ModelVersion.objects.all()
        serializer = ModelVersionSerializer(models, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class SetActiveModelView(APIView):
    """Sets a specific model version as active in production."""
    def post(self, request, pk):
        model_version = get_object_or_404(ModelVersion, pk=pk)
        
        # Deactivate all and activate selected
        ModelVersion.objects.update(is_active=False)
        model_version.is_active = True
        model_version.save(update_fields=['is_active'])

        # Invalidate in-memory cache so next request loads this model
        ChurnModelService._model = None
        ChurnModelService._shap_explainer = None
        ChurnModelService._shap_background = None
        ChurnModelService._feature_names = None
        ChurnModelService._active_version_id = model_version.id

        return Response({
            "message": f"Activated model: {model_version.name}",
            "active_model": ModelVersionSerializer(model_version).data
        }, status=status.HTTP_200_OK)


class PredictionHistoryView(APIView):
    """Returns recent prediction records with filtering."""
    def get(self, request):
        risk = request.query_params.get('risk')
        search = request.query_params.get('search')
        limit = int(request.query_params.get('limit', 50))
        
        queryset = PredictionRecord.objects.select_related('model_version').order_by('-created_at')
        if risk and risk != 'all':
            queryset = queryset.filter(risk_level__iexact=risk)
        if search:
            queryset = queryset.filter(customer_id__icontains=search)
            
        records = queryset[:limit]
        serializer = PredictionRecordSerializer(records, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class PredictionStatsView(APIView):
    """Returns aggregated stats for real-time dashboard charts."""
    def get(self, request):
        records = PredictionRecord.objects.all()
        total = records.count()
        if total == 0:
            return Response({
                "total": 0,
                "high_risk": 0,
                "medium_risk": 0,
                "low_risk": 0,
                "churn_count": 0,
                "churn_rate": 0,
                "avg_tenure": 0,
                "avg_charges": 0,
                "contracts": []
            })
            
        high = records.filter(risk_level='High').count()
        medium = records.filter(risk_level='Medium').count()
        low = records.filter(risk_level='Low').count()
        churn = records.filter(churn_predicted=True).count()
        avg_tenure = records.aggregate(avg=Avg('tenure'))['avg'] or 0
        avg_charges = records.aggregate(avg=Avg('monthly_charges'))['avg'] or 0

        # Contract type distribution
        contract_stats = list(
            PredictionRecord.objects.values('contract_type')
            .annotate(count=Count('id'), churn_count=Count('id', filter=Q(churn_predicted=True)))
        )

        return Response({
            "total": total,
            "high_risk": high,
            "medium_risk": medium,
            "low_risk": low,
            "churn_count": churn,
            "churn_rate": round((churn / total) * 100, 1),
            "avg_tenure": round(avg_tenure, 1),
            "avg_charges": round(avg_charges, 2),
            "contracts": contract_stats
        })


def download_sample_csv(request):
    """Generates a downloadable sample CSV template for batch predictions."""
    sample_data = """customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges
7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85
5575-GNVDE,Male,0,No,No,34,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,No,Mailed check,56.95,1889.5
3668-QPYBK,Male,0,No,No,2,Yes,No,DSL,Yes,Yes,No,No,No,No,Month-to-month,Yes,Mailed check,53.85,108.15
7795-CFOCW,Male,0,No,No,45,No,No phone service,DSL,Yes,No,Yes,Yes,No,No,One year,No,Bank transfer (automatic),42.30,1840.75
9237-HQITU,Female,0,No,No,2,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,70.70,151.65
9305-CDSKC,Female,0,No,No,8,Yes,Yes,Fiber optic,No,No,Yes,No,Yes,Yes,Month-to-month,Yes,Electronic check,99.65,820.5
1452-KIOVK,Male,0,No,Yes,22,Yes,Yes,Fiber optic,No,Yes,No,No,Yes,No,Month-to-month,Yes,Credit card (automatic),89.10,1949.4
6713-OKOMC,Female,0,No,No,10,No,No phone service,DSL,Yes,No,No,No,No,No,Month-to-month,No,Mailed check,29.75,301.9
"""
    response = HttpResponse(sample_data, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="telco_churn_batch_template.csv"'
    return response