from django.db import models

class PredictionRecord(models.Model):
    """Stores individual customer predictions for audit and history."""
    created_at = models.DateTimeField(auto_now_add=True)
    customer_id = models.CharField(max_length=50, blank=True, null=True, default="Anonymous")
    
    # Just a few key features to remember who this was
    tenure = models.IntegerField()
    monthly_charges = models.FloatField()
    contract_type = models.CharField(max_length=50)
    
    # The Machine Learning Results
    churn_probability = models.FloatField()
    risk_level = models.CharField(max_length=20) # 'High', 'Medium', 'Low'
    churn_predicted = models.BooleanField()      # True if probability >= 0.5

    def __str__(self):
        return f"{self.customer_id} - {self.risk_level} Risk ({self.churn_probability:.2f})"


class BatchUploadJob(models.Model):
    """Tracks CSV files uploaded for bulk processing."""
    uploaded_at = models.DateTimeField(auto_now_add=True)
    csv_file = models.FileField(upload_to='batch_uploads/')
    
    # Status tracks where Celery is in the background
    status = models.CharField(max_length=20, default='PENDING') 
    
    # Analytics for the dashboard
    total_records = models.IntegerField(default=0)
    churn_detected_count = models.IntegerField(default=0)

    def __str__(self):
        return f"Batch {self.id} - {self.status}"