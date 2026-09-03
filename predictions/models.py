from django.db import models


class ModelVersion(models.Model):
    """Registry of trained ML model versions with evaluation metrics.
    
    Design decision: A database-backed registry (instead of file-based versioning)
    provides an audit trail, metric comparison, and atomic switching between models.
    This is a mini-MLflow pattern — each model records its algorithm, training date,
    dataset hash, and held-out evaluation metrics so you can compare RF vs XGBoost
    vs LightGBM side-by-side and switch the active production model without redeploying.
    """
    name = models.CharField(max_length=100)              # e.g., "RandomForest v1"
    algorithm = models.CharField(max_length=50)           # "RandomForest", "XGBoost", "LightGBM"
    version = models.CharField(max_length=20)             # "1.0", "1.1", "2.0"
    model_file = models.CharField(max_length=255)         # relative path to .joblib
    shap_background_file = models.CharField(max_length=255, blank=True, null=True)
    feature_names_file = models.CharField(max_length=255, blank=True, null=True)
    is_active = models.BooleanField(default=False)        # only one should be active at a time

    # Training metadata
    trained_at = models.DateTimeField()
    dataset_hash = models.CharField(max_length=64)        # SHA256 of training CSV
    dataset_rows = models.IntegerField()

    # Evaluation metrics (on held-out test set)
    accuracy = models.FloatField(default=0.0)
    roc_auc = models.FloatField(default=0.0)
    pr_auc = models.FloatField(default=0.0)
    f1_churn = models.FloatField(default=0.0)             # F1 on the churn (positive) class
    precision_churn = models.FloatField(default=0.0)
    recall_churn = models.FloatField(default=0.0)

    # Class imbalance handling
    imbalance_strategy = models.CharField(max_length=50, default='none')  # "none", "class_weight", "smote"

    notes = models.TextField(blank=True)

    class Meta:
        ordering = ['-trained_at']

    def __str__(self):
        active_label = " [ACTIVE]" if self.is_active else ""
        return f"{self.name} ({self.algorithm}) — ROC-AUC: {self.roc_auc:.4f}{active_label}"


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

    # Track which model made this prediction
    model_version = models.ForeignKey(
        ModelVersion, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='predictions'
    )

    class Meta:
        indexes = [
            models.Index(fields=['-created_at'], name='idx_pred_created'),
            models.Index(fields=['risk_level'], name='idx_pred_risk'),
            models.Index(fields=['customer_id'], name='idx_pred_customer'),
        ]

    def __str__(self):
        return f"{self.customer_id} - {self.risk_level} Risk ({self.churn_probability:.2f})"


class BatchUploadJob(models.Model):
    """Tracks CSV files uploaded for bulk processing.
    
    Design decision: Batch jobs run asynchronously via Celery. The API returns
    a job ID immediately and the frontend polls GET /api/batch-status/<id>/
    for completion. This avoids HTTP request timeouts on large CSV files.
    """
    uploaded_at = models.DateTimeField(auto_now_add=True)
    csv_file = models.FileField(upload_to='batch_uploads/')
    
    # Status tracks where Celery is in the background
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('PROCESSING', 'Processing'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]
    status = models.CharField(max_length=20, default='PENDING', choices=STATUS_CHOICES)
    
    # Celery task tracking
    celery_task_id = models.CharField(max_length=255, blank=True, null=True)
    progress_percentage = models.IntegerField(default=0)
    error_message = models.TextField(blank=True, null=True)
    
    # Analytics for the dashboard
    total_records = models.IntegerField(default=0)
    churn_detected_count = models.IntegerField(default=0)

    def __str__(self):
        return f"Batch {self.id} - {self.status}"