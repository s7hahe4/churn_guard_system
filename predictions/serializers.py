from rest_framework import serializers
from .models import PredictionRecord, BatchUploadJob, ModelVersion


class CustomerDataSerializer(serializers.Serializer):
    customerID = serializers.CharField(max_length=50, required=False, default="Anonymous")
    tenure = serializers.IntegerField(required=False, default=1, min_value=0, max_value=120)
    MonthlyCharges = serializers.FloatField(required=False, default=70.0, min_value=0.0)
    TotalCharges = serializers.FloatField(required=False, allow_null=True)
    Contract = serializers.CharField(max_length=50, required=False, default="Month-to-month")
    
    gender = serializers.CharField(max_length=10, required=False, default="Female")
    SeniorCitizen = serializers.IntegerField(required=False, default=0)
    Partner = serializers.CharField(max_length=5, required=False, default="No")
    Dependents = serializers.CharField(max_length=5, required=False, default="No")
    PhoneService = serializers.CharField(max_length=5, required=False, default="Yes")
    MultipleLines = serializers.CharField(max_length=20, required=False, default="No")
    InternetService = serializers.CharField(max_length=20, required=False, default="Fiber optic")
    OnlineSecurity = serializers.CharField(max_length=20, required=False, default="No")
    OnlineBackup = serializers.CharField(max_length=20, required=False, default="No")
    DeviceProtection = serializers.CharField(max_length=20, required=False, default="No")
    TechSupport = serializers.CharField(max_length=20, required=False, default="No")
    StreamingTV = serializers.CharField(max_length=20, required=False, default="No")
    StreamingMovies = serializers.CharField(max_length=20, required=False, default="No")
    PaperlessBilling = serializers.CharField(max_length=5, required=False, default="Yes")
    PaymentMethod = serializers.CharField(max_length=50, required=False, default="Electronic check")

    def validate(self, attrs):
        # Auto-compute TotalCharges if not supplied
        if attrs.get('TotalCharges') is None:
            tenure = attrs.get('tenure', 1)
            monthly = attrs.get('MonthlyCharges', 70.0)
            attrs['TotalCharges'] = round(float(tenure) * float(monthly), 2)
        return attrs


class PredictionRecordSerializer(serializers.ModelSerializer):
    created_at_formatted = serializers.DateTimeField(source='created_at', format='%Y-%m-%d %H:%M', read_only=True)
    model_name = serializers.CharField(source='model_version.name', read_only=True, default="Baseline")

    class Meta:
        model = PredictionRecord
        fields = [
            'id',
            'customer_id',
            'tenure',
            'monthly_charges',
            'contract_type',
            'churn_probability',
            'risk_level',
            'churn_predicted',
            'model_version',
            'model_name',
            'created_at',
            'created_at_formatted'
        ]


class ModelVersionSerializer(serializers.ModelSerializer):
    trained_at_formatted = serializers.DateTimeField(source='trained_at', format='%Y-%m-%d %H:%M', read_only=True)

    class Meta:
        model = ModelVersion
        fields = [
            'id',
            'name',
            'algorithm',
            'version',
            'is_active',
            'accuracy',
            'roc_auc',
            'pr_auc',
            'f1_churn',
            'precision_churn',
            'recall_churn',
            'imbalance_strategy',
            'dataset_rows',
            'trained_at_formatted'
        ]


class BatchJobStatusSerializer(serializers.ModelSerializer):
    uploaded_at_formatted = serializers.DateTimeField(source='uploaded_at', format='%Y-%m-%d %H:%M', read_only=True)

    class Meta:
        model = BatchUploadJob
        fields = [
            'id',
            'status',
            'progress_percentage',
            'total_records',
            'churn_detected_count',
            'error_message',
            'uploaded_at_formatted'
        ]