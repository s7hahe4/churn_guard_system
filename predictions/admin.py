from django.contrib import admin
from .models import ModelVersion, PredictionRecord, BatchUploadJob


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ['name', 'algorithm', 'version', 'is_active', 'roc_auc', 'f1_churn', 'recall_churn', 'imbalance_strategy', 'trained_at']
    list_filter = ['algorithm', 'is_active', 'imbalance_strategy']
    ordering = ['-trained_at']
    readonly_fields = ['trained_at', 'dataset_hash', 'dataset_rows']

    actions = ['activate_model']

    @admin.action(description="Set selected model as active (deactivates others)")
    def activate_model(self, request, queryset):
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one model to activate.", level='error')
            return
        ModelVersion.objects.update(is_active=False)
        queryset.update(is_active=True)
        self.message_user(request, f"Activated: {queryset.first().name}")


@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):
    list_display = ['customer_id', 'risk_level', 'churn_probability', 'contract_type', 'tenure', 'monthly_charges', 'created_at']
    list_filter = ['risk_level', 'churn_predicted', 'contract_type']
    search_fields = ['customer_id']
    ordering = ['-created_at']


@admin.register(BatchUploadJob)
class BatchUploadJobAdmin(admin.ModelAdmin):
    list_display = ['id', 'status', 'total_records', 'churn_detected_count', 'progress_percentage', 'uploaded_at']
    list_filter = ['status']
    ordering = ['-uploaded_at']
