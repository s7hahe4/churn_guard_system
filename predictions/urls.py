from django.urls import path
from .views import (
    DashboardView,
    PredictChurnView,
    BatchPredictView,
    BatchJobStatusView,
    PredictionHistoryView,
    PredictionStatsView,
    ModelRegistryView,
    SetActiveModelView,
    download_sample_csv
)

urlpatterns = [
    path('', DashboardView.as_view(), name='dashboard'),
    path('api/predict/', PredictChurnView.as_view(), name='api-predict'),
    path('api/batch-predict/', BatchPredictView.as_view(), name='api-batch-predict'),
    path('api/batch-status/<int:job_id>/', BatchJobStatusView.as_view(), name='api-batch-status'),
    path('api/history/', PredictionHistoryView.as_view(), name='api-history'),
    path('api/stats/', PredictionStatsView.as_view(), name='api-stats'),
    path('api/models/', ModelRegistryView.as_view(), name='api-models'),
    path('api/models/<int:pk>/activate/', SetActiveModelView.as_view(), name='api-models-activate'),
    path('api/sample-csv/', download_sample_csv, name='api-sample-csv'),
]