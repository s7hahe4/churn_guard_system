import io
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.generic import TemplateView
from django.db.models import Avg, Count, Q
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .serializers import CustomerDataSerializer, PredictionRecordSerializer
from .services import ChurnModelService
from .models import PredictionRecord, BatchUploadJob


class DashboardView(TemplateView):
    """Renders the interactive ChurnGuard AI dashboard."""
    template_name = 'predictions/dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
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
        return context


class PredictChurnView(APIView):
    """API endpoint for single customer churn prediction."""
    def post(self, request):
        serializer = CustomerDataSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({"errors": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)
        
        input_data = serializer.validated_data
        
        try:
            prediction_result = ChurnModelService.predict(input_data)
            
            # Save record in database
            record = PredictionRecord.objects.create(
                customer_id=input_data.get('customerID', 'Anonymous'),
                tenure=prediction_result['tenure'],
                monthly_charges=prediction_result['monthly_charges'],
                contract_type=prediction_result['contract'],
                churn_probability=prediction_result['churn_probability'],
                risk_level=prediction_result['risk_level'],
                churn_predicted=prediction_result['churn_predicted']
            )
            
            response_data = {
                "id": record.id,
                "customer_id": record.customer_id,
                **prediction_result
            }
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class BatchPredictView(APIView):
    """API endpoint to upload CSV and run bulk churn predictions."""
    def post(self, request):
        if 'file' not in request.FILES:
            return Response({"error": "No CSV file provided."}, status=status.HTTP_400_BAD_REQUEST)
        
        uploaded_file = request.FILES['file']
        if not uploaded_file.name.endswith(('.csv', '.CSV')):
            return Response({"error": "Invalid file type. Please upload a .csv file."}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Read CSV into pandas DataFrame
            content = uploaded_file.read().decode('utf-8', errors='ignore')
            df = pd.read_csv(io.StringIO(content))
            
            if df.empty:
                return Response({"error": "Uploaded CSV file is empty."}, status=status.HTTP_400_BAD_REQUEST)

            # Clean column names (strip whitespace)
            df.columns = df.columns.str.strip()

            # Execute batch predictions
            results = ChurnModelService.predict_batch(df)
            
            # Bulk create PredictionRecords
            records_to_create = [
                PredictionRecord(
                    customer_id=r['customer_id'],
                    tenure=r['tenure'],
                    monthly_charges=r['monthly_charges'],
                    contract_type=r['contract_type'],
                    churn_probability=r['churn_probability'],
                    risk_level=r['risk_level'],
                    churn_predicted=r['churn_predicted']
                )
                for r in results
            ]
            PredictionRecord.objects.bulk_create(records_to_create)

            # Record Batch Job
            churn_detected = sum(1 for r in results if r['churn_predicted'])
            job = BatchUploadJob.objects.create(
                status='COMPLETED',
                total_records=len(results),
                churn_detected_count=churn_detected
            )

            high_count = sum(1 for r in results if r['risk_level'] == 'High')
            med_count = sum(1 for r in results if r['risk_level'] == 'Medium')
            low_count = sum(1 for r in results if r['risk_level'] == 'Low')

            return Response({
                "job_id": job.id,
                "total_records": len(results),
                "churn_detected": churn_detected,
                "churn_rate": round((churn_detected / len(results)) * 100, 1),
                "high_risk_count": high_count,
                "medium_risk_count": med_count,
                "low_risk_count": low_count,
                "preview": results[:20]  # First 20 results for preview
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": f"Error processing batch: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PredictionHistoryView(APIView):
    """Returns recent prediction records with filtering."""
    def get(self, request):
        risk = request.query_params.get('risk')
        search = request.query_params.get('search')
        limit = int(request.query_params.get('limit', 50))
        
        queryset = PredictionRecord.objects.order_by('-created_at')
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
                "avg_charges": 0
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