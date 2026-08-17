from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from .models import PredictionRecord, BatchUploadJob
from .services import ChurnModelService
import io

class ChurnPredictionTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_churn_model_service_prediction(self):
        """Test that ChurnModelService accurately generates predictions with risk analysis."""
        sample_input = {
            'customerID': 'TEST-001',
            'tenure': 2,
            'MonthlyCharges': 85.0,
            'Contract': 'Month-to-month'
        }
        result = ChurnModelService.predict(sample_input)
        self.assertIn('churn_predicted', result)
        self.assertIn('churn_probability', result)
        self.assertIn('risk_level', result)
        self.assertIn('risk_factors', result)
        self.assertIsInstance(result['churn_probability'], float)
        self.assertGreaterEqual(result['churn_probability'], 0.0)
        self.assertLessEqual(result['churn_probability'], 1.0)

    def test_predict_api_endpoint(self):
        """Test POST /api/predict/ returns 200 and records the prediction."""
        payload = {
            'customerID': 'TEST-USER-99',
            'tenure': 12,
            'MonthlyCharges': 70.0,
            'Contract': 'Month-to-month',
            'InternetService': 'Fiber optic'
        }
        response = self.client.post(
            reverse('api-predict'),
            data=payload,
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['customer_id'], 'TEST-USER-99')
        self.assertIn('churn_probability', data)
        self.assertIn('risk_level', data)
        
        # Verify saved in database
        self.assertTrue(PredictionRecord.objects.filter(customer_id='TEST-USER-99').exists())

    def test_batch_predict_api_endpoint(self):
        """Test POST /api/batch-predict/ with a valid CSV file."""
        csv_content = (
            "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\n"
            "BATCH-01,Female,0,No,No,2,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,75.0,150.0\n"
            "BATCH-02,Male,0,Yes,Yes,60,Yes,Yes,DSL,Yes,Yes,Yes,Yes,No,No,Two year,No,Credit card (automatic),40.0,2400.0\n"
        )
        file = SimpleUploadedFile("batch_test.csv", csv_content.encode('utf-8'), content_type="text/csv")
        
        response = self.client.post(
            reverse('api-batch-predict'),
            {'file': file}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['total_records'], 2)
        self.assertIn('churn_rate', data)
        self.assertEqual(len(data['preview']), 2)

    def test_dashboard_view(self):
        """Test GET / renders dashboard HTML template."""
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'predictions/dashboard.html')
        self.assertContains(response, 'ChurnGuard')

    def test_history_and_stats_api(self):
        """Test stats and history API endpoints."""
        PredictionRecord.objects.create(
            customer_id='HIST-01',
            tenure=5,
            monthly_charges=50.0,
            contract_type='Month-to-month',
            churn_probability=0.85,
            risk_level='High',
            churn_predicted=True
        )

        stats_resp = self.client.get(reverse('api-stats'))
        self.assertEqual(stats_resp.status_code, 200)
        stats_data = stats_resp.json()
        self.assertGreaterEqual(stats_data['total'], 1)
        self.assertGreaterEqual(stats_data['high_risk'], 1)

        hist_resp = self.client.get(reverse('api-history'))
        self.assertEqual(hist_resp.status_code, 200)
        hist_data = hist_resp.json()
        self.assertTrue(any(r['customer_id'] == 'HIST-01' for r in hist_data))

    def test_sample_csv_download(self):
        """Test GET /api/sample-csv/ returns CSV file attachment."""
        response = self.client.get(reverse('api-sample-csv'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/csv')
        self.assertIn('attachment;', response['Content-Disposition'])
