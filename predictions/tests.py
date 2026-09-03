from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from .models import PredictionRecord, BatchUploadJob, ModelVersion
from .services import ChurnModelService
from datetime import datetime, timezone
import io


class ChurnPredictionTests(TestCase):
    def setUp(self):
        self.client = Client()
        # Seed a test model version in the database
        self.test_model = ModelVersion.objects.create(
            name="Test XGBoost (class_weight)",
            algorithm="XGBoost",
            version="xgboost_test_v1",
            model_file="ml_engine/saved_models/churn_pipeline.joblib",
            is_active=True,
            trained_at=datetime.now(timezone.utc),
            dataset_hash="test_sha256_hash",
            dataset_rows=7043,
            accuracy=0.78,
            roc_auc=0.8513,
            pr_auc=0.6828,
            f1_churn=0.6316,
            recall_churn=0.7239,
            imbalance_strategy="class_weight"
        )

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
        # Verify factors exist (either SHAP or fallback)
        self.assertGreater(len(result['risk_factors']), 0)

    def test_predict_api_endpoint(self):
        """Test POST /api/predict/ returns 200 and records prediction with model tracking."""
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
        self.assertIn('model_name', data)
        
        # Verify saved in database with model_version link
        record = PredictionRecord.objects.filter(customer_id='TEST-USER-99').first()
        self.assertIsNotNone(record)
        self.assertEqual(record.model_version, self.test_model)

    def test_batch_predict_and_status_polling(self):
        """Test POST /api/batch-predict/ and polling GET /api/batch-status/<id>/."""
        csv_content = (
            "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\n"
            "BATCH-01,Female,0,No,No,2,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,75.0,150.0\n"
            "BATCH-02,Male,0,Yes,Yes,60,Yes,Yes,DSL,Yes,Yes,Yes,Yes,No,No,Two year,No,Credit card (automatic),40.0,2400.0\n"
        )
        file = SimpleUploadedFile("batch_test.csv", csv_content.encode('utf-8'), content_type="text/csv")
        
        # Submit batch
        response = self.client.post(
            reverse('api-batch-predict'),
            {'file': file}
        )
        self.assertIn(response.status_code, [200, 202])
        data = response.json()
        self.assertIn('job_id', data)
        self.assertIn('status', data)

        job_id = data['job_id']

        # Poll status endpoint
        status_resp = self.client.get(reverse('api-batch-status', kwargs={'job_id': job_id}))
        self.assertEqual(status_resp.status_code, 200)
        status_data = status_resp.json()
        self.assertEqual(status_data['id'], job_id)
        self.assertIn('status', status_data)
        self.assertIn('progress_percentage', status_data)

    def test_model_registry_api(self):
        """Test GET /api/models/ returns list of registered models with benchmark metrics."""
        response = self.client.get(reverse('api-models'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(len(data), 1)
        first_model = data[0]
        self.assertIn('name', first_model)
        self.assertIn('algorithm', first_model)
        self.assertIn('roc_auc', first_model)
        self.assertIn('recall_churn', first_model)

    def test_model_activation_api(self):
        """Test POST /api/models/<id>/activate/ switches active production model."""
        second_model = ModelVersion.objects.create(
            name="Test LightGBM",
            algorithm="LightGBM",
            version="lgbm_test_v1",
            model_file="ml_engine/saved_models/churn_pipeline.joblib",
            is_active=False,
            trained_at=datetime.now(timezone.utc),
            dataset_hash="test_sha256_hash_2",
            dataset_rows=7043,
            roc_auc=0.8493
        )

        response = self.client.post(reverse('api-models-activate', kwargs={'pk': second_model.id}))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['active_model']['is_active'])

        # Verify in DB
        second_model.refresh_from_db()
        self.test_model.refresh_from_db()
        self.assertTrue(second_model.is_active)
        self.assertFalse(self.test_model.is_active)

    def test_dashboard_view(self):
        """Test GET / renders dashboard HTML template with active model in context."""
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'predictions/dashboard.html')
        self.assertContains(response, 'ChurnGuard')
        self.assertIn('active_model', response.context)

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
