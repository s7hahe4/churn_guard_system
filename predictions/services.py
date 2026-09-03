"""
ChurnGuard AI — ML Inference & Explainability Service

Singleton service that loads the active model from the registry, runs predictions,
and computes per-prediction SHAP feature attributions.

Design Decisions:
    - Lazy-loaded singleton: The model, SHAP explainer, and feature names are loaded
      once into class-level variables on first use. Avoids 20MB disk read per request.
    - Model registry integration: get_model() checks which ModelVersion is marked
      'is_active' in the database. On model switch, cached artifacts are invalidated.
    - SHAP fallback: If SHAP artifacts are missing (e.g., old model), falls back to
      the original rule-based risk factor analysis. Backward compatible.
"""

import os
import joblib
import pandas as pd
import numpy as np
from django.conf import settings


class ChurnModelService:
    # Class-level singletons — loaded once, reused across requests
    _model = None
    _active_version_id = None
    _shap_explainer = None
    _shap_background = None
    _feature_names = None

    # Default baseline feature values for Telco dataset
    DEFAULT_FEATURES = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 1,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 70.0,
        'TotalCharges': 70.0
    }

    @classmethod
    def _get_active_model_version(cls):
        """Query the model registry for the active model version."""
        from predictions.models import ModelVersion
        return ModelVersion.objects.filter(is_active=True).first()

    @classmethod
    def get_model(cls):
        """Loads the active model. Invalidates cache if the active model changed."""
        active_version = cls._get_active_model_version()

        if active_version and active_version.id != cls._active_version_id:
            # Active model changed — invalidate everything
            cls._model = None
            cls._shap_explainer = None
            cls._shap_background = None
            cls._feature_names = None
            cls._active_version_id = active_version.id

        if cls._model is None:
            if active_version and active_version.model_file and os.path.exists(active_version.model_file):
                model_path = active_version.model_file
            else:
                # Fallback to legacy path
                model_path = os.path.join(settings.BASE_DIR, 'ml_engine', 'saved_models', 'churn_pipeline.joblib')

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            cls._model = joblib.load(model_path)
            print(f"ML Model loaded: {model_path}")

            # Load SHAP artifacts if available
            cls._load_shap_artifacts(active_version)

        return cls._model

    @classmethod
    def _load_shap_artifacts(cls, model_version):
        """Load SHAP background data and feature names if artifacts exist."""
        try:
            if model_version and model_version.shap_background_file:
                shap_path = model_version.shap_background_file
                features_path = model_version.feature_names_file

                if shap_path and os.path.exists(shap_path):
                    cls._shap_background = joblib.load(shap_path)
                    print(f"SHAP background loaded: {shap_path}")

                if features_path and os.path.exists(features_path):
                    cls._feature_names = joblib.load(features_path)
                    print(f"Feature names loaded: {features_path}")
            else:
                # Try legacy paths
                save_dir = os.path.join(settings.BASE_DIR, 'ml_engine', 'saved_models')
                for f in sorted(os.listdir(save_dir), reverse=True):
                    if f.startswith('shap_background_') and f.endswith('.joblib'):
                        cls._shap_background = joblib.load(os.path.join(save_dir, f))
                        print(f"SHAP background loaded (legacy scan): {f}")
                        break
                for f in sorted(os.listdir(save_dir), reverse=True):
                    if f.startswith('feature_names_') and f.endswith('.joblib'):
                        cls._feature_names = joblib.load(os.path.join(save_dir, f))
                        print(f"Feature names loaded (legacy scan): {f}")
                        break
        except Exception as e:
            print(f"Warning: Could not load SHAP artifacts: {e}")
            cls._shap_background = None
            cls._feature_names = None

    @classmethod
    def get_active_model_info(cls):
        """Returns metadata about the currently active model for display."""
        version = cls._get_active_model_version()
        if version:
            return {
                'name': version.name,
                'algorithm': version.algorithm,
                'version': version.version,
                'roc_auc': round(version.roc_auc, 4),
                'f1_churn': round(version.f1_churn, 4),
                'recall_churn': round(version.recall_churn, 4),
                'trained_at': version.trained_at.strftime('%Y-%m-%d %H:%M') if version.trained_at else '',
                'is_active': True,
            }
        return {
            'name': 'RandomForest v1.0 (Legacy)',
            'algorithm': 'RandomForest',
            'version': 'legacy',
            'roc_auc': 0,
            'is_active': True,
        }

    @classmethod
    def prepare_input(cls, input_data: dict) -> pd.DataFrame:
        """Prepares a clean dataframe row with all expected features filled in."""
        full_data = cls.DEFAULT_FEATURES.copy()
        
        # Merge input data
        for k, v in input_data.items():
            if k in full_data and v is not None and v != "":
                # Convert types appropriately
                if k in ['tenure', 'SeniorCitizen']:
                    try:
                        full_data[k] = int(v)
                    except (ValueError, TypeError):
                        pass
                elif k in ['MonthlyCharges', 'TotalCharges']:
                    try:
                        full_data[k] = float(v)
                    except (ValueError, TypeError):
                        pass
                else:
                    full_data[k] = str(v)

        # Smart fallback for TotalCharges if omitted or 0
        if 'TotalCharges' not in input_data or input_data.get('TotalCharges') in [0, 0.0, None, '']:
            full_data['TotalCharges'] = round(float(full_data['tenure']) * float(full_data['MonthlyCharges']), 2)

        return pd.DataFrame([full_data])

    @classmethod
    def explain_prediction(cls, input_data: dict, prob: float) -> list:
        """
        Compute real SHAP feature attributions for a single prediction.
        
        Returns a list of the top contributing features with their SHAP values
        and direction (positive = pushes toward churn, negative = pushes toward retention).
        Falls back to rule-based heuristics if SHAP artifacts are unavailable.
        """
        model = cls.get_model()

        # If SHAP artifacts aren't available, fall back to heuristics
        if cls._shap_background is None or cls._feature_names is None:
            return cls.analyze_risk_factors(input_data, prob)

        try:
            import shap

            # Transform the input through the preprocessor
            df = cls.prepare_input(input_data)
            preprocessor = model.named_steps['preprocessor']
            classifier = model.named_steps['classifier']

            X_transformed = preprocessor.transform(df)
            if hasattr(X_transformed, 'toarray'):
                X_transformed = X_transformed.toarray()

            # Create TreeExplainer (tree_path_dependent works across RF, XGBoost, and LightGBM)
            if cls._shap_explainer is None:
                try:
                    cls._shap_explainer = shap.TreeExplainer(classifier, feature_perturbation='tree_path_dependent')
                except Exception:
                    cls._shap_explainer = shap.TreeExplainer(classifier, cls._shap_background)

            # Compute SHAP values for this prediction
            shap_values = cls._shap_explainer.shap_values(X_transformed)

            # For binary classification, extract the churn (positive) class values
            if isinstance(shap_values, list):
                # e.g. RandomForest returns [class_0_vals, class_1_vals]
                shap_vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
                # e.g. shape (1, n_features, 2)
                shap_vals = shap_values[0, :, 1]
            elif hasattr(shap_values, 'ndim') and shap_values.ndim == 2:
                # e.g. XGBoost shape (1, n_features)
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values

            # Map SHAP values to human-readable feature names
            feature_contributions = []
            for i, (name, val) in enumerate(zip(cls._feature_names, shap_vals)):
                # Clean up one-hot encoded names: "cat__Contract_Month-to-month" → "Contract: Month-to-month"
                display_name = cls._humanize_feature_name(name)
                feature_contributions.append({
                    'feature': display_name,
                    'shap_value': round(float(val), 4),
                    'direction': 'increases_churn' if val > 0 else 'decreases_churn',
                    'abs_impact': abs(float(val)),
                })

            # Sort by absolute impact, take top 8
            feature_contributions.sort(key=lambda x: x['abs_impact'], reverse=True)
            top_factors = feature_contributions[:8]

            # Format for frontend
            shap_factors = []
            for f in top_factors:
                if f['abs_impact'] < 0.001:
                    continue  # Skip negligible factors
                impact_label = "Increases Risk" if f['direction'] == 'increases_churn' else "Reduces Risk"
                shap_factors.append({
                    'factor': f['feature'],
                    'impact': impact_label,
                    'shap_value': f['shap_value'],
                    'detail': cls._get_shap_advice(f['feature'], f['direction']),
                    'is_shap': True,
                })

            return shap_factors if shap_factors else cls.analyze_risk_factors(input_data, prob)

        except Exception as e:
            print(f"SHAP computation failed, falling back to heuristics: {e}")
            return cls.analyze_risk_factors(input_data, prob)

    @classmethod
    def _humanize_feature_name(cls, raw_name: str) -> str:
        """Convert sklearn feature names to human-readable labels."""
        if raw_name.startswith('cat__'):
            raw_name = raw_name[5:]
        elif raw_name.startswith('num__'):
            raw_name = raw_name[5:]

        name_map = {
            'tenure': 'Tenure (Months)',
            'MonthlyCharges': 'Monthly Charges ($)',
            'TotalCharges': 'Total Charges ($)',
        }
        if raw_name in name_map:
            return name_map[raw_name]

        # Check for categorical features e.g. "Contract_Month-to-month"
        for cat in cls.DEFAULT_FEATURES.keys():
            if raw_name.startswith(f"{cat}_"):
                val = raw_name[len(cat)+1:]
                return f"{cat}: {val}"

        return raw_name

    @classmethod
    def _get_shap_advice(cls, feature: str, direction: str) -> str:
        """Return actionable advice based on SHAP feature and direction."""
        advice_map = {
            'Contract: Month-to-month': 'Month-to-month contracts have the highest churn rate. Recommend offering a 1-year discount.',
            'Contract: Two year': 'Two-year contracts strongly reduce churn. This is a positive retention signal.',
            'Contract: One year': 'One-year contracts moderately reduce churn compared to month-to-month.',
            'Tenure (Months)': 'Early-lifecycle customers need active check-in campaigns.',
            'Monthly Charges': 'High monthly bills increase churn risk. Consider bundle optimizations or loyalty discounts.',
            'Total Charges': 'Total charges reflect overall customer lifetime value.',
            'TechSupport: No': 'Subscribers without Tech Support churn more when experiencing issues.',
            'TechSupport: Yes': 'Tech Support is a strong retention driver.',
            'OnlineSecurity: No': 'Adding Online Security increases customer stickiness.',
            'OnlineSecurity: Yes': 'Online Security is a positive retention signal.',
            'InternetService: Fiber optic': 'Fiber optic customers have higher churn — possibly due to higher competition and pricing.',
            'InternetService: DSL': 'DSL customers tend to have lower churn rates.',
            'PaymentMethod: Electronic check': 'Electronic check users have higher churn; suggest switching to auto-debit.',
            'PaperlessBilling: Yes': 'Paperless billing correlates with slightly higher churn.',
        }
        default_msg = 'This feature has a measurable impact on the churn prediction.'
        if direction == 'decreases_churn':
            default_msg = 'This feature is helping retain this customer.'
        return advice_map.get(feature, default_msg)

    @classmethod
    def analyze_risk_factors(cls, data: dict, prob: float) -> list:
        """Legacy rule-based risk factor analysis. Used as fallback when SHAP is unavailable."""
        factors = []
        
        contract = data.get('Contract', 'Month-to-month')
        if contract == 'Month-to-month':
            factors.append({
                'factor': 'Contract Type',
                'impact': 'High Risk',
                'detail': 'Month-to-month contracts have the highest churn rate. Recommend offering a 1-year discount.',
                'is_shap': False,
            })
        
        tenure = int(data.get('tenure', 1))
        if tenure <= 6:
            factors.append({
                'factor': 'New Customer Onboarding',
                'impact': 'High Risk',
                'detail': f'Tenure is only {tenure} month(s). Early lifecycle customers need active check-in campaigns.',
                'is_shap': False,
            })
        
        monthly = float(data.get('MonthlyCharges', 70.0))
        if monthly >= 80.0:
            factors.append({
                'factor': 'High Monthly Bill',
                'impact': 'Medium Risk',
                'detail': f'Monthly bill of ${monthly:.2f} is above average. Bundle optimizations or loyalty discounts can help.',
                'is_shap': False,
            })
            
        tech_support = data.get('TechSupport', 'No')
        if tech_support == 'No':
            factors.append({
                'factor': 'No Tech Support',
                'impact': 'Medium Risk',
                'detail': 'Subscribers without Tech Support churn more frequently when experiencing technical issues.',
                'is_shap': False,
            })

        online_sec = data.get('OnlineSecurity', 'No')
        if online_sec == 'No':
            factors.append({
                'factor': 'No Online Security',
                'impact': 'Low Risk',
                'detail': 'Adding Online Security increases customer stickiness and retention.',
                'is_shap': False,
            })

        payment = data.get('PaymentMethod', '')
        if 'Electronic check' in payment:
            factors.append({
                'factor': 'Payment Method',
                'impact': 'Low Risk',
                'detail': 'Electronic check users have higher churn; suggest switching to auto-debit / credit card.',
                'is_shap': False,
            })

        return factors

    @classmethod
    def predict(cls, input_data: dict):
        """Takes a dictionary of customer data, predicts churn, and returns formatted results."""
        model = cls.get_model()
        df = cls.prepare_input(input_data)
        
        prob = model.predict_proba(df)[0][1]
        is_churn = bool(prob >= 0.5)
        
        if prob >= 0.70:
            risk = "High"
            badge_color = "danger"
        elif prob >= 0.35:
            risk = "Medium"
            badge_color = "warning"
        else:
            risk = "Low"
            badge_color = "success"
            
        # Use SHAP-based explanation (falls back to heuristics if unavailable)
        risk_factors = cls.explain_prediction(input_data, prob)

        return {
            "churn_predicted": is_churn,
            "churn_probability": round(float(prob), 4),
            "churn_percentage": round(float(prob) * 100, 1),
            "risk_level": risk,
            "badge_color": badge_color,
            "risk_factors": risk_factors,
            "tenure": int(df['tenure'].iloc[0]),
            "monthly_charges": float(df['MonthlyCharges'].iloc[0]),
            "contract": str(df['Contract'].iloc[0]),
        }

    @classmethod
    def predict_batch(cls, df: pd.DataFrame):
        """Processes a pandas DataFrame with multiple customer records."""
        model = cls.get_model()
        
        processed_rows = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            full_data = cls.DEFAULT_FEATURES.copy()
            for k, v in row_dict.items():
                if k in full_data and pd.notna(v):
                    if k in ['tenure', 'SeniorCitizen']:
                        try:
                            full_data[k] = int(float(v))
                        except (ValueError, TypeError):
                            pass
                    elif k in ['MonthlyCharges', 'TotalCharges']:
                        try:
                            full_data[k] = float(v)
                        except (ValueError, TypeError):
                            pass
                    else:
                        full_data[k] = str(v)
            if 'TotalCharges' not in row_dict or pd.isna(row_dict.get('TotalCharges')):
                full_data['TotalCharges'] = round(float(full_data['tenure']) * float(full_data['MonthlyCharges']), 2)
            processed_rows.append(full_data)

        eval_df = pd.DataFrame(processed_rows)
        probabilities = model.predict_proba(eval_df)[:, 1]
        
        results = []
        for i, prob in enumerate(probabilities):
            is_churn = bool(prob >= 0.5)
            if prob >= 0.70:
                risk = "High"
            elif prob >= 0.35:
                risk = "Medium"
            else:
                risk = "Low"
            
            cust_id = df.iloc[i].get('customerID', f"CUST-{i+1:04d}")
            results.append({
                "customer_id": str(cust_id) if pd.notna(cust_id) else f"CUST-{i+1:04d}",
                "tenure": int(eval_df['tenure'].iloc[i]),
                "monthly_charges": float(eval_df['MonthlyCharges'].iloc[i]),
                "contract_type": str(eval_df['Contract'].iloc[i]),
                "churn_probability": round(float(prob), 4),
                "churn_percentage": round(float(prob) * 100, 1),
                "risk_level": risk,
                "churn_predicted": is_churn
            })
            
        return results