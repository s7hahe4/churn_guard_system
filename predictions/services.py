import os
import joblib
import pandas as pd
import numpy as np
from django.conf import settings

class ChurnModelService:
    # This class variable will hold our model in memory
    _model = None

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
    def get_model(cls):
        """Loads the model only if it hasn't been loaded yet."""
        if cls._model is None:
            model_path = os.path.join(settings.BASE_DIR, 'ml_engine', 'saved_models', 'churn_pipeline.joblib')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            cls._model = joblib.load(model_path)
            print("ML Model loaded into memory successfully!")
        return cls._model

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
    def analyze_risk_factors(cls, data: dict, prob: float) -> list:
        """Identifies key drivers contributing to churn risk and retention advice."""
        factors = []
        
        contract = data.get('Contract', 'Month-to-month')
        if contract == 'Month-to-month':
            factors.append({
                'factor': 'Contract Type',
                'impact': 'High Risk',
                'detail': 'Month-to-month contracts have the highest churn rate. Recommend offering a 1-year discount.'
            })
        
        tenure = int(data.get('tenure', 1))
        if tenure <= 6:
            factors.append({
                'factor': 'New Customer Onboarding',
                'impact': 'High Risk',
                'detail': f'Tenure is only {tenure} month(s). Early lifecycle customers need active check-in campaigns.'
            })
        
        monthly = float(data.get('MonthlyCharges', 70.0))
        if monthly >= 80.0:
            factors.append({
                'factor': 'High Monthly Bill',
                'impact': 'Medium Risk',
                'detail': f'Monthly bill of ${monthly:.2f} is above average. Bundle optimizations or loyalty discounts can help.'
            })
            
        tech_support = data.get('TechSupport', 'No')
        if tech_support == 'No':
            factors.append({
                'factor': 'No Tech Support',
                'impact': 'Medium Risk',
                'detail': 'Subscribers without Tech Support churn more frequently when experiencing technical issues.'
            })

        online_sec = data.get('OnlineSecurity', 'No')
        if online_sec == 'No':
            factors.append({
                'factor': 'No Online Security',
                'impact': 'Low Risk',
                'detail': 'Adding Online Security increases customer stickiness and retention.'
            })

        payment = data.get('PaymentMethod', '')
        if 'Electronic check' in payment:
            factors.append({
                'factor': 'Payment Method',
                'impact': 'Low Risk',
                'detail': 'Electronic check users have higher churn; suggest switching to auto-debit / credit card.'
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
            
        risk_factors = cls.analyze_risk_factors(input_data, prob)

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