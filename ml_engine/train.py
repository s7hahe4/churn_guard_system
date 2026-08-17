import pandas as pd
import numpy as np
import joblib # Added for saving the model
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline # Added to build the assembly line
from sklearn.ensemble import RandomForestClassifier # The ML Brain
from sklearn.metrics import classification_report # For grading the model

# 1. Load the data
print("Loading data...")
df = pd.read_csv('ml_engine/data/Telco-Customer-Churn.csv')

# 2. Clean the data
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan))
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# 3. Define Features (X) and Target (y)
X = df.drop(columns=['Churn', 'customerID'])
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Build Preprocessor
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 6. Create the Full Pipeline (Preprocessor + Model)
print("Building and training the model (this might take a few seconds)...")
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 7. Train the model!
pipeline.fit(X_train, y_train)

# 8. Grade the model
print("\n--- Model Evaluation ---")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Stayed (0)', 'Churned (1)']))

# 9. Save the pipeline for Django
export_path = 'ml_engine/saved_models/churn_pipeline.joblib'
joblib.dump(pipeline, export_path)
print(f"\nSuccess! Production Pipeline saved to: {export_path}")