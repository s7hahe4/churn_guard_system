"""
ChurnGuard AI — Model Training & Registry Pipeline

Trains churn prediction models, evaluates them on held-out test data, computes
SHAP explainability artifacts, and registers results in the Django model registry.

Usage:
    python ml_engine/train.py                                     # Train RF with defaults
    python ml_engine/train.py --algorithm xgboost --set-active     # Train XGBoost, set as active
    python ml_engine/train.py --algorithm lightgbm --imbalance-strategy class_weight
    python ml_engine/train.py --algorithm all --set-active          # Train all 3 and activate the best

Design Decisions:
    - Why TreeSHAP over LIME: TreeSHAP is exact for tree-based models (RF, XGBoost, LightGBM).
      LIME is model-agnostic but approximate — it perturbs inputs and fits a local linear model,
      which can be noisy and inconsistent between runs. TreeSHAP runs in polynomial time for
      tree ensembles and gives mathematically consistent Shapley values.
    - Why class_weight over SMOTE by default: class_weight='balanced' adjusts the loss function
      without creating synthetic samples, avoiding potential overfitting from SMOTE's interpolation.
      SMOTE is available as an option for comparison.
"""

import os
import sys
import hashlib
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    average_precision_score, f1_score, precision_score, recall_score
)

# Add project root to path for Django imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ──────────────────────────────────────────────
# Feature definitions (must match services.py)
# ──────────────────────────────────────────────
NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]


def compute_dataset_hash(filepath: str) -> str:
    """SHA256 hash of the training CSV for reproducibility tracking."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_and_clean_data(data_path: str):
    """Load Telco dataset and return cleaned X, y."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Clean TotalCharges: blank spaces → NaN → 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan))
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    X = df.drop(columns=['Churn', 'customerID'])
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    return X, y, len(df)


def build_preprocessor():
    """Build the ColumnTransformer for numeric scaling + categorical one-hot encoding."""
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ])


def get_classifier(algorithm: str, imbalance_strategy: str):
    """Return the classifier based on algorithm choice and imbalance strategy."""
    # Determine class_weight parameter
    weight_param = 'balanced' if imbalance_strategy == 'class_weight' else None

    if algorithm == 'RandomForest':
        return RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight=weight_param
        )
    elif algorithm == 'XGBoost':
        from xgboost import XGBClassifier
        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
        }
        if imbalance_strategy == 'class_weight':
            # XGBoost uses scale_pos_weight instead of class_weight
            params['scale_pos_weight'] = 1  # Will be set after computing ratio
        return XGBClassifier(**params)
    elif algorithm == 'LightGBM':
        from lightgbm import LGBMClassifier
        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': -1,
        }
        if imbalance_strategy == 'class_weight':
            params['is_unbalance'] = True
        return LGBMClassifier(**params)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: RandomForest, XGBoost, LightGBM")


def evaluate_model(pipeline, X_test, y_test) -> dict:
    """Evaluate model on test set and return metrics dict."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'roc_auc': round(roc_auc_score(y_test, y_proba), 4),
        'pr_auc': round(average_precision_score(y_test, y_proba), 4),
        'f1_churn': round(f1_score(y_test, y_pred, pos_label=1), 4),
        'precision_churn': round(precision_score(y_test, y_pred, pos_label=1), 4),
        'recall_churn': round(recall_score(y_test, y_pred, pos_label=1), 4),
    }
    return metrics


def compute_shap_artifacts(pipeline, X_train, preprocessor):
    """
    Compute SHAP background data and feature names for TreeExplainer.
    
    Design Decision — Why we precompute background data here:
    SHAP's TreeExplainer needs a reference dataset to compute expected values.
    If we computed this on every Django request, we'd need to transform training
    data every time. By saving a small transformed sample (100 rows), Django
    loads it once and reuses it for all predictions — milliseconds vs seconds.
    """
    print("Computing SHAP background data...")

    # Get feature names after one-hot encoding
    preprocessor_fitted = pipeline.named_steps['preprocessor']
    feature_names = list(NUMERIC_FEATURES)
    cat_encoder = preprocessor_fitted.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(cat_feature_names)

    # Transform a small sample of training data as SHAP background
    background_sample = X_train.sample(n=min(100, len(X_train)), random_state=42)
    background_transformed = preprocessor_fitted.transform(background_sample)

    # Convert sparse matrix to dense if needed
    if hasattr(background_transformed, 'toarray'):
        background_transformed = background_transformed.toarray()

    return background_transformed, feature_names


def apply_smote(X_train, y_train):
    """Apply SMOTE oversampling to balance classes."""
    from imblearn.over_sampling import SMOTE
    print("Applying SMOTE oversampling...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"  Before SMOTE: {len(X_train)} samples | After: {len(X_resampled)} samples")
    return X_resampled, y_resampled


def train_and_register(algorithm: str, imbalance_strategy: str, set_active: bool = False):
    """
    Full training pipeline: load data → preprocess → train → evaluate → save artifacts → register.
    """
    data_path = os.path.join(PROJECT_ROOT, 'ml_engine', 'data', 'Telco-Customer-Churn.csv')
    save_dir = os.path.join(PROJECT_ROOT, 'ml_engine', 'saved_models')
    os.makedirs(save_dir, exist_ok=True)

    # ── Load and split ──
    X, y, total_rows = load_and_clean_data(data_path)
    dataset_hash = compute_dataset_hash(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ── Apply SMOTE if requested (before pipeline fitting) ──
    # Note: SMOTE must be applied AFTER splitting but BEFORE fitting the pipeline,
    # and it operates on numeric data. We'll preprocess first, then SMOTE, then fit classifier.
    if imbalance_strategy == 'smote':
        # Build preprocessor, fit-transform training data, then SMOTE, then fit classifier
        preprocessor = build_preprocessor()
        X_train_transformed = preprocessor.fit_transform(X_train)
        if hasattr(X_train_transformed, 'toarray'):
            X_train_transformed = X_train_transformed.toarray()

        X_train_resampled, y_train_resampled = apply_smote(X_train_transformed, y_train)

        # Build classifier and train on resampled data
        classifier = get_classifier(algorithm, imbalance_strategy='none')  # no class_weight with SMOTE
        print(f"\nTraining {algorithm} with SMOTE oversampling...")
        classifier.fit(X_train_resampled, y_train_resampled)

        # Build a pipeline for prediction (preprocessor is already fitted)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        # The preprocessor is already fitted, and the classifier is already fitted.
        # We need a complete pipeline for prediction. Since sklearn Pipeline.predict
        # calls transform then predict, and our preprocessor is fitted, this works.
        # But we need to "re-fit" the pipeline so it's in a consistent state.
        # The cleanest approach: just build a new pipeline and set its components.
        pipeline.named_steps['preprocessor'] = preprocessor
        pipeline.named_steps['classifier'] = classifier

    else:
        # Standard pipeline flow (no SMOTE)
        preprocessor = build_preprocessor()
        classifier = get_classifier(algorithm, imbalance_strategy)

        # Handle XGBoost scale_pos_weight
        if algorithm == 'XGBoost' and imbalance_strategy == 'class_weight':
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            classifier.set_params(scale_pos_weight=neg_count / pos_count)

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])

        print(f"\nTraining {algorithm} (imbalance: {imbalance_strategy})...")
        pipeline.fit(X_train, y_train)

    # ── Evaluate ──
    print("\n--- Model Evaluation ---")
    metrics = evaluate_model(pipeline, X_test, y_test)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Stayed (0)', 'Churned (1)']))
    print(f"  ROC-AUC:    {metrics['roc_auc']}")
    print(f"  PR-AUC:     {metrics['pr_auc']}")
    print(f"  F1 (Churn): {metrics['f1_churn']}")
    print(f"  Recall:     {metrics['recall_churn']}")

    # ── Compute SHAP artifacts ──
    shap_background, feature_names = compute_shap_artifacts(pipeline, X_train, preprocessor)

    # ── Generate version string ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    version_tag = f"{algorithm.lower()}_{imbalance_strategy}_{timestamp}"

    # ── Save artifacts ──
    pipeline_path = os.path.join(save_dir, f'churn_pipeline_{version_tag}.joblib')
    shap_path = os.path.join(save_dir, f'shap_background_{version_tag}.joblib')
    features_path = os.path.join(save_dir, f'feature_names_{version_tag}.joblib')

    joblib.dump(pipeline, pipeline_path)
    joblib.dump(shap_background, shap_path)
    joblib.dump(feature_names, features_path)

    print(f"\n  Pipeline saved:       {pipeline_path}")
    print(f"  SHAP background saved: {shap_path}")
    print(f"  Feature names saved:   {features_path}")

    # Also save as the "latest" for backward compatibility
    latest_pipeline = os.path.join(save_dir, 'churn_pipeline.joblib')
    joblib.dump(pipeline, latest_pipeline)

    # ── Register in Django database ──
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
    import django
    django.setup()
    from predictions.models import ModelVersion

    # Deactivate all if we're setting this one as active
    if set_active:
        ModelVersion.objects.update(is_active=False)

    model_version = ModelVersion.objects.create(
        name=f"{algorithm} ({imbalance_strategy})",
        algorithm=algorithm,
        version=version_tag,
        model_file=pipeline_path,
        shap_background_file=shap_path,
        feature_names_file=features_path,
        is_active=set_active,
        trained_at=datetime.now(timezone.utc),
        dataset_hash=dataset_hash,
        dataset_rows=total_rows,
        imbalance_strategy=imbalance_strategy,
        **metrics
    )

    status = "[ACTIVE]" if set_active else "[registered]"
    print(f"\n[OK] Model registered in database: {model_version.name} {status}")

    # Print comparison table
    all_models = ModelVersion.objects.all()
    if all_models.count() > 1:
        print("\n--- Model Registry ---")
        print(f"{'Name':<35} {'ROC-AUC':>8} {'PR-AUC':>8} {'F1':>8} {'Recall':>8} {'Active':>7}")
        print("-" * 80)
        for m in all_models:
            active = "  YES" if m.is_active else ""
            print(f"{m.name:<35} {m.roc_auc:>8.4f} {m.pr_auc:>8.4f} {m.f1_churn:>8.4f} {m.recall_churn:>8.4f} {active:>7}")

    return model_version, metrics


def main():
    parser = argparse.ArgumentParser(description='ChurnGuard AI — Model Training Pipeline')
    parser.add_argument('--algorithm', type=str, default='RandomForest',
                        choices=['RandomForest', 'XGBoost', 'LightGBM', 'all'],
                        help='ML algorithm to train (default: RandomForest)')
    parser.add_argument('--imbalance-strategy', type=str, default='none',
                        choices=['none', 'class_weight', 'smote'],
                        help='Class imbalance handling strategy (default: none)')
    parser.add_argument('--set-active', action='store_true',
                        help='Set the trained model as the active production model')
    args = parser.parse_args()

    if args.algorithm == 'all':
        # Train all three algorithms with the specified imbalance strategy
        algorithms = ['RandomForest', 'XGBoost', 'LightGBM']
        best_model = None
        best_auc = 0.0

        for algo in algorithms:
            print(f"\n{'='*60}")
            print(f"  Training: {algo}")
            print(f"{'='*60}")
            model_version, metrics = train_and_register(
                algorithm=algo,
                imbalance_strategy=args.imbalance_strategy,
                set_active=False  # Don't activate yet
            )
            if metrics['roc_auc'] > best_auc:
                best_auc = metrics['roc_auc']
                best_model = model_version

        # Activate the best model if requested
        if args.set_active and best_model:
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
            import django
            django.setup()
            from predictions.models import ModelVersion
            ModelVersion.objects.update(is_active=False)
            best_model.is_active = True
            best_model.save()
            print(f"\n[*] Best model activated: {best_model.name} (ROC-AUC: {best_auc:.4f})")
    else:
        train_and_register(
            algorithm=args.algorithm,
            imbalance_strategy=args.imbalance_strategy,
            set_active=args.set_active
        )


if __name__ == '__main__':
    main()