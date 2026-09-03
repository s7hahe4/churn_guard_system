#!/usr/bin/env bash
# Exit immediately on error
set -o errexit

echo "==> Installing pinned dependencies..."
pip install -r requirements.txt

echo "==> Applying database migrations..."
python manage.py migrate

echo "==> Training and registering production ML models (RF, XGBoost, LightGBM)..."
python ml_engine/train.py --algorithm all --imbalance-strategy class_weight --set-active

echo "==> Collecting and compressing static files via WhiteNoise..."
python manage.py collectstatic --noinput

echo "==> Build complete and ready for deployment!"
