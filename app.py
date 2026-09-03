"""
WSGI entrypoint alias for cloud deployment platforms (e.g. Render, DigitalOcean, Heroku).
Includes automatic database migration and model bootstrapping on container boot.
"""
import os
import sys

# Ensure Django settings are configured
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

# Ensure models and static directories exist
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(PROJECT_ROOT, 'ml_engine', 'saved_models'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'staticfiles'), exist_ok=True)

from core.wsgi import application as app
application = app

# Auto-migrate database on container boot
try:
    from django.core.management import call_command
    print("==> Running database migrations on boot...")
    call_command('migrate', interactive=False)
    print("==> Database migrations ready.")
except Exception as e:
    print(f"==> Startup migration note: {e}")

# Check if model version exists; if not, bootstrap production model
try:
    from predictions.models import ModelVersion
    if not ModelVersion.objects.filter(is_active=True).exists():
        print("==> Bootstrapping production ML model...")
        from ml_engine.train import train_and_register
        train_and_register(algorithm='RandomForest', imbalance_strategy='class_weight', set_active=True)
        print("==> Production model registered and active.")
except Exception as e:
    print(f"==> Model bootstrap note: {e}")
