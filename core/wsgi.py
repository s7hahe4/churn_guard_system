"""
WSGI config for core project.
It exposes the WSGI callable as a module-level variable named ``application``.
Includes automatic database migration and model bootstrapping on container boot.
"""

import os
import sys

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

# Ensure models and static directories exist
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(PROJECT_ROOT, 'ml_engine', 'saved_models'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'staticfiles'), exist_ok=True)

from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

# Run database migrations on container boot
try:
    from django.core.management import call_command
    print("==> Running database migrations on boot (wsgi)...")
    call_command('migrate', interactive=False)
    print("==> Database migrations ready.")
    print("==> Collecting static assets (wsgi)...")
    call_command('collectstatic', interactive=False, verbosity=0)
    print("==> Static assets collected.")
except Exception as e:
    print(f"==> Startup migration/static note: {e}")

# Check if model version exists; if not, train baseline model
try:
    from predictions.models import ModelVersion
    if not ModelVersion.objects.filter(is_active=True).exists():
        print("==> Bootstrapping production ML model (wsgi)...")
        from ml_engine.train import train_and_register
        train_and_register(algorithm='RandomForest', imbalance_strategy='class_weight', set_active=True)
        print("==> Production model registered and active.")
except Exception as e:
    print(f"==> Model bootstrap note: {e}")
