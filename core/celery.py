"""
Celery application configuration for ChurnGuard AI.

Design Decision — Why Celery over increasing request timeout:
    Synchronous batch processing blocks the web server thread. A 50,000-row CSV
    takes ~30 seconds to score — well beyond typical HTTP timeout limits. Celery
    moves this work to a background worker process. The API returns immediately
    with a job ID, and the frontend polls for completion. This pattern is:
    1. Non-blocking: The web server stays responsive during batch processing.
    2. Horizontally scalable: Add more Celery workers to handle more concurrent jobs.
    3. Fault-tolerant: Failed jobs are recorded with error messages, not lost to timeouts.
"""

import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

app = Celery('core')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
