"""
WSGI entrypoint alias for cloud deployment platforms (e.g. Render, DigitalOcean, Heroku)
that default to running 'gunicorn app:app'.
"""

from core.wsgi import application as app
application = app
