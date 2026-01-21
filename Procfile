web: gunicorn -k uvicorn.workers.UvicornWorker --workers ${WEB_CONCURRENCY:-2} --max-requests 200 --max-requests-jitter 50 --timeout ${WEB_TIMEOUT:-120} app:app
worker: python -m backend.worker
