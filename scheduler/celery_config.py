from celery import Celery

# Celery configuration: run background tasks asynchronously
celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",  # Redis URL (localhost and default port)
    backend="redis://localhost:6379/0",  # To store results (optional)
)

celery_app.conf.task_routes = {
    "tasks.process_and_store_embeddings": {"queue": "default"},
}

