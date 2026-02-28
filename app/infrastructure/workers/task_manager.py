"""Реализация TaskPublisher: постановка фоновых задач через Celery."""
from typing import Optional

from app.infrastructure.workers.celery_app import celery_app


class TaskManager:
    """Реализация TaskPublisher — отправка задач в Celery."""

    def __init__(self, app=None):
        self._celery = app or celery_app

    def send_embeddings_task(
        self,
        content_file_id: int,
        text: str,
        namespace_id: Optional[int],
        filename: str,
        user_file_id: int,
    ) -> Optional[str]:
        result = self._celery.send_task(
            "process_file_embeddings",
            kwargs={
                "file_id": content_file_id,
                "text": text,
                "namespace_id": namespace_id,
                "filename": filename,
                "user_file_id": user_file_id,
            },
        )
        return str(result.id) if result and getattr(result, "id", None) else None

    def send_summary_url_task(self, url: str, user_id: int) -> Optional[str]:
        result = self._celery.send_task(
            "process_summary_url",
            args=[url, user_id],
        )
        return str(result.id) if result and getattr(result, "id", None) else None
