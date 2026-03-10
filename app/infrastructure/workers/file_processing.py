import asyncio
import logging
from celery import Task
from celery.exceptions import MaxRetriesExceededError

from app.infrastructure.db import base as db_base
from app.infrastructure.repositories import PgUserFileRepository
from app.core.dependencies import create_file_service_for_celery
from app.infrastructure.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


def _ensure_async_db() -> None:
    if db_base.AsyncSessionLocal is None:
        db_base.setup_async_engine_for_celery()


async def _delete_user_file_async(user_file_id: int) -> None:
    """Удаление UserFile по id в async-контексте (для воркера)."""
    _ensure_async_db()
    async with db_base.AsyncSessionLocal() as db:
        repo = PgUserFileRepository(db)
        await repo.delete(user_file_id)
        await db.commit()


async def _process_file_async(
    file_id: int,
    text: str,
    namespace_id: int | None,
    filename: str | None,
) -> dict:
    _ensure_async_db()
    async with db_base.AsyncSessionLocal() as db:
        file_service = create_file_service_for_celery(db)
        result = await file_service.process_file(
            file_id=file_id,
            text=text,
            namespace_id=namespace_id,
            filename=filename,
        )
        await db.commit()
        return result.model_dump()


@celery_app.task(bind=True, name="process_file_embeddings")
def process_file_embeddings(
    self: Task,
    file_id: int,
    text: str,
    namespace_id: int | None = None,
    filename: str | None = None,
    user_file_id: int | None = None,
) -> dict:
    """
    Обработка контента: чанки и эмбеддинги. file_id = content_file_id (File.id), user_file_id = UserFile.id (для удаления при сбое).
    """
    try:
        result = asyncio.run(
            _process_file_async(
                file_id=file_id,
                text=text,
                namespace_id=namespace_id,
                filename=filename,
            )
        )
        logger.info("Content file %s processed successfully: %s chunks", file_id, result.get("chunks_count"))
        return result
    except Exception as e:
        logger.error("Error processing content file %s: %s", file_id, e, exc_info=True)
        try:
            max_retries = getattr(self, "max_retries", 3) or 3
            if self.request.retries < max_retries:
                logger.warning(
                    "Retrying file %s in 60s (retry %d of %d)",
                    file_id,
                    self.request.retries + 1,
                    max_retries,
                )
            raise self.retry(exc=e, countdown=60, max_retries=3)
        except MaxRetriesExceededError:
            if user_file_id is not None:
                logger.error("All retries exhausted for file %s. Deleting user file %s.", file_id, user_file_id)
                try:
                    asyncio.run(_delete_user_file_async(user_file_id))
                    logger.info("User file %s deleted after failed processing", user_file_id)
                except Exception as delete_error:
                    logger.error("Failed to delete user file %s: %s", user_file_id, delete_error)
            raise
