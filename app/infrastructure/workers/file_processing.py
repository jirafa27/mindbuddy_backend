import asyncio
import logging
from celery import Task
from celery.exceptions import MaxRetriesExceededError

from app.infrastructure.db.sync_session import SessionLocal
from app.infrastructure.repositories import FileRepository
from app.core.dependencies import create_file_service_for_celery
from app.infrastructure.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="process_file_embeddings")
def process_file_embeddings(
    self: Task,
    file_id: int,
    text: str,
    namespace_id: int | None = None,
    filename: str | None = None,
) -> dict:
    """
    Задача Celery для обработки файла: разбиение на чанки и генерация эмбеддингов.
    Запускает асинхронный process_file через asyncio.run().

    Args:
        file_id: ID файла в БД
        text: Текст файла для обработки
        namespace_id: ID пространства знаний (опционально)
        filename: Название файла для включения в чанки (опционально)

    Returns:
        Словарь с результатами обработки
    """
    db = SessionLocal()
    try:
        file_service = create_file_service_for_celery(db)
        result = asyncio.run(
            file_service.process_file(
                file_id=file_id,
                text=text,
                namespace_id=namespace_id,
                filename=filename,
            )
        )
        db.commit()
        logger.info(f"File {file_id} processed successfully: {result.chunks_count} chunks")
        return result.model_dump()
    except Exception as e:
        logger.error(f"Error processing file {file_id}: {e}", exc_info=True)
        db.rollback()

        try:
            logger.warning(
                f"Retrying file {file_id} processing (attempt {self.request.retries + 1}/3)"
            )
            raise self.retry(exc=e, countdown=60, max_retries=3)
        except MaxRetriesExceededError:
            logger.error(f"All retries exhausted for file {file_id}. Deleting file from database.")
            try:
                FileRepository(db).delete(file_id)
                db.commit()
                logger.info(f"File {file_id} deleted after failed processing")
            except Exception as delete_error:
                logger.error(f"Failed to delete file {file_id}: {delete_error}")
                db.rollback()
            raise
    finally:
        db.close()
