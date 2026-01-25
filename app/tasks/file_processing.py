from celery import Task

from app.celery_app import celery_app
from app.db.sync_session import SessionLocal
from app.db.repositories import FileRepository
from app.core.dependencies import create_file_processing_service


@celery_app.task(bind=True, name="process_file_embeddings")
def process_file_embeddings(
    self: Task,
    file_id: int,
    text: str,
    namespace_id: int,
) -> dict:
    """
    Задача Celery для обработки файла: разбиение на чанки и генерация эмбеддингов.

    Args:
        file_id: ID файла в БД
        text: Текст файла для обработки
        namespace_id: ID пространства знаний

    Returns:
        Словарь с результатами обработки
    """
    db = SessionLocal()
    try:
        file_service = create_file_processing_service()
        result = file_service.process_file(db, file_id, text, namespace_id)
        db.commit()
        return result.model_dump()
    except Exception as e:
        db.rollback()
        FileRepository(db).delete(file_id)
        db.commit()
        raise self.retry(exc=e, countdown=60, max_retries=3)
    finally:
        db.close()
