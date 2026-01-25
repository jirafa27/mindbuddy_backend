from fastapi import Depends

from app.services.text_chunker import TextChunker
from app.services.embedding_service import EmbeddingService
from app.services.file_service import FileService
from app.services.yandex_iam_service import YandexIAMService


def get_text_chunker() -> TextChunker:
    """Провайдер для TextChunker"""
    return TextChunker()


def get_yandex_iam_service() -> YandexIAMService:
    """Провайдер для YandexIAMService"""
    return YandexIAMService()


def get_embedding_service(
    iam_service: YandexIAMService = Depends(get_yandex_iam_service),
) -> EmbeddingService:
    """Провайдер для EmbeddingService"""
    return EmbeddingService(iam_service=iam_service)


def get_file_service() -> FileService:
    """Провайдер для FileService (для загрузки файлов)"""
    return FileService()


def create_file_processing_service() -> FileService:
    """
    Фабрика для создания FileService со всеми зависимостями для обработки файлов.
    Используется в Celery задачах.
    """
    iam_service = YandexIAMService()
    embedding_service = EmbeddingService(iam_service=iam_service)
    text_chunker = TextChunker()
    
    return FileService(
        text_chunker=text_chunker,
        embedding_service=embedding_service,
    )
