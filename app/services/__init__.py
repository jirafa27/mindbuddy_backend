"""
Сервисный слой приложения.
Содержит сервисы для работы с бизнес-логикой приложения.
"""
from app.services.file_service import FileService, DeduplicationResult
from app.services.text_chunker import TextChunkerService

__all__ = [
    "FileService",
    "DeduplicationResult",
    "TextChunkerService",
]
