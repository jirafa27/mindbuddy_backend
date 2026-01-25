import asyncio
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.db.models import File as FileModel, Namespace
from app.db.repositories import  VectorEmbeddingRepository
from app.schemas.file import FileCreated, FileProcessingResult
from app.core.config import settings
from app.core.exceptions import (
    ValidationError,
    NotFoundError,
    ForbiddenError,
    FileTooLargeError,
)
from app.services.text_chunker import TextChunker
from app.services.embedding_service import EmbeddingService


class FileService:
    """Сервис для работы с файлами: загрузка и обработка (разбиение на чанки и векторизация)"""

    def __init__(
        self,
        text_chunker: Optional[TextChunker] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """
        Args:
            text_chunker: Опционально, для обработки файлов (используется в Celery)
            embedding_service: Опционально, для обработки файлов (используется в Celery)
        """
        self.text_chunker = text_chunker
        self.embedding_service = embedding_service

    async def create_file(
        self,
        file_content: bytes,
        filename: str,
        namespace_id: int,
        user_id: int,
        db: AsyncSession,
    ) -> FileCreated:
        """
        Валидирует и сохраняет файл в БД.

        Args:
            file_content: Содержимое файла в виде bytes
            filename: Имя файла
            namespace_id: ID пространства знаний
            user_id: ID пользователя
            db: Сессия базы данных

        Returns:
            FileCreated с информацией о созданном файле

        Raises:
            ValidationError: При ошибках валидации
            NotFoundError: Если ресурс не найден
            ForbiddenError: При отсутствии доступа
            FileTooLargeError: Если файл слишком большой
        """
        # Валидация типа файла
        if not filename:
            raise ValidationError("Filename is required")

        file_ext = filename.split(".")[-1].lower()
        if file_ext not in settings.ALLOWED_FILE_TYPES:
            raise ValidationError(
                f"File type '{file_ext}' not allowed. Allowed types: {settings.ALLOWED_FILE_TYPES}"
            )

        # Проверка существования namespace
        namespace = await db.get(Namespace, namespace_id)
        if not namespace:
            raise NotFoundError(f"Namespace with id {namespace_id} not found")

        if namespace.user_id != user_id:
            raise ForbiddenError("Namespace does not belong to this user")

        # Проверка размера
        if len(file_content) > settings.MAX_FILE_SIZE:
            raise FileTooLargeError(
                f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes"
            )

        # Декодируем текст
        text = file_content.decode("utf-8")

        if not text or not text.strip():
            raise ValidationError("File content is empty or could not be processed")

        # Создаем запись о файле в БД
        db_file = FileModel(
            user_id=user_id,
            namespace_id=namespace_id,
            filename=filename,
            file_type=file_ext,
            file_size=len(file_content),
        )
        db.add(db_file)
        await db.flush()  # Получаем file_id
        await db.commit()

        return FileCreated(
            file_id=db_file.id,
            filename=db_file.filename,
            text=text,
        )

    def process_file(
        self,
        db: Session,
        file_id: int,
        text: str,
        namespace_id: int,
    ) -> FileProcessingResult:
        """
        Обрабатывает файл: разбивает на чанки, генерирует эмбеддинги и сохраняет в БД.

        Args:
            db: Синхронная сессия БД
            file_id: ID файла в БД
            text: Текст файла для обработки
            namespace_id: ID пространства знаний

        Returns:
            FileProcessingResult с результатами обработки

        Raises:
            Exception: При ошибке обработки
        """
        if not self.text_chunker or not self.embedding_service:
            raise ValueError("TextChunker and EmbeddingService must be provided for file processing")

        # Разбиваем на чанки
        chunks = self.text_chunker.chunk_text(text)

        if not chunks:
            raise ValueError("File content is empty or could not be processed")

        # Генерируем эмбеддинги (Celery не поддерживает async напрямую)
        # Создаем новый event loop для async операций
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        embeddings = loop.run_until_complete(
            self.embedding_service.generate_embeddings_batch(chunks)
        )

        # Сохраняем эмбеддинги в БД через репозиторий
        embedding_repo = VectorEmbeddingRepository(db)
        embedding_repo.create_batch(
            file_id=file_id,
            namespace_id=namespace_id,
            chunks=chunks,
            embeddings=embeddings,
        )

        return FileProcessingResult(
            file_id=file_id,
            chunks_count=len(chunks),
            status="success"
        )
