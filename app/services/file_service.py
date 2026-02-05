import base64
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.protocols import AsyncFileRepository, EmbeddingProvider, FileStorage, VectorRepository, NamespaceRepository
from app.infrastructure.db.models import File as FileModel
from app.schemas.file import FileCreated, FileProcessingResult, FileInfo
from app.core.config import settings
from app.core.exceptions import (
    ValidationError,
    NotFoundError,
    ForbiddenError,
    FileTooLargeError,
)
from app.services.text_chunker import TextChunkerService
from app.utils.file_readers import FileReaderFactory

logger = logging.getLogger(__name__)


class FileService:
    """Сервис для работы с файлами: загрузка и обработка (разбиение на чанки и векторизация)"""

    def __init__(
        self,
        storage: FileStorage,
        file_repository: Optional[AsyncFileRepository] = None,
        file_reader_factory: Optional[FileReaderFactory] = None,
        vector_repository: Optional[VectorRepository] = None,
        text_chunker: Optional[TextChunkerService] = None,
        embedding_service: Optional[EmbeddingProvider] = None,
        namespace_repository: Optional[NamespaceRepository] = None,
        db: Optional[AsyncSession] = None,
    ):
        """
        Args:
            storage: Хранилище файлов (обязательно).
            file_repository: Репозиторий файлов (для download/delete/upload; опционально для Celery).
            file_reader_factory: Фабрика ридеров по расширению файла.
            vector_repository: Репозиторий эмбеддингов (для process_file; опционально для API).
            text_chunker: Для обработки файлов (Celery).
            embedding_service: Для обработки файлов (Celery).
            namespace_repository: Репозиторий пространств (для валидации).
            db: Сессия БД для управления транзакциями (commit/rollback); опционально для Celery.
        """
        self.storage = storage
        self.file_repository = file_repository
        self.file_reader_factory = file_reader_factory
        self.vector_repository = vector_repository
        self.text_chunker = text_chunker
        self.embedding_service = embedding_service
        self.namespace_repository = namespace_repository
        self.db = db


    def _extract_text_from_file(self, file_content: bytes, file_ext: str) -> str:
        """
        Извлекает текст из файла в зависимости от его типа.

        Args:
            file_content: Содержимое файла в виде bytes
            file_ext: Расширение файла (без точки)

        Returns:
            Извлечённый текст

        Raises:
            ValidationError: При ошибке извлечения текста
        """
        try:
            reader = self.file_reader_factory.get_reader(file_ext)
            return reader.read(file_content)
        except UnicodeDecodeError:
            raise ValidationError("File encoding not supported. Please use UTF-8 encoded files.")
        except ValueError as e:
            raise ValidationError(str(e))
        except Exception as e:
            raise ValidationError(f"Failed to extract text from file: {str(e)}")

    def _get_content_type(self, file_ext: str) -> str:
        """
        Возвращает MIME type для файла.

        Args:
            file_ext: Расширение файла (без точки)

        Returns:
            MIME type
        """
        content_types = {
            'txt': 'text/plain',
            'md': 'text/markdown',
            'pdf': 'application/pdf',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        }
        return content_types.get(file_ext, 'application/octet-stream')

    async def download_file(
        self,
        file_id: int,
        user_id: int,
    ) -> tuple[bytes, str, str]:
        """
        Скачивает файл из хранилища.

        Args:
            file_id: ID файла
            user_id: ID пользователя (для проверки прав доступа)

        Returns:
            Кортеж: (содержимое файла, имя файла, content_type)

        Raises:
            NotFoundError: Если файл не найден
            ForbiddenError: Если нет доступа к файлу
        """
        file_entity = await self.file_repository.get_by_id(file_id)
        if not file_entity:
            raise NotFoundError(f"File with id {file_id} not found")

        if file_entity.user_id != user_id:
            raise ForbiddenError("You don't have access to this file")

        if not file_entity.file_path:
            raise ValidationError("File path not found in database")

        try:
            file_content = await self.storage.download_file(file_entity.file_path)
            content_type = self._get_content_type(file_entity.file_type)
            return file_content, file_entity.filename, content_type
        except Exception as e:
            raise ValidationError(f"Failed to download file from storage: {str(e)}")

    async def delete_file(
        self,
        file_id: int,
        user_id: int,
    ) -> bool:
        """
        Удаляет файл из БД и хранилища.

        Args:
            file_id: ID файла
            user_id: ID пользователя (для проверки прав доступа)

        Returns:
            True если удаление успешно

        Raises:
            NotFoundError: Если файл не найден
            ForbiddenError: Если нет доступа к файлу
        """
        file_entity = await self.file_repository.get_by_id(file_id)
        if not file_entity:
            raise NotFoundError(f"File with id {file_id} not found")

        if file_entity.user_id != user_id:
            raise ForbiddenError("You don't have access to this file")

        if file_entity.file_path:
            try:
                await self.storage.delete_file(file_entity.file_path)
            except Exception as e:
                logger.warning("Failed to delete file from storage: %s", e)

        await self.file_repository.delete(file_entity)
        await self.db.commit()

        return True

    async def _validate_upload_params(
        self,
        filename: str,
        file_content: bytes,
        user_id: int,
        namespace_id: Optional[int],
    ) -> str:
        """
        Валидация параметров загрузки. Возвращает расширение файла.
        Raises: ValidationError, NotFoundError, ForbiddenError, FileTooLargeError
        """
        if not filename:
            raise ValidationError("Filename is required")

        file_ext = filename.split(".")[-1].lower()
        if file_ext not in settings.ALLOWED_FILE_TYPES:
            raise ValidationError(
                f"File type '{file_ext}' not allowed. Allowed types: {settings.ALLOWED_FILE_TYPES}"
            )

        if namespace_id is not None:
            namespace = await self.namespace_repository.get_by_id(namespace_id)
            if not namespace:
                raise NotFoundError(f"Namespace with id {namespace_id} not found")
            if namespace.user_id != user_id:
                raise ForbiddenError("Namespace does not belong to this user")

        if len(file_content) > settings.MAX_FILE_SIZE:
            raise FileTooLargeError(
                f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes"
            )

        return file_ext

    async def _upload_to_storage(
        self,
        file_content: bytes,
        file_ext: str,
        filename: str,
        user_id: int,
        namespace_id: Optional[int],
    ) -> str:
        """Загружает файл в FileStorage. Возвращает object_name (путь в хранилище)."""
        object_name = self.storage.generate_object_name(user_id, namespace_id, filename)
        filename_base64 = base64.b64encode(filename.encode("utf-8")).decode("ascii")
        try:
            await self.storage.upload_file(
                file_content=file_content,
                object_name=object_name,
                content_type=self._get_content_type(file_ext),
                metadata={
                    "user_id": str(user_id),
                    "namespace_id": str(namespace_id),
                    "original_filename": filename_base64,
                },
            )
        except Exception as e:
            raise ValidationError(f"Failed to upload file to storage: {str(e)}")
        return object_name

    async def _persist_file_record(
        self,
        *,
        object_name: str,
        user_id: int,
        namespace_id: Optional[int],
        filename: str,
        file_ext: str,
        file_size: int,
        db: AsyncSession,
        file_repository: AsyncFileRepository,
    ) -> FileModel:
        """
        Создаёт запись о файле в БД. При ошибке откатывает транзакцию и удаляет файл из MinIO.
        Raises ValidationError при ошибке сохранения в БД.
        """
        try:
            db_file = await file_repository.create(
                user_id=user_id,
                namespace_id=namespace_id,
                filename=filename,
                file_path=object_name,
                file_type=file_ext,
                file_size=file_size,
            )
            await db.commit()
            await db.refresh(db_file)
            logger.info(f"File created in DB: id={db_file.id}, filename={filename}")
            return db_file
        except Exception as e:
            logger.error(f"Failed to save file to DB: {e}")
            await db.rollback()
            try:
                await self.storage.delete_file(object_name)
            except Exception:
                pass
            raise ValidationError(f"Failed to save file to database: {str(e)}")

    async def upload_file(
        self,
        user_id: int,
        file_content: bytes,
        filename: str,
        namespace_id: Optional[int] = None,
        db: Optional[AsyncSession] = None,
        file_repository: Optional[AsyncFileRepository] = None,
    ) -> FileCreated:
        """
        Валидирует и сохраняет файл в БД.

        Args:
            file_content: Содержимое файла в виде bytes
            filename: Имя файла
            user_id: ID пользователя
            namespace_id: ID пространства знаний
            db: Опционально — сессия для графа (DBAgent).
            file_repository: Опционально — репозиторий для графа (DBAgent).

        Returns:
            FileCreated с информацией о созданном файле

        Raises:
            ValidationError: При ошибках валидации
            NotFoundError: Если ресурс не найден
            ForbiddenError: При отсутствии доступа
            FileTooLargeError: Если файл слишком большой
        """
        file_ext = await self._validate_upload_params(
            filename, file_content, user_id, namespace_id
        )
        text = self._extract_text_from_file(file_content, file_ext)
        if not text or not text.strip():
            raise ValidationError("File content is empty or could not be processed")

        object_name = await self._upload_to_storage(
            file_content, file_ext, filename, user_id, namespace_id
        )

        _db = db or self.db
        _repo = file_repository or self.file_repository
        if not _db or not _repo:
            raise ValidationError("DB session and file repository are required for upload_file")

        db_file = await self._persist_file_record(
            object_name=object_name,
            user_id=user_id,
            namespace_id=namespace_id,
            filename=filename,
            file_ext=file_ext,
            file_size=len(file_content),
            db=_db,
            file_repository=_repo,
        )

        return FileCreated(
            file_id=db_file.id,
            filename=db_file.filename,
            text=text,
        )

    async def process_file(
        self,
        file_id: int,
        text: str,
        namespace_id: Optional[int] = None,
        filename: Optional[str] = None,
    ) -> FileProcessingResult:
        """
        Обрабатывает файл: разбивает на чанки, генерирует эмбеддинги и сохраняет в БД.
        Запускать из Celery через asyncio.run(service.process_file(...)).

        Args:
            file_id: ID файла в БД
            text: Текст файла для обработки
            namespace_id: ID пространства знаний (опционально)
            filename: Название файла для включения в чанки (опционально)

        Returns:
            FileProcessingResult с результатами обработки

        Raises:
            ValueError: Если не переданы text_chunker, embedding_service или vector_repository
        """
        if not self.text_chunker or not self.embedding_service:
            raise ValueError(
                "TextChunkerService and EmbeddingProvider must be provided for file processing"
            )
        if not self.vector_repository:
            raise ValueError("VectorRepository must be provided for file processing")

        chunks = self.text_chunker.chunk_text(text, filename=filename)
        if not chunks:
            raise ValueError("File content is empty or could not be processed")

        embeddings = await self.embedding_service.generate_embeddings_batch(chunks)

        self.vector_repository.create_batch(
            file_id=file_id,
            chunks=chunks,
            embeddings=embeddings,
            namespace_id=namespace_id,
        )

        if self.db:
            await self.db.commit()

        return FileProcessingResult(
            file_id=file_id,
            chunks_count=len(chunks),
            status="success",
        )


    async def get_file(
        self,
        file_id: int,
        user_id: int,
    ) -> Optional[FileInfo]:
        """Файл по ID с проверкой доступа. Возвращает FileInfo или None."""
        file = await self.file_repository.get_by_id(file_id)
        if not file:
            raise NotFoundError(f"File with id {file_id} not found")
        if file.user_id != user_id:
            raise ForbiddenError("You don't have access to this file")
        return FileInfo(
            id=file.id,
            user_id=file.user_id,
            namespace_id=file.namespace_id,
            filename=file.filename,
            file_type=file.file_type,
            file_size=file.file_size,
            created_at=file.created_at,
            updated_at=file.updated_at,
            file_path=file.file_path,
        )