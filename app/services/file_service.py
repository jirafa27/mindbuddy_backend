import base64
import hashlib
import logging
import re
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.protocols import EmbeddingProvider, FileStorage, VectorRepository, NamespaceRepository, FileRepository, UserFileRepository
from app.infrastructure.db.models import UserFile
from app.schemas.file import FileCreated, FileProcessingResult, FileInfo, DeduplicationResult, ProcessUserLinkResult
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
    """Сервис для работы с файлами: загрузка и обработка"""

    def __init__(
        self,
        storage: FileStorage,
        file_reader_factory: Optional[FileReaderFactory] = None,
        vector_repository: Optional[VectorRepository] = None,
        text_chunker: Optional[TextChunkerService] = None,
        embedding_service: Optional[EmbeddingProvider] = None,
        namespace_repository: Optional[NamespaceRepository] = None,
        db: Optional[AsyncSession] = None,
        file_repository: Optional[FileRepository] = None,
        user_file_repository: Optional[UserFileRepository] = None,
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
            db: Сессия БД (для API — из Depends(get_db); для Celery — None).
            user_file_repository: AsyncUserFileRepository для process_user_link.
        """
        self.storage = storage
        self.file_reader_factory = file_reader_factory
        self.vector_repository = vector_repository
        self.text_chunker = text_chunker
        self.embedding_service = embedding_service
        self.namespace_repository = namespace_repository
        self.db = db
        self.file_repository = file_repository
        self.user_file_repository = user_file_repository

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Убирает из имени файла только недопустимые в путях символы (\\ / : * ? \" < > |). Пробелы и Unicode (названия на любом языке) сохраняются."""
        if not filename or not filename.strip():
            return filename
        unsafe = r'\\/:*?"<>|\x00-\x1f'
        result = re.sub(r"[" + unsafe + "]", "_", filename)
        result = " ".join(result.split()).strip()
        return result or "unnamed"

    @staticmethod
    def compute_content_hash(content: bytes | str) -> str:
        """Вычисляет SHA-256 хэш контента."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    async def check_deduplication(
        self,
        user_id: int,
        source_url: Optional[str] = None,
        content_hash: Optional[str] = None,
    ) -> DeduplicationResult:
        """
        Проверяет, существует ли файл с таким URL или хэшем.
        
        Args:
            user_id: ID пользователя
            source_url: URL источника (для YouTube/HTML)
            content_hash: SHA-256 хэш контента
            file_repository: Репозиторий файлов (опционально)
            
        Returns:
            DeduplicationResult с информацией о дубликате
        """
        if not self.file_repository:
            raise ValueError("File repository is required for deduplication check")
        
        # Сначала проверяем по URL (быстрее)
        if source_url:
            existing = await self.file_repository.find_by_source_url(source_url, user_id)
            if existing:    
                logger.info("[Deduplication] Found existing file for URL: %s (file_id=%d)", source_url, existing.id)
                return DeduplicationResult(is_duplicate=True, existing_file_id=existing.id)
        
        # Затем по хэшу контента
        if content_hash:
            existing = await self.file_repository.find_by_content_hash(content_hash, user_id)
            if existing:
                logger.info("[Deduplication] Found existing file for hash: %s (file_id=%d)", content_hash[:16], existing.id)
                return DeduplicationResult(is_duplicate=True, existing_file_id=existing.id)
        
        return DeduplicationResult(is_duplicate=False)

    @staticmethod
    def content_hash_from_url(url: str) -> str:
        """
        Вычисляет хэш контента из URL.
        Args:
            url: URL источника

        Returns:
            Хэш контента
        """
        url = (url or "").strip()
        _YOUTUBE_VIDEO_ID_RE = re.compile(
            r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})"
        )
        m = _YOUTUBE_VIDEO_ID_RE.search(url)
        if m:
            return f"yt:{m.group(1)}"
        return f"url:{hashlib.sha256(url.encode('utf-8')).hexdigest()[:32]}"

    async def process_user_link(
        self,
        user_id: int,
        url: str,
        namespace_id: Optional[int],
        content_extractor: Any,
        trigger_background_task: Any,
    ) -> ProcessUserLinkResult:
        """
        Обработка ссылки пользователя.

        1. Вычисляет хэш контента из URL.
        2. Если файл с таким хэшем уже существует (кэш hit): создает запись пользователя и возвращает результат.
        3. Если файл с таким хэшем не существует (кэш miss): создает запись файла и пользователя, запускает фоновую задачу.

        Args:
            user_id: ID пользователя
            url: URL источника
            namespace_id: ID пространства
            content_extractor: Сервис для извлечения контента
            trigger_background_task: Функция для запуска фоновой задачи

        Returns:
            ProcessUserLinkResult с информацией о пользователе, файле, статусе и текстом.
        """
        content_hash = self.content_hash_from_url(url)
        user_file_repo = self.user_file_repository
        content_file_repo = self.file_repository

        existing = await self.file_repository.get_by_content_hash(content_hash)
        if existing:
            # Cache hit: user may already have this file (unique on user_id+file_id)
            existing_user = await user_file_repo.find_by_user_and_file(user_id, existing.id)
            default_title = (existing.media_metadata or {}).get("title") or existing.source_url or "Document"
            if existing_user:
                existing_user.namespace_id = namespace_id
                existing_user.custom_title = existing_user.custom_title or default_title
                await self.db.flush()
                await self.db.commit()
                logger.info("[process_user_link] Cache hit, existing user_file: %d", existing_user.id)
                return ProcessUserLinkResult(
                    user_file_id=existing_user.id,
                    content_file_id=existing.id,
                    custom_title=existing_user.custom_title or default_title,
                    is_cache_hit=True,
                    processing_status=existing.processing_status,
                    text=existing.transcript_text,
                )
            user_file = await user_file_repo.create(
                user_id=user_id,
                file_id=existing.id,
                namespace_id=namespace_id,
                custom_title=default_title,
            )
            await self.db.commit()
            logger.info("[process_user_link] Cache hit: content_hash=%s, user_file_id=%d", content_hash, user_file.id)
            return ProcessUserLinkResult(
                user_file_id=user_file.id,
                content_file_id=existing.id,
                custom_title=user_file.custom_title or default_title,
                is_cache_hit=True,
                processing_status=existing.processing_status,
                text=existing.transcript_text,
            )

        parsed = await content_extractor.extract(url)
        content_file = await content_file_repo.create(
            content_hash=content_hash,
            source_url=url,
            transcript_text=parsed.text,
            media_metadata={"title": parsed.title, "content_type": parsed.content_type},
            processing_status="pending",
        )
        default_title = parsed.title or "Document"
        user_file = await user_file_repo.create(
            user_id=user_id,
            file_id=content_file.id,
            namespace_id=namespace_id,
            custom_title=default_title,
        )
        await self.db.commit()

        trigger_background_task(
            content_file.id,
            parsed.text,
            namespace_id,
            self.sanitize_filename(f"{default_title}.md"),
        )
        logger.info(
            "[process_user_link] Cache miss: content_hash=%s, content_file_id=%d, user_file_id=%d",
            content_hash, content_file.id, user_file.id,
        )
        return ProcessUserLinkResult(
            user_file_id=user_file.id,
            content_file_id=content_file.id,
            custom_title=default_title,
            is_cache_hit=False,
            processing_status="pending",
            text=parsed.text,
        )

    def extract_text(self, file_content: bytes, file_ext: str) -> str:
        """
        Извлекает текст из файла (публичный метод).
        
        Args:
            file_content: Содержимое файла в виде bytes
            file_ext: Расширение файла (без точки)
            
        Returns:
            Извлечённый текст
            
        Raises:
            ValidationError: При ошибке извлечения текста
        """
        return self._extract_text_from_file(file_content, file_ext)

    def _extract_text_from_file(self, file_content: bytes, file_ext: str) -> str:
        """
        Извлекает текст из файла в зависимости от его типа.
        Использует ту же file_reader_factory, что и FileAgent, для единообразной обработки типов файлов.

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
        user_file = await self.user_file_repository.get_by_id(file_id)
        if not user_file:
            raise NotFoundError(f"File with id {file_id} not found")
        if user_file.user_id != user_id:
            raise ForbiddenError("You don't have access to this file")

        content_file = await self.file_repository.get_by_id(user_file.file_id)
        if not content_file or not content_file.file_path:
            raise NotFoundError("File path not found in database")

        try:
            file_content = await self.storage.download_file(content_file.file_path)
            filename = user_file.custom_title or (content_file.media_metadata or {}).get("title") or "document"
            file_type = (content_file.media_metadata or {}).get("file_type") or "md"
            content_type = self._get_content_type(file_type)
            return file_content, filename, content_type
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
        user_file = await self.user_file_repository.get_by_id(file_id)
        if not user_file:
            raise NotFoundError(f"File with id {file_id} not found")
        if user_file.user_id != user_id:
            raise ForbiddenError("You don't have access to this file")

        await self.user_file_repository.delete(user_file.id)
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
        safe_filename = self.sanitize_filename(filename)
        object_name = self.storage.generate_object_name(user_id, namespace_id, safe_filename)
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
        content_hash: Optional[str] = None,
    ) -> UserFile:
        """
        Создаёт запись о контенте (File) и ссылку пользователя (UserFile). 
        При ошибке откатывает транзакцию и удаляет файл из MinIO.
        Raises ValidationError при ошибке сохранения в БД.
        """
        ch = content_hash or f"upload:{hashlib.sha256(object_name.encode()).hexdigest()[:32]}"
        try:
            content_file = await self.file_repository.create(
                content_hash=ch,
                file_path=object_name,
                media_metadata={"title": filename, "file_type": file_ext},
                processing_status="completed",
            )
            user_file = await self.user_file_repository.create(
                user_id=user_id,
                file_id=content_file.id,
                namespace_id=namespace_id,
                custom_title=filename,
            )
            await self.db.commit()
            logger.info("File created in DB: user_file_id=%s, content_file_id=%s, filename=%s", user_file.id, content_file.id, filename)
            return user_file
        except Exception as e:
            logger.error("Failed to save file to DB: %s", e)
            await self.db.rollback()
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
        content_hash: Optional[str] = None,
    ) -> FileCreated:
        """
        Валидирует и сохраняет файл в БД.

        Args:
            file_content: Содержимое файла в виде bytes
            filename: Имя файла
            user_id: ID пользователя
            namespace_id: ID пространства знаний
            content_hash: Опциональный хэш контента.

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

        user_file = await self._persist_file_record(
            object_name=object_name,
            user_id=user_id,
            namespace_id=namespace_id,
            filename=filename,
            file_ext=file_ext,
            content_hash=content_hash,
        )

        return FileCreated(
            file_id=user_file.id,
            content_file_id=user_file.file_id,
            filename=user_file.custom_title or filename,
            text=text,
        )

    async def create_file_from_text(
        self,
        text: str,
        user_id: int,
        namespace_id: Optional[int] = None,
        title: Optional[str] = None,
        file_repository: Optional[Any] = None,
    ) -> FileCreated:
        """
        Сохраняет переданный текст как MD-файл в пространстве пользователя.

        Args:
            text: Содержимое (не пустое).
            user_id: ID пользователя.
            namespace_id: ID пространства знаний (опционально; None — без пространства).
            title: Заголовок для имени файла; если не передан — note_YYYYMMDD_HHMMSS.
            file_repository: Опциональный репозиторий файлов.

        Returns:
            FileCreated с file_id, filename, text.

        Raises:
            ValidationError: Если текст пустой или загрузка не удалась.
        """
        text_clean = (text or "").strip()
        if not text_clean:
            raise ValidationError("Текст не может быть пустым")
        name = (title or "").strip() or f"note_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        if not name.endswith(".md"):
            name = f"{name}.md"
        filename = self.sanitize_filename(name)
        file_content = text_clean.encode("utf-8")
        logger.info("[FileService] Creating file from text: %s (user=%d, namespace_id=%s)", filename, user_id, namespace_id)
        return await self.upload_file(
            file_content=file_content,
            filename=filename,
            namespace_id=namespace_id,
            user_id=user_id,
            file_repository=file_repository,
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

        await self.vector_repository.create_batch(
            file_id=file_id,
            chunks=chunks,
            embeddings=embeddings,
            namespace_id=namespace_id,
        )

        return FileProcessingResult(
            file_id=file_id,
            chunks_count=len(chunks),
            status="success",
        )


    async def save_extracted_content(
        self,
        user_id: int,
        text: str,
        title: str,
        source_url: str,
        content_hash: str,
        content_type: str,
        namespace_id: Optional[int] = None,
    ) -> UserFile:
        """
        Сохраняет извлечённый контент (из YouTube, HTML): создаёт File (контент) и UserFile (ссылка пользователя).

        Returns:
            Созданная запись UserFile.
        """
        if self.db is None or not self.file_repository or not self.user_file_repository:
            raise ValidationError("DB session, file_repository and user_file_repository are required")

        filename = self.sanitize_filename(f"{title}.md")
        text_bytes = text.encode("utf-8")
        object_name = self.storage.generate_object_name(user_id, namespace_id, filename)
        try:
            await self.storage.upload_file(
                file_content=text_bytes,
                object_name=object_name,
                content_type="text/markdown",
                metadata={
                    "user_id": str(user_id),
                    "source_url": source_url,
                    "content_type": content_type,
                },
            )
            logger.info("[FileService] Saved extracted content to MinIO: %s", object_name)
        except Exception as e:
            raise ValidationError(f"Failed to upload to storage: {e}")

        try:
            content_file = await self.file_repository.create(
                content_hash=content_hash,
                source_url=source_url,
                transcript_text=text,
                file_path=object_name,
                media_metadata={"title": title, "content_type": content_type},
                processing_status="completed",
            )
            user_file = await self.user_file_repository.create(
                user_id=user_id,
                file_id=content_file.id,
                namespace_id=namespace_id,
                custom_title=title,
            )
            await self.db.commit()
            logger.info("[FileService] Created file record: user_file_id=%d, source=%s", user_file.id, source_url)
            return user_file
        except Exception as e:
            logger.error("[FileService] Failed to save file to DB: %s", e)
            await self.db.rollback()
            try:
                await self.storage.delete_file(object_name)
            except Exception:
                pass
            raise ValidationError(f"Failed to save to database: {e}")

    async def get_file_text(self, user_file_id: int, user_id: int) -> str:
        """
        Вернуть текст существующего файла по user_file_id.
        Проверяет принадлежность пользователю. Загружает файл из хранилища и извлекает текст.
        """
        if not self.file_repository or not self.user_file_repository:
            raise ValidationError("file_repository and user_file_repository required")
        uf = await self.user_file_repository.get_by_id(user_file_id)
        if not uf:
            raise NotFoundError(f"Файл с ID {user_file_id} не найден")
        if uf.user_id != user_id:
            raise ForbiddenError("Файл не принадлежит пользователю")
        content_file = await self.file_repository.get_by_id(uf.file_id)
        if not content_file or not content_file.file_path:
            raise ValueError("Файл или путь не найден")
        file_content = await self.storage.download_file(content_file.file_path)
        if not file_content:
            raise ValueError("Не удалось загрузить файл из хранилища")
        meta = content_file.media_metadata or {}
        name_for_ext = uf.custom_title or meta.get("title") or "document"
        file_ext = name_for_ext.rsplit(".", 1)[-1].lower() if "." in name_for_ext else "md"
        text = self.extract_text(file_content, file_ext)
        if not text or not text.strip():
            raise ValueError("Не удалось извлечь текст из файла")
        return text

    async def get_file_info(
        self,
        file_id: int,
        user_id: int,
    ) -> FileInfo:
        """Файл по ID с проверкой доступа. Возвращает FileInfo или None."""
        uf = await self.user_file_repository.get_by_id(file_id)
        if not uf:
            raise NotFoundError(f"Файл с id {file_id} не найден")
        if uf.user_id != user_id:
            raise ForbiddenError("У вас нет доступа к этому файлу")
        content_file = await self.file_repository.get_by_id(uf.file_id)
        if not content_file:
            raise NotFoundError("Файл или путь не найден")
        meta = content_file.media_metadata or {}
        return FileInfo(
            user_file_id=uf.id,
            content_file_id=uf.file_id,
            user_id=uf.user_id,
            namespace_id=uf.namespace_id,
            filename=uf.custom_title or meta.get("title", "document"),
            file_type=meta.get("file_type", "md"),
            file_size=0,
            created_at=content_file.created_at,
            updated_at=content_file.created_at,
            file_path=content_file.file_path,
        )

    async def move_to_namespace(
        self,
        file_id: int,
        namespace_id: int,
        user_id: int,
    ) -> FileInfo:
        """
        Перемещает файл в указанное пространство.

        Args:
            file_id: ID файла
            namespace_id: ID пространства назначения
            user_id: ID пользователя

        Returns:
            Обновлённая информация о файле

        Raises:
            NotFoundError: Файл или пространство не найдены
            ForbiddenError: Нет доступа к файлу или пространству
        """

        # Проверяем файл (UserFile)
        file = await self.user_file_repository.get_by_id(file_id)
        if not file:
            raise NotFoundError(f"Файл с id {file_id} не найден")
        if file.user_id != user_id:
            raise ForbiddenError("У вас нет доступа к этому файлу")

        # Проверяем пространство
        namespace = await self.namespace_repository.get_by_id(namespace_id)
        if not namespace:
            raise NotFoundError(f"Пространство с id {namespace_id} не найдено")
        if namespace.user_id != user_id:
            raise ForbiddenError("У вас нет доступа к этому пространству")

        await self.user_file_repository.update_namespace(file.id, namespace_id)
        await self.db.commit()

        logger.info("[FileService] Moved file %d to namespace %d", file_id, namespace_id)

        content_file = await self.file_repository.get_by_id(file.file_id)
        meta = (content_file.media_metadata or {}) if content_file else {}
        return FileInfo(
            user_file_id=file.id,
            content_file_id=file.file_id,
            user_id=file.user_id,
            namespace_id=namespace_id,
            filename=file.custom_title or meta.get("title", "document"),
            file_type=meta.get("file_type", "md"),
            file_size=0,
            created_at=content_file.created_at if content_file else datetime.utcnow(),
            updated_at=content_file.created_at if content_file else datetime.utcnow(),
            file_path=content_file.file_path if content_file else None,
        )