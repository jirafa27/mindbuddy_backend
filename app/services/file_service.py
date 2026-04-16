import base64
import hashlib
import logging
import re
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.protocols import (
    EmbeddingProvider,
    FileStorage,
    VectorRepository,
    NamespaceRepository,
    FileRepository,
    UserFileRepository,
    SummaryRepository,
    TaskPublisher,
    FileSyncNotifier,
)
from app.infrastructure.db.models import UserFile
from app.schemas.file import FileCreated, FileProcessingResult, FileInfo, FileVersionInfo, DeduplicationResult, ProcessUserLinkResult, CommandType
from app.core.config import settings
from app.core.namespace_constants import (
    TRASH_NAMESPACE_KIND,
    TRASH_NAMESPACE_NAME,
    VAULT_ROOT_NAMESPACE_KIND,
    VAULT_ROOT_NAMESPACE_NAME,
)
from app.core.exceptions import (
    ValidationError,
    NotFoundError,
    ForbiddenError,
    FileTooLargeError,
)
from app.services.file_content_service import FileContentService
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
        summary_repository: Optional[SummaryRepository] = None,
        task_publisher: Optional[TaskPublisher] = None,
        sync_notifier: Optional[FileSyncNotifier] = None,
        file_content_service: Optional[FileContentService] = None,
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
            summary_repository: Для инвалидации суммаризации при replace (опционально).
            task_publisher: Для постановки задачи эмбеддингов при replace (опционально).
        """
        self.storage = storage
        self.file_reader_factory = file_reader_factory
        self.vector_repository = vector_repository
        self.summary_repository = summary_repository
        self.task_publisher = task_publisher
        self.text_chunker = text_chunker
        self.embedding_service = embedding_service
        self.namespace_repository = namespace_repository
        self.db = db
        self.file_repository = file_repository
        self.user_file_repository = user_file_repository
        self.sync_notifier = sync_notifier
        self.file_content_service = file_content_service or FileContentService(
            storage=storage,
            file_reader_factory=file_reader_factory,
        )

    async def _get_or_create_trash_namespace_id(self, user_id: int) -> int:
        if not self.namespace_repository or not self.db:
            raise ValidationError("Namespace repository is required")
        vault_root = await self.namespace_repository.get_by_name_and_parent(
            user_id=user_id,
            parent_id=None,
            name=VAULT_ROOT_NAMESPACE_NAME,
        )
        if vault_root is None:
            vault_root = await self.namespace_repository.create(
                name=VAULT_ROOT_NAMESPACE_NAME,
                user_id=user_id,
                parent_id=None,
                kind=VAULT_ROOT_NAMESPACE_KIND,
                description=None,
            )
            await self.db.commit()
        elif vault_root.kind != VAULT_ROOT_NAMESPACE_KIND:
            vault_root.kind = VAULT_ROOT_NAMESPACE_KIND
            vault_root = await self.namespace_repository.update(vault_root)
            await self.db.commit()

        existing = await self.namespace_repository.get_by_name_and_parent(
            user_id=user_id,
            parent_id=vault_root.id,
            name=TRASH_NAMESPACE_NAME,
        )
        if existing:
            if existing.kind != TRASH_NAMESPACE_KIND:
                existing.kind = TRASH_NAMESPACE_KIND
                updated = await self.namespace_repository.update(existing)
                await self.db.commit()
                return updated.id
            return existing.id
        namespace = await self.namespace_repository.create(
            name=TRASH_NAMESPACE_NAME,
            user_id=user_id,
            parent_id=vault_root.id,
            kind=TRASH_NAMESPACE_KIND,
            description=None,
        )
        await self.db.commit()
        return namespace.id

    @staticmethod
    def _build_file_info(user_file: Any, content_file: Any) -> FileInfo:
        meta = (content_file.media_metadata or {}) if content_file else {}
        transcript_text = (content_file.transcript_text or "") if content_file else ""
        file_size = meta.get("file_size") or len(transcript_text.encode("utf-8"))
        created_at = content_file.created_at if content_file else datetime.utcnow()
        updated_at = getattr(user_file, "updated_at", None) or created_at
        return FileInfo(
            user_file_id=user_file.id,
            content_file_id=user_file.file_id,
            user_id=user_file.user_id,
            namespace_id=user_file.namespace_id,
            filename=user_file.custom_title or meta.get("title", "document"),
            file_type=meta.get("file_type", "md"),
            file_size=file_size,
            created_at=created_at,
            updated_at=updated_at,
            file_path=content_file.file_path if content_file else None,
            content_revision=getattr(user_file, "content_revision", 1) or 1,
            desktop_updated_at=getattr(user_file, "desktop_updated_at", None),
            app_updated_at=getattr(user_file, "app_updated_at", None),
            last_update_source=getattr(user_file, "last_update_source", None),
            content_hash=content_file.content_hash,
            vault_relative_path=getattr(user_file, "vault_relative_path", None),
            is_conflict_copy=getattr(user_file, "is_conflict_copy", False),
        )

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
        Проверяет, существует ли файл с таким URL или хэшем у пользователя.
        
        Args:
            user_id: ID пользователя
            source_url: URL источника (для YouTube/HTML)
            content_hash: SHA-256 хэш контента
            
        Returns:
            DeduplicationResult с информацией о дубликате (existing_file_id — id user_file)
        """
        if not self.user_file_repository:
            raise ValueError("User file repository is required for deduplication check")
        
        # Сначала проверяем по URL (быстрее)
        if source_url:
            existing = await self.user_file_repository.find_by_source_url(source_url, user_id)
            if existing:
                logger.info("[Deduplication] Found existing file for URL: %s (user_file_id=%d)", source_url, existing.id)
                return DeduplicationResult(is_duplicate=True, existing_file_id=existing.id)
        
        # Затем по хэшу контента
        if content_hash:
            existing = await self.user_file_repository.find_by_content_hash(content_hash, user_id)
            if existing:
                logger.info("[Deduplication] Found existing file for hash: %s (user_file_id=%d)", content_hash[:16], existing.id)
                return DeduplicationResult(is_duplicate=True, existing_file_id=existing.id)
        
        return DeduplicationResult(is_duplicate=False)

    async def list_user_file_ids_in_namespace(
        self, user_id: int, namespace_id: int
    ) -> list[int]:
        """Все user_files.id пользователя в пространстве (порядок — по created_at)."""
        if not self.user_file_repository:
            raise ValueError("user_file_repository is required")
        ids = await self.user_file_repository.list_ids_by_user_and_namespace(
            user_id, namespace_id
        )
        return list(ids)

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
        return self.file_content_service.extract_text(
            file_content,
            file_ext=file_ext,
            strict=True,
        )

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
        return self.file_content_service.extract_text(
            file_content,
            file_ext=file_ext,
            strict=True,
        )

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
        Перемещает файл пользователя в корзину.

        Args:
            file_id: ID файла
            user_id: ID пользователя (для проверки прав доступа)

        Returns:
            True если перемещение успешно

        Raises:
            NotFoundError: Если файл не найден
            ForbiddenError: Если нет доступа к файлу
        """
        user_file = await self.user_file_repository.get_by_id(file_id)
        if not user_file:
            raise NotFoundError(f"File with id {file_id} not found")
        if user_file.user_id != user_id:
            raise ForbiddenError("You don't have access to this file")
        trash_namespace_id = await self._get_or_create_trash_namespace_id(user_id)
        if user_file.namespace_id == trash_namespace_id:
            raise ValidationError("Файл уже находится в корзине")

        if self.sync_notifier:
            await self.sync_notifier.add_trash_command_to_queue(
                user_file_id=file_id,
                user_id=user_id,
            )
        else:
            await self.user_file_repository.update_namespace(user_file.id, trash_namespace_id)
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
        transcript_text: Optional[str] = None,
    ) -> UserFile:
        """
        Создаёт запись о контенте (File) и ссылку пользователя (UserFile). 
        При ошибке откатывает транзакцию и удаляет файл из MinIO.
        Raises ValidationError при ошибке сохранения в БД.
        """
        ch = content_hash or f"upload:{hashlib.sha256(object_name.encode()).hexdigest()[:32]}"
        try:
            existing_content_file = await self.file_repository.get_by_content_hash(ch)
            if existing_content_file:
                logger.info("Reusing existing File record (content_hash=%s, id=%s)", ch[:16], existing_content_file.id)
                content_file = existing_content_file
                try:
                    await self.storage.delete_file(object_name)
                except Exception:
                    pass
            else:
                content_file = await self.file_repository.create(
                    content_hash=ch,
                    file_path=object_name,
                    transcript_text=transcript_text,
                    media_metadata={"title": filename, "file_type": file_ext},
                    processing_status="completed",
                )
            user_file = await self.user_file_repository.create(
                user_id=user_id,
                file_id=content_file.id,
                namespace_id=namespace_id,
                custom_title=filename,
            )
            if self.sync_notifier:
                await self.sync_notifier.add_upsert_command_to_queue(
                    user_file_id=user_file.id,
                    user_id=user_id,
                    command_type=CommandType.UPSERT,
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

        # Вычисляем хэш от содержимого заранее, чтобы поймать дубликат до загрузки в MinIO
        ch = content_hash or self.compute_content_hash(file_content)

        if self.file_repository and self.user_file_repository:
            existing_content_file = await self.file_repository.get_by_content_hash(ch)
            if existing_content_file:
                # File уже существует — пропускаем MinIO и создание File.
                # Пробуем создать UserFile (может быть новое пространство).
                # Из-за ограничения uq_user_file(user_id, file_id) у пользователя не может быть
                # двух UserFile на один File, поэтому если запись уже есть — просто возвращаем её.
                is_new_user_file = True
                try:
                    user_file = await self.user_file_repository.create(
                        user_id=user_id,
                        file_id=existing_content_file.id,
                        namespace_id=namespace_id,
                        custom_title=filename,
                    )
                    if self.sync_notifier:
                        await self.sync_notifier.add_upsert_command_to_queue(
                            user_file_id=user_file.id,
                            user_id=user_id,
                            command_type=CommandType.UPSERT,
                        )
                    await self.db.commit()
                    logger.info(
                        "Reused existing File, created new UserFile: user_file_id=%d, content_file_id=%d (hash=%s)",
                        user_file.id, existing_content_file.id, ch[:16],
                    )
                except Exception:
                    await self.db.rollback()
                    user_file = await self.user_file_repository.find_by_user_and_file(
                        user_id, existing_content_file.id, namespace_id
                    )
                    if user_file is None:
                        raise
                    is_new_user_file = False
                    logger.info(
                        "Duplicate upload: reusing existing UserFile user_file_id=%d, content_file_id=%d (hash=%s)",
                        user_file.id, existing_content_file.id, ch[:16],
                    )
                return FileCreated(
                    file_id=user_file.id,
                    content_file_id=existing_content_file.id,
                    filename=user_file.custom_title or filename,
                    text="",
                    is_new_file=False,
                    is_new_user_file=is_new_user_file,
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
            content_hash=ch,
            transcript_text=text,
        )

        return FileCreated(
            file_id=user_file.id,
            content_file_id=user_file.file_id,
            filename=user_file.custom_title or filename,
            text=text,
            is_new_file=True,
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

        # S3/MinIO metadata допускает только строки (ASCII); content_type может прийти как enum ContentType
        content_type_str = content_type.value if hasattr(content_type, "value") else str(content_type)

        # Используем ASCII-имя для ключа S3/MinIO (кириллица в ключе может ломать загрузку)
        safe_name = f"content_{content_hash[:16]}.md"
        text_bytes = text.encode("utf-8")
        object_name = self.storage.generate_object_name(user_id, namespace_id, safe_name)
        try:
            await self.storage.upload_file(
                file_content=text_bytes,
                object_name=object_name,
                content_type="text/markdown",
                metadata={
                    "user_id": str(user_id),
                    "source_url": source_url,
                    "content_type": content_type_str,
                },
            )
            logger.info("[FileService] Saved extracted content to MinIO: %s", object_name)
        except Exception as e:
            raise ValidationError(f"Failed to upload to storage: {e}")

        try:
            content_file = await self.file_repository.get_by_content_hash(content_hash)
            if content_file is None:
                content_file = await self.file_repository.create(
                    content_hash=content_hash,
                    source_url=source_url,
                    transcript_text=text,
                    file_path=object_name,
                    media_metadata={"title": title, "content_type": content_type_str},
                    processing_status="completed",
                )
            else:
                logger.info("[FileService] Reusing existing file record: file_id=%d, hash=%s", content_file.id, content_hash[:16])
            user_file = await self.user_file_repository.create(
                user_id=user_id,
                file_id=content_file.id,
                namespace_id=namespace_id,
                custom_title=title,
            )
            if self.sync_notifier:
                await self.sync_notifier.add_upsert_command_to_queue(
                    user_file_id=user_file.id,
                    user_id=user_id,
                    command_type=CommandType.UPSERT,
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

    async def save_unattached_content(
        self,
        text: str,
        title: str,
        source_url: str,
        content_hash: str,
        content_type: str,
        user_id: int,
    ):
        """
        Сохраняет извлечённый контент без привязки к пользователю: создаёт File и загружает в MinIO,
        но НЕ создаёт запись UserFile. Файл можно позже привязать к пространству через attach.

        Returns:
            FileEntity созданного (или уже существующего) контент-файла.
        """
        if self.db is None or not self.file_repository:
            raise ValidationError("DB session и file_repository обязательны")

        existing = await self.file_repository.get_by_content_hash(content_hash)
        if existing:
            logger.info("[FileService] Unattached: reusing existing file by hash (file_id=%d)", existing.id)
            return existing

        existing_by_url = await self.file_repository.get_by_source_url(source_url)
        if existing_by_url:
            logger.info("[FileService] Unattached: reusing existing file by URL (file_id=%d)", existing_by_url.id)
            return existing_by_url

        content_type_str = content_type.value if hasattr(content_type, "value") else str(content_type)
        safe_name = f"content_{content_hash[:16]}.md"
        text_bytes = text.encode("utf-8")
        object_name = self.storage.generate_object_name(user_id, None, safe_name)

        try:
            await self.storage.upload_file(
                file_content=text_bytes,
                object_name=object_name,
                content_type="text/markdown",
                metadata={
                    "user_id": str(user_id),
                    "source_url": source_url,
                    "content_type": content_type_str,
                },
            )
            logger.info("[FileService] Unattached content uploaded to MinIO: %s", object_name)
        except Exception as e:
            raise ValidationError(f"Не удалось загрузить файл в хранилище: {e}")

        try:
            content_file = await self.file_repository.create(
                content_hash=content_hash,
                source_url=source_url,
                transcript_text=text,
                file_path=object_name,
                media_metadata={"title": title, "content_type": content_type_str},
                processing_status="completed",
            )
            await self.db.commit()
            logger.info("[FileService] Unattached content file created: file_id=%d, url=%s", content_file.id, source_url)
            return content_file
        except Exception as e:
            logger.error("[FileService] Failed to save unattached file to DB: %s", e)
            await self.db.rollback()
            try:
                await self.storage.delete_file(object_name)
            except Exception:
                pass
            raise ValidationError(f"Не удалось сохранить файл в БД: {e}")

    async def attach_file_to_namespace(
        self,
        content_file_id: int,
        user_id: int,
        namespace_id: int,
    ):
        """
        Привязывает контент-файл (File.id) к пространству пользователя, создавая запись UserFile.
        Если UserFile для этой пары (user_id, file_id) уже существует — обновляет namespace_id.

        Returns:
            UserFileEntity созданной или обновлённой записи.
        """
        if not self.file_repository or not self.user_file_repository or not self.namespace_repository:
            raise ValidationError("file_repository, user_file_repository и namespace_repository обязательны")

        content_file = await self.file_repository.get_by_id(content_file_id)
        if not content_file:
            raise NotFoundError(f"Файл с ID {content_file_id} не найден")

        namespace = await self.namespace_repository.get_by_id(namespace_id)
        if not namespace:
            raise NotFoundError(f"Пространство с ID {namespace_id} не найдено")
        if namespace.user_id != user_id:
            raise ForbiddenError("У вас нет доступа к этому пространству")

        existing = await self.user_file_repository.find_by_user_and_file(user_id, content_file_id)
        if existing:
            if existing.namespace_id != namespace_id:
                moved = await self.user_file_repository.update_namespace(existing.id, namespace_id)
                if not moved:
                    raise NotFoundError(f"Файл с ID {existing.id} не найден")
                if self.sync_notifier:
                    await self.sync_notifier.add_upsert_command_to_queue(
                        user_file_id=moved.id,
                        user_id=user_id,
                        command_type=CommandType.MOVE,
                    )
                await self.db.commit()
                logger.info(
                    "[FileService] Attach: moved existing user_file=%d to namespace=%d",
                    moved.id, namespace_id,
                )
                return await self.user_file_repository.get_by_id(moved.id)
            logger.info("[FileService] Attach: user_file=%d already in namespace=%d", existing.id, namespace_id)
            return existing

        title = (content_file.media_metadata or {}).get("title") or "Document"
        user_file = await self.user_file_repository.create(
            user_id=user_id,
            file_id=content_file_id,
            namespace_id=namespace_id,
            custom_title=title,
        )
        if self.sync_notifier:
            await self.sync_notifier.add_upsert_command_to_queue(
                user_file_id=user_file.id,
                user_id=user_id,
                command_type=CommandType.UPSERT,
            )
        await self.db.commit()
        logger.info(
            "[FileService] Attach: created user_file=%d for file=%d, namespace=%d",
            user_file.id, content_file_id, namespace_id,
        )
        return user_file

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
        meta = content_file.media_metadata or {}
        name_for_ext = uf.custom_title or meta.get("title") or "document"
        text = await self.file_content_service.get_text_content(
            content_file,
            filename=name_for_ext,
            strict=True,
        )
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
        return self._build_file_info(uf, content_file)

    async def get_file_version(
        self,
        file_id: int,
        user_id: int,
    ) -> FileVersionInfo:
        if self.sync_notifier:
            return await self.sync_notifier.get_file_version(file_id, user_id)  # type: ignore[return-value]
        info = await self.get_file_info(file_id, user_id)
        return FileVersionInfo(
            user_file_id=info.user_file_id,
            content_file_id=info.content_file_id,
            content_revision=info.content_revision,
            updated_at=info.updated_at,
            desktop_updated_at=info.desktop_updated_at,
            app_updated_at=info.app_updated_at,
            last_update_source=info.last_update_source,
            content_hash=info.content_hash,
            vault_relative_path=info.vault_relative_path,
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

        moved = await self.user_file_repository.update_namespace(file.id, namespace_id)
        if not moved:
            raise NotFoundError(f"Файл с id {file_id} не найден")
        if self.sync_notifier:
            await self.sync_notifier.add_upsert_command_to_queue(
                user_file_id=moved.id,
                user_id=user_id,
                command_type=CommandType.MOVE,
            )
        await self.db.commit()

        logger.info("[FileService] Moved file %d to namespace %d", file_id, namespace_id)
        return await self.get_file_info(file_id=moved.id, user_id=user_id)

    async def rename_file(
        self,
        file_id: int,
        user_id: int,
        new_title: str,
    ) -> FileInfo:
        if not self.user_file_repository or not self.db:
            raise ValidationError("user_file_repository required")
        if self.sync_notifier:
            await self.sync_notifier.add_rename_command_to_queue(
                user_file_id=file_id,
                user_id=user_id,
                new_title=new_title,
            )

        user_file = await self.user_file_repository.get_by_id(file_id)
        if not user_file:
            logger.error(f"Файл с id {file_id} не найден")
            await self.db.rollback()
            raise NotFoundError(f"Файл с id {file_id} не найден")
        if user_file.user_id != user_id:
            logger.error(f"У вас нет доступа к этому файлу: {file_id}")
            await self.db.rollback()
            raise ForbiddenError("У вас нет доступа к этому файлу")
        await self.user_file_repository.update_custom_title(file_id, new_title)
        await self.db.commit()
        return await self.get_file_info(file_id=file_id, user_id=user_id)

    async def replace_file_content(
        self,
        file_id: int,
        user_id: int,
        file_content: bytes,
        filename: str,
        *,
        base_hash: Optional[str] = None,
        force_overwrite: bool = False,
        summary_repository: Optional[SummaryRepository] = None,
        task_publisher: Optional[TaskPublisher] = None,
    ) -> FileInfo:
        """
        Заменяет содержимое существующего файла.

        Если на File ссылаются несколько UserFile (файл в нескольких пространствах),
        применяется Copy-on-Write: создаётся новый File, а текущий UserFile
        переключается на него. Остальные UserFile остаются нетронутыми.

        Проверяет владельца, перезаписывает файл в хранилище, обновляет метаданные в БД,
        инвалидирует суммаризацию, удаляет эмбеддинги и перезапускает конвейер индексации.
        """
        if not self.file_repository or not self.user_file_repository:
            raise ValidationError("file_repository and user_file_repository required")

        if self.sync_notifier:
            await self.sync_notifier.assert_can_save(
                user_file_id=file_id,
                user_id=user_id,
                base_hash=base_hash,
                force_overwrite=force_overwrite,
            )

        user_file = await self.user_file_repository.get_by_id(file_id)
        if not user_file:
            raise NotFoundError(f"File with id {file_id} not found")
        if user_file.user_id != user_id:
            raise ForbiddenError("You don't have access to this file")

        content_file = await self.file_repository.get_by_id(user_file.file_id)
        if not content_file:
            raise NotFoundError("File content record not found")
        if not content_file.file_path:
            raise NotFoundError("File path not found in storage")

        file_ext = await self._validate_upload_params(
            filename, file_content, user_id, user_file.namespace_id
        )
        text = self._extract_text_from_file(file_content, file_ext)
        if not text or not text.strip():
            raise ValidationError("File content is empty or could not be processed")

        new_hash = self.compute_content_hash(file_content)[:64]
        updated_meta = {
            **(content_file.media_metadata or {}),
            "title": filename,
            "file_type": file_ext,
            "file_size": len(file_content),
        }
        upload_metadata = {
            "user_id": str(user_id),
            "namespace_id": str(user_file.namespace_id or ""),
            "original_filename": base64.b64encode(filename.encode("utf-8")).decode("ascii"),
        }
        publisher = task_publisher or self.task_publisher

        # --- Copy-on-Write: если File используется несколькими UserFile ---
        ref_count = await self.user_file_repository.count_by_file_id(content_file.id)
        if ref_count > 1:
            logger.info(
                "[FileService] COW: File %d has %d refs, creating new File for UserFile %d",
                content_file.id, ref_count, user_file.id,
            )
            # Создаём новый файл в MinIO с новым путём
            new_object_name = self.storage.generate_object_name(
                user_id=user_id,
                namespace_id=user_file.namespace_id,
                filename=filename,
            )
            await self.storage.upload_file(
                file_content=file_content,
                object_name=new_object_name,
                content_type=self._get_content_type(file_ext),
                metadata=upload_metadata,
            )
            # Создаём новую запись File
            new_file = await self.file_repository.create(
                content_hash=new_hash,
                file_path=new_object_name,
                transcript_text=text,
                media_metadata=updated_meta,
                processing_status="pending",
            )
            # Переключаем UserFile на новый File
            await self.user_file_repository.update_file_id(user_file.id, new_file.id)
            # Обновляем custom_title
            if self.db:
                from sqlalchemy import update
                await self.db.execute(
                    update(UserFile).where(UserFile.id == user_file.id).values(custom_title=filename)
                )
            # Запускаем индексацию для нового File
            if publisher:
                publisher.send_embeddings_task(
                    content_file_id=new_file.id,
                    text=text,
                    namespace_id=user_file.namespace_id,
                    filename=filename,
                    user_file_id=user_file.id,
                )
                logger.info("[FileService] COW: sent embeddings task for new file %d", new_file.id)

            if self.sync_notifier:
                await self.sync_notifier.add_upsert_command_to_queue(
                    user_file_id=user_file.id,
                    user_id=user_id,
                    command_type=CommandType.UPSERT,
                )
            await self.db.commit()
            return await self.get_file_info(file_id=file_id, user_id=user_id)

        # --- Стандартный путь: единственный владелец, edit in-place ---
        await self.storage.upload_file(
            file_content=file_content,
            object_name=content_file.file_path,
            content_type=self._get_content_type(file_ext),
            metadata=upload_metadata,
        )

        await self.file_repository.update_content_metadata(
            file_id=content_file.id,
            content_hash=new_hash,
            media_metadata=updated_meta,
            transcript_text=text,
        )

        if self.db:
            from sqlalchemy import update
            await self.db.execute(
                update(UserFile).where(UserFile.id == user_file.id).values(custom_title=filename)
            )

        summary_repo = summary_repository or self.summary_repository
        if summary_repo:
            await summary_repo.delete_by_file_id(content_file.id)
            logger.info("[FileService] Invalidated summary for file %d", content_file.id)

        if self.vector_repository:
            await self.vector_repository.delete_by_file_id(content_file.id)
            logger.info("[FileService] Deleted embeddings for file %d", content_file.id)

        if publisher:
            publisher.send_embeddings_task(
                content_file_id=content_file.id,
                text=text,
                namespace_id=user_file.namespace_id,
                filename=filename,
                user_file_id=user_file.id,
            )
            logger.info("[FileService] Sent embeddings task for replaced file %d", file_id)

        if self.sync_notifier:
            await self.sync_notifier.add_upsert_command_to_queue(
                user_file_id=user_file.id,
                user_id=user_id,
                command_type=CommandType.UPSERT,
            )
        await self.db.commit()
        return await self.get_file_info(file_id=file_id, user_id=user_id)