import base64
import hashlib
import logging
import re
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, ForbiddenError, NotFoundError, ValidationError
from app.domain.protocols import (
    FileRepository,
    FileStorage,
    NamespaceRepository,
    SummaryRepository,
    TaskPublisher,
    UserFileRepository,
    VectorRepository,
    SyncRepository,
)
from app.infrastructure.db.models import File, Namespace, SyncCommand, UserFile
from app.schemas.file import (
    FileInfo,
    FileStructureItem,
    FileVersionInfo,
    NamespaceStructureItem,
    StructureResponse,
    SyncAckResponse,
    SyncCommandItem,
    SyncCommandsResponse,
    SyncUploadRequest,
    SyncUploadResponse,
    CommandStatus,
    CommandType,
)
from app.utils.file_readers import FileReaderFactory

from app.domain.entities import SyncCommandEntity

logger = logging.getLogger(__name__)


class SyncService:
    def __init__(
        self,
        *,
        db: AsyncSession,
        storage: FileStorage,
        file_repository: FileRepository,
        user_file_repository: UserFileRepository,
        namespace_repository: NamespaceRepository,
        sync_repository: SyncRepository,
        summary_repository: Optional[SummaryRepository] = None,
        vector_repository: Optional[VectorRepository] = None,
        task_publisher: Optional[TaskPublisher] = None,
        file_reader_factory: Optional[FileReaderFactory] = None,
    ) -> None:
        self.db = db
        self.storage = storage
        self.file_repository = file_repository
        self.user_file_repository = user_file_repository
        self.namespace_repository = namespace_repository
        self.sync_repository = sync_repository
        self.summary_repository = summary_repository
        self.vector_repository = vector_repository
        self.task_publisher = task_publisher
        self.file_reader_factory = file_reader_factory
        self.inbox_namespace_kind = "inbox"
        self.trash_namespace_name = "Trash"
        self.trash_namespace_kind = "trash"
        self.vault_root_namespace_name = "Vault"
        self.vault_root_namespace_kind = "vault_root"
        self.regular_namespace_kind = "regular"

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Очищает имя файла от недопустимых символов и возвращает его в безопасном формате
        Args:
            filename: Имя файла
        Returns:
            Имя файла в безопасном формате
        """
        unsafe = r'\\/:*?"<>|\x00-\x1f'
        result = re.sub(r"[" + unsafe + "]", "_", filename or "")
        result = " ".join(result.split()).strip()
        return result or "unnamed.md"

    @staticmethod
    def _file_ext(filename: str) -> str:
        """
        Возвращает расширение файла
        Args:
            filename: Имя файла
        Returns:
            Расширение файла
        """
        return filename.rsplit(".", 1)[-1].lower() if "." in filename else "md"

    @staticmethod
    def _compute_hash(content: bytes | str) -> str:
        """
        Вычисляет SHA-256 хэш контента
        Args:
            content: Контент
        Returns:
            SHA-256 хэш контента
        """
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _content_type(file_ext: str) -> str:
        """
        Возвращает тип контента по расширению файла
        Args:
            file_ext: Расширение файла
        Returns:
            Тип контента
        """
        content_types = {
            "txt": "text/plain",
            "md": "text/markdown",
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }
        return content_types.get(file_ext, "application/octet-stream")

    @staticmethod
    def _build_storage_metadata(*, user_id: int, namespace_id: Optional[int], filename: str) -> dict[str, str]:
        """
        Строит метаданные для загрузки файла в хранилище
        Args:
            user_id: ID пользователя
            namespace_id: ID пространства
            filename: Имя файла
        Returns:
            Метаданные для загрузки файла в хранилище
        """
        return {
            "user_id": str(user_id),
            "namespace_id": str(namespace_id or ""),
            "original_filename": base64.b64encode(filename.encode("utf-8")).decode("ascii"),
        }

    def _extract_text(self, file_bytes: bytes, file_ext: str) -> str:
        """
        Извлекает текст из файла
        Args:
            file_bytes: Контент файла
            file_ext: Расширение файла
        Returns:
            Текст из файла
        """
        
        if not file_bytes:
            return ""
        if self.file_reader_factory is not None:
            try:
                reader = self.file_reader_factory.get_reader(file_ext)
                return reader.read(file_bytes)
            except Exception:
                logger.warning("[SyncService] Failed to extract text for .%s", file_ext)
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return ""

    async def _resolve_namespace_from_path(
        self,
        *,
        user_id: int,
        vault_name: str,
        vault_relative_path: str,
    ) -> int:
        normalized_path = (vault_relative_path or "").replace("\\", "/").strip("/")
        path_parts = [part for part in normalized_path.split("/") if part]
        folder_parts = path_parts[:-1] if path_parts else []

        current_namespace = await self.namespace_repository.get_by_name_and_parent(
            user_id=user_id,
            parent_id=None,
            name=vault_name,
        )
        if current_namespace is None:
            current_namespace = await self.namespace_repository.create(
                name=vault_name,
                user_id=user_id,
                parent_id=None,
                kind=self.vault_root_namespace_kind,
                description=None,
            )
        elif current_namespace.kind != self.vault_root_namespace_kind:
            current_namespace.kind = self.vault_root_namespace_kind
            current_namespace = await self.namespace_repository.update(current_namespace)

        current_id = current_namespace.id
        for part in folder_parts:
            child_namespace = await self.namespace_repository.get_by_name_and_parent(
                user_id=user_id,
                parent_id=current_id,
                name=part,
            )
            if child_namespace is None:
                child_namespace = await self.namespace_repository.create(
                    name=part,
                    user_id=user_id,
                    parent_id=current_id,
                    kind=self.regular_namespace_kind,
                    description=None,
                )
            current_id = child_namespace.id

        return current_id




    def _build_file_info(self, user_file: UserFile, content_file: File) -> FileInfo:
        meta = content_file.media_metadata or {}
        file_size = meta.get("file_size") or len((content_file.transcript_text or "").encode("utf-8"))
        return FileInfo(
            user_file_id=user_file.id,
            content_file_id=user_file.file_id,
            user_id=user_file.user_id,
            namespace_id=user_file.namespace_id,
            filename=user_file.custom_title or meta.get("title", "document"),
            file_type=meta.get("file_type", "md"),
            file_size=file_size,
            created_at=content_file.created_at,
            updated_at=user_file.updated_at or content_file.created_at,
            file_path=content_file.file_path,
            desktop_updated_at=user_file.desktop_updated_at,
            app_updated_at=user_file.app_updated_at,
            last_update_source=user_file.last_update_source,
            content_hash=content_file.content_hash,
            vault_relative_path=user_file.vault_relative_path,
            is_conflict_copy=user_file.is_conflict_copy,
        )

    def _build_version_info(self, user_file: UserFile, content_file: File) -> FileVersionInfo:
        return FileVersionInfo(
            user_file_id=user_file.id,
            content_file_id=user_file.file_id,
            updated_at=user_file.updated_at or content_file.created_at,
            desktop_updated_at=user_file.desktop_updated_at,
            app_updated_at=user_file.app_updated_at,
            last_update_source=user_file.last_update_source,
            content_hash=content_file.content_hash,
            vault_relative_path=user_file.vault_relative_path,
        )




    async def get_file_version(self, user_file_id: int, user_id: int) -> FileVersionInfo:
        user_file = await self.user_file_repository.get_by_id(user_file_id)
        if not user_file:
            raise NotFoundError(f"Файл с id {user_file_id} не найден")
        if user_file.user_id != user_id:
            raise ForbiddenError("У вас нет доступа к этому файлу")
        content_file = await self.file_repository.get_by_id(user_file.file_id)
        if not content_file:
            raise NotFoundError(f"Файл с id {user_file.file_id} не найден")
        return self._build_version_info(user_file, content_file)

    async def _build_command_payload(self, user_file: UserFile, content_file: Optional[File]) -> dict:
        payload = {
            "user_file_id": user_file.id,
            "vault_relative_path": user_file.vault_relative_path,
            "namespace_id": user_file.namespace_id,
            "filename": user_file.custom_title or ((content_file.media_metadata or {}).get("title") if content_file else None),
            "updated_at": (user_file.updated_at or datetime.utcnow()).isoformat(),
            "last_update_source": user_file.last_update_source,
            "content_hash": content_file.content_hash if content_file else None,
        }
        if content_file is not None:
            payload["content"] = await self._get_text_content(content_file)
            if content_file.file_path:
                try:
                    binary = await self.storage.download_file(content_file.file_path)
                    payload["content_base64"] = base64.b64encode(binary).decode("ascii")
                    payload["content_encoding"] = "base64"
                except Exception:
                    logger.warning("[SyncService] Failed to add binary payload for user_file=%s", user_file.id)
        return payload


    def _touch_user_file(
        self,
        user_file: UserFile,
        *,
        source: str,
        vault_relative_path: Optional[str] = None,
    ) -> None:
        now = datetime.utcnow()
        user_file.updated_at = now
        user_file.last_update_source = source
        if source == "app":
            user_file.app_updated_at = now
        elif source == "desktop":
            user_file.desktop_updated_at = now
        if vault_relative_path:
            user_file.vault_relative_path = vault_relative_path

    async def add_upsert_command_to_queue(
        self,
        *,
        user_file_id: int,
        user_id: int,
        command_type: CommandType = CommandType.UPSERT.value,
        vault_relative_path: Optional[str] = None,
    ) -> Optional[SyncCommandEntity]:
        """
        Добавляет команду в список команд синхронизации ожидающих выполнения обновления файла
        Args:
            user_file_id: ID файла
            user_id: ID пользователя
            command_type: Тип команды
            vault_relative_path: Путь к файлу в хранилище
        Returns:
            Optional[SyncCommandEntity]: Команда синхронизации или None, если файл не найден или пользователь не имеет доступа к файлу
        """
    
        user_file = await self.user_file_repository.get_by_id(user_file_id)
        if not user_file:
            raise NotFoundError(f"Файл с id {user_file_id} не найден")
        if user_file.user_id != user_id:
            raise ForbiddenError("У вас нет доступа к этому файлу")
        content_file = await self.file_repository.get_by_id(user_file.file_id)
        if not content_file:
            raise NotFoundError(f"Файл с id {user_file.file_id} не найден")
        self._touch_user_file(
            user_file,
            source="app",
            vault_relative_path=vault_relative_path,
        )
        payload = await self._build_command_payload(user_file, content_file)
        command = await self.sync_repository.create_command(
            user_id=user_id,
            user_file_id=user_file.id,
            command_type=command_type.value,
            payload_json=payload,
            status=CommandStatus.PENDING.value,
        )
        return command

    async def add_trash_command_to_queue(
        self,
        *,
        user_file_id: int,
        user_id: int,
    ) -> Optional[SyncCommandEntity]:
        """
        Добавляет команду в список команд синхронизации ожидающих выполнения перемещения файла в корзину
        Args:
            user_file_id: ID файла
            user_id: ID пользователя
        Returns:
            Optional[SyncCommandEntity]: Команда синхронизации или None, если файл не найден или пользователь не имеет доступа к файлу
        """

        user_file = await self.user_file_repository.get_by_id(user_file_id)
        if not user_file:
            raise NotFoundError(f"Файл с id {user_file_id} не найден")

        trash_namespace_id = await self.namespace_repository.get_or_create_trash(user_id=user_id)
        await self.user_file_repository.update_namespace(user_file_id, trash_namespace_id)
        payload = {
            "user_file_id": user_file.id,
            "namespace_id": trash_namespace_id,
        }
        command = await self.sync_repository.create_command(
            user_id=user_id,
            user_file_id=user_file.id,
            command_type=CommandType.TRASH.value,
            payload_json=payload,
            status=CommandStatus.PENDING.value,
        )
        return command

    async def add_rename_command_to_queue(
        self,
        *,
        user_file_id: int,
        user_id: int,
        new_title: str,
    ) -> Optional[SyncCommandEntity]:
        """
        Добавляет команду в список команд синхронизации ожидающих выполнения переименования файла
        Args:
            user_file_id: ID файла
            user_id: ID пользователя
            new_title: Новое название файла
        Returns:
            Optional[SyncCommandEntity]: Команда синхронизации или None, если файл не найден или пользователь не имеет доступа к файлу
        """
        
        user_file = await self.user_file_repository.get_by_id(user_file_id)
        if not user_file:
            raise NotFoundError(f"Файл с id {user_file_id} не найден")
        if user_file.user_id != user_id:
            raise ForbiddenError("У вас нет доступа к этому файлу")
        user_file.custom_title = new_title
        content_file = await self.file_repository.get_by_id(user_file.file_id)
        if not content_file:
            raise NotFoundError(f"Файл с id {user_file.file_id} не найден")
        self._touch_user_file(user_file, source="app")
        payload = await self._build_command_payload(user_file, content_file)
        command = await self.sync_repository.create_command(
            user_id=user_id,
            user_file_id=user_file.id,
            command_type=CommandType.RENAME.value,
            payload_json=payload,
            status=CommandStatus.PENDING.value,
        )
        return command

    async def list_pending_commands(self, *, user_id: int, limit: int = 100) -> SyncCommandsResponse:
        result = await self.db.execute(
            select(SyncCommand)
            .where(SyncCommand.user_id == user_id, SyncCommand.status == CommandStatus.PENDING.value)
            .order_by(SyncCommand.created_at.asc())
            .limit(limit)
        )
        commands = result.scalars().all()
        return SyncCommandsResponse(
            commands=[
                SyncCommandItem(
                    id=cmd.id,
                    user_file_id=cmd.user_file_id or (cmd.payload_json or {}).get("user_file_id"),
                    command_type=cmd.command_type,
                    payload=cmd.payload_json or {},
                    status=cmd.status,
                    created_at=cmd.created_at,
                )
                for cmd in commands
            ]
        )

    async def ack_command(self, *, user_id: int, command_id: int, status: CommandStatus) -> SyncAckResponse:
        """
        Подтверждение выполнения или ошибки команды синхронизации
        Args:
            user_id: ID пользователя
            command_id: ID команды
            status: Статус команды
        Returns:
            SyncAckResponse: Ответ на подтверждение выполнения или ошибки команды синхронизации
        """
        command = await self.sync_repository.get_command(command_id, user_id)
        if command is None:
            raise NotFoundError(f"Команда с id {command_id} не найдена")
        command.status = status
        command.acked_at = datetime.utcnow()
        await self.sync_repository.commit()
        return SyncAckResponse(acked_at=command.acked_at, command_id=command_id, command_type=command.command_type, status=command.status)
        

    async def apply_desktop_delete(
        self,
        *,
        user_id: int,
        user_file_id: int,
    ) -> None:
        """
        Применение команды на удаление файла на сервере
        Args:
            user_id: ID пользователя
            user_file_id: ID файла
        Returns:
            None
        """
        user_file = await self.user_file_repository.get_by_id(user_file_id)
        if not user_file:
            raise NotFoundError(f"Файл с id {user_file_id} не найден")
        if user_file.user_id != user_id:
            raise ForbiddenError("У вас нет доступа к этому файлу")
        await self.user_file_repository.delete(user_file_id)
        await self.db.commit()

    async def _invalidate_related_artifacts(self, *, content_file_id: int, text: str, namespace_id: Optional[int], filename: str, user_file_id: int) -> None:
        if self.summary_repository:
            await self.summary_repository.delete_by_file_id(content_file_id)
        if self.vector_repository:
            await self.vector_repository.delete_by_file_id(content_file_id)
        if self.task_publisher:
            self.task_publisher.send_embeddings_task(
                content_file_id=content_file_id,
                text=text,
                namespace_id=namespace_id,
                filename=filename,
                user_file_id=user_file_id,
            )

    async def _create_conflict_copy(self, *, user_file: UserFile, content_file: File) -> Optional[UserFile]:
        conflict_title_base = user_file.custom_title or (content_file.media_metadata or {}).get("title") or "document.md"
        if "." in conflict_title_base:
            stem, ext = conflict_title_base.rsplit(".", 1)
            conflict_title = f"{stem}.conflict.{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{ext}"
        else:
            conflict_title = f"{conflict_title_base}.conflict.{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
        conflict_file = UserFile(
            user_id=user_file.user_id,
            file_id=user_file.file_id,
            namespace_id=user_file.namespace_id,
            custom_title=conflict_title,
            vault_relative_path=None,
            updated_at=datetime.utcnow(),
            desktop_updated_at=user_file.desktop_updated_at,
            app_updated_at=user_file.app_updated_at,
            last_update_source="app",
            is_conflict_copy=True,
            conflict_origin_user_file_id=user_file.id,
        )
        self.db.add(conflict_file)
        await self.db.flush()
        return conflict_file

    async def _apply_desktop_content(
        self,
        *,
        user_file: UserFile,
        content_file: File,
        file_bytes: bytes,
        content_text: str,
        content_hash: str,
        filename: str,
        vault_relative_path: str,
    ) -> FileInfo:
        file_ext = self._file_ext(filename)
        updated_meta = {
            **(content_file.media_metadata or {}),
            "title": filename,
            "file_type": file_ext,
            "file_size": len(file_bytes),
        }
        content_changed = content_file.content_hash != content_hash

        if not content_changed:
            user_file.custom_title = filename
            self._touch_user_file(
                user_file,
                source="desktop",
                vault_relative_path=vault_relative_path,
            )
            await self.db.flush()
            return self._build_file_info(user_file, content_file)

        existing_content_file = await self._get_content_file_by_hash(content_hash)
        if existing_content_file is not None and existing_content_file.id != content_file.id:
            user_file.file_id = existing_content_file.id
            content_file = existing_content_file
        else:
            ref_count = await self.user_file_repository.count_by_file_id(content_file.id)
            if ref_count > 1:
                new_object_name = self.storage.generate_object_name(
                    user_id=user_file.user_id,
                    namespace_id=user_file.namespace_id,
                    filename=self._sanitize_filename(filename),
                )
                await self.storage.upload_file(
                    file_content=file_bytes,
                    object_name=new_object_name,
                    content_type=self._content_type(file_ext),
                    metadata=self._build_storage_metadata(
                        user_id=user_file.user_id,
                        namespace_id=user_file.namespace_id,
                        filename=filename,
                    ),
                )
                new_content_file = await self.file_repository.create(
                    content_hash=content_hash,
                    file_path=new_object_name,
                    transcript_text=content_text,
                    media_metadata=updated_meta,
                    processing_status="completed",
                )
                user_file.file_id = new_content_file.id
                content_file = await self.file_repository.get_by_id(user_file.file_id)
                await self._invalidate_related_artifacts(
                    content_file_id=new_content_file.id,
                    text=content_text,
                    namespace_id=user_file.namespace_id,
                    filename=filename,
                    user_file_id=user_file.id,
                )
            else:
                if not content_file.file_path:
                    raise NotFoundError("Файл или путь не найден")
                await self.storage.upload_file(
                    file_content=file_bytes,
                    object_name=content_file.file_path,
                    content_type=self._content_type(file_ext),
                    metadata=self._build_storage_metadata(
                        user_id=user_file.user_id,
                        namespace_id=user_file.namespace_id,
                        filename=filename,
                    ),
                )
                await self.file_repository.update_content_metadata(
                    file_id=content_file.id,
                    content_hash=content_hash,
                    media_metadata=updated_meta,
                    transcript_text=content_text,
                )
                content_file = await self.file_repository.get_by_id(user_file.file_id)
                await self._invalidate_related_artifacts(
                    content_file_id=content_file.id,
                    text=content_text,
                    namespace_id=user_file.namespace_id,
                    filename=filename,
                    user_file_id=user_file.id,
                )

        user_file.custom_title = filename
        self._touch_user_file(
            user_file,
            source="desktop",
            vault_relative_path=vault_relative_path,
        )
        await self.db.flush()
        return self._build_file_info(user_file, content_file)

    async def upload_file(
        self,
        user_id: int,
        upload_request: SyncUploadRequest,
    ) -> SyncUploadResponse:
        """
        Загрузка файла от watcher'а. Если файл уже существует (по user_file_id)
        и содержимое изменилось — обновляет; иначе создаёт новый.
        """
        safe_filename = self._sanitize_filename(upload_request.filename)
        file_ext = self._file_ext(safe_filename)
        incoming_bytes = upload_request.file_bytes
        incoming_text = self._extract_text(incoming_bytes, file_ext)
        incoming_hash = upload_request.content_hash or self._compute_hash(incoming_bytes)

        if upload_request.user_file_id is not None:
            user_file = await self._get_user_file_model(upload_request.user_file_id, user_id)
            content_file = await self._get_content_file_model(user_file.file_id)

            if content_file.content_hash == incoming_hash:
                return SyncUploadResponse(
                    file=self._build_file_info(user_file, content_file),
                    created=False,
                )

            file_info = await self._apply_desktop_content(
                user_file=user_file,
                content_file=content_file,
                file_bytes=incoming_bytes,
                content_text=incoming_text,
                content_hash=incoming_hash,
                filename=safe_filename,
                vault_relative_path=upload_request.vault_relative_path,
            )
            await self.db.commit()
            return SyncUploadResponse(file=file_info, created=False)

        namespace_id = await self._resolve_namespace_from_path(
            user_id=user_id,
            vault_name=upload_request.vault_name or self.vault_root_namespace_name,
            vault_relative_path=upload_request.vault_relative_path,
        )

        object_name = self.storage.generate_object_name(
            user_id=user_id,
            namespace_id=namespace_id,
            filename=safe_filename,
        )
        await self.storage.upload_file(
            file_content=incoming_bytes,
            object_name=object_name,
            content_type=self._content_type(file_ext),
            metadata=self._build_storage_metadata(
                user_id=user_id,
                namespace_id=namespace_id,
                filename=safe_filename,
            ),
        )
        content_file_entity = await self.file_repository.create(
            content_hash=incoming_hash,
            transcript_text=incoming_text,
            file_path=object_name,
            media_metadata={
                "title": safe_filename,
                "file_type": file_ext,
                "file_size": len(incoming_bytes),
            },
            processing_status="completed",
        )
        content_file = await self._get_content_file_model(content_file_entity.id)
        new_user_file = UserFile(
            user_id=user_id,
            file_id=content_file.id,
            namespace_id=namespace_id,
            custom_title=safe_filename,
            vault_relative_path=upload_request.vault_relative_path,
            updated_at=datetime.utcnow(),
            desktop_updated_at=datetime.utcnow(),
            last_update_source="desktop",
        )
        self.db.add(new_user_file)
        await self.db.flush()
        await self.db.commit()
        return SyncUploadResponse(
            file=self._build_file_info(new_user_file, content_file),
            created=True,
        )

    async def assert_can_save(
        self,
        *,
        user_file_id: int,
        user_id: int,
        base_hash: Optional[str],
        force_overwrite: bool,
    ) -> None:
        """
        Проверяет, может ли пользователь сохранить файл.
        Если файл был изменен после открытия редактора, то выбрасывается ConflictError.
        Args:
            user_file_id: ID файла
            user_id: ID пользователя
            base_hash: Хеш файла
            force_overwrite: Обязательно ли перезаписывать файл
        Returns:
            None
        """
        if force_overwrite:
            return
        if base_hash is None:
            return
        version = await self.get_file_version(user_file_id, user_id)
        if version.content_hash != base_hash:
            raise ConflictError(
                "Файл был изменён после открытия редактора",
                payload={
                    "message": "Файл был изменён после открытия редактора",
                    "server": version.model_dump(mode="json"),
                },
            )

    async def get_namespaces_with_files(self, *, user_id: int) -> StructureResponse:
        """
        Получение списка пространств с файлами пользователя
        Args:
            user_id: ID пользователя
        Returns:
            StructureResponse: Список пространств с файлами пользователя
        """
        namespaces = await self.sync_repository.get_namespaces_with_files(user_id)
        items: list[NamespaceStructureItem] = []
        for ns in namespaces:
            files: list[FileStructureItem] = []
            for uf in ns.user_files:
                if uf.is_conflict_copy:
                    continue
                content_file = uf.file
                meta = (content_file.media_metadata or {}) if content_file else {}
                files.append(
                    FileStructureItem(
                        id=uf.id,
                        filename=uf.custom_title or meta.get("title") or "document",
                        file_size=meta.get("file_size") or len(
                            (content_file.transcript_text or "").encode("utf-8")
                        ) if content_file else 0,
                        updated_at=uf.updated_at or (
                            content_file.created_at if content_file else datetime.utcnow()
                        ),
                        content_hash=content_file.content_hash if content_file else None,
                        vault_relative_path=uf.vault_relative_path,
                    )
                )
            items.append(
                NamespaceStructureItem(
                    id=ns.id,
                    name=ns.name,
                    parent_id=ns.parent_id,
                    kind=ns.kind,
                    files=files,
                )
            )
        return StructureResponse(namespaces=items)


    async def apply_desktop_rename(
        self,
        *,
        user_id: int,
        user_file_id: int,
        new_name: str,
    ) -> None:
        """
        Применение команды на переименование файла на сервере
        Args:
            user_id: ID пользователя
            user_file_id: ID файла
            new_name: Новое название файла
        Returns:
            None

        Raises:
            NotFoundError: Файл не найден
            ForbiddenError: Нет доступа к файлу
        """
        
        user_file = await self.user_file_repository.get_by_id(user_file_id)
        if not user_file:
            raise NotFoundError(f"Файл с id {user_file_id} не найден")
        if user_file.user_id != user_id:
            raise ForbiddenError("У вас нет доступа к этому файлу")
        await self.user_file_repository.update_custom_title(user_file_id, new_name)
        await self.db.commit()



    async def apply_desktop_move(
        self,
        *,
        user_id: int,
        user_file_id: int,
        namespace_id: int,
    ) -> None:
        """
        Применение команды на перемещение файла на сервере
        Args:
            user_id: ID пользователя
            user_file_id: ID файла
            namespace_id: ID пространства
        Returns:
            None
        """
        
        user_file = await self.user_file_repository.get_by_id(user_file_id)
        if not user_file:
            raise NotFoundError(f"Файл с id {user_file_id} не найден")
        if user_file.user_id != user_id:
            raise ForbiddenError("У вас нет доступа к этому файлу")
        await self.user_file_repository.update_namespace(user_file_id, namespace_id)
        await self.db.commit()