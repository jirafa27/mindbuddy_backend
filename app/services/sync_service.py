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
from app.infrastructure.db.models import File, SyncCommand, UserFile
from app.schemas.namespace import NamespaceStructureItem
from app.schemas.file import (
    FileInfo,
    FileVersionInfo,
    SyncAckResponse,
    SyncCommandItem,
    SyncUploadRequest,
    SyncUploadResponse,
    CommandStatus,
    CommandType,
)
from app.services.file_content_service import FileContentService
from app.utils.file_readers import FileReaderFactory

from app.domain.entities import FileEntity, NamespaceEntity, SyncCommandEntity, UserFileEntity

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
        file_content_service: Optional[FileContentService] = None,
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
        self.file_content_service = file_content_service or FileContentService(
            storage=storage,
            file_reader_factory=file_reader_factory,
        )
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
    def _ext_from_content_type(content_type: Optional[str]) -> Optional[str]:
        mapping = {
            "text/plain": "txt",
            "text/markdown": "md",
            "text/x-markdown": "md",
            "application/pdf": "pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        }
        if not content_type:
            return None
        return mapping.get(str(content_type).split(";", 1)[0].strip().lower())

    def _ensure_filename_extension(
        self,
        filename: Optional[str],
        content_file: Optional[File],
    ) -> Optional[str]:
        if not filename:
            return filename
        if "." in filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]:
            return filename
        if content_file is None:
            return filename

        meta = content_file.media_metadata or {}
        file_ext = meta.get("file_type") or self._ext_from_content_type(meta.get("content_type"))
        if not file_ext:
            return filename
        return f"{filename}.{file_ext}"

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

    async def _resolve_namespace_from_path(
        self,
        *,
        user_id: int,
        vault_name: str,
        folder_parts: list[str],
    ) -> int:
        root = await self.namespace_repository.get_by_name_and_parent(
            user_id=user_id,
            parent_id=None,
            name=vault_name,
        )
        if root is None:
            root = await self.namespace_repository.create(
                name=vault_name,
                user_id=user_id,
                parent_id=None,
                kind=self.vault_root_namespace_kind,
                description=None,
            )

        current_id = root.id
        for part in folder_parts:
            node = await self.namespace_repository.get_by_name_and_parent(
                user_id=user_id,
                parent_id=current_id,
                name=part,
            )
            if node is None:
                node = await self.namespace_repository.create(
                    name=part,
                    user_id=user_id,
                    parent_id=current_id,
                    kind=self.regular_namespace_kind,
                    description=None,
                )
            current_id = node.id
        return current_id

    async def _get_or_create_trash_namespace_id(self, *, user_id: int) -> int:
        existing = await self.namespace_repository.get_by_name_and_user(
            name=self.trash_namespace_name,
            user_id=user_id,
        )
        if existing and existing.parent_id is None:
            if existing.kind != self.trash_namespace_kind:
                existing.kind = self.trash_namespace_kind
                updated = await self.namespace_repository.update(existing)
                return updated.id
            return existing.id

        namespace = await self.namespace_repository.create(
            name=self.trash_namespace_name,
            user_id=user_id,
            parent_id=None,
            kind=self.trash_namespace_kind,
            description=None,
        )
        return namespace.id

    async def _get_user_file_entity(self, user_file_id: int, user_id: int) -> UserFileEntity:
        user_file = await self.user_file_repository.get_by_id(user_file_id)
        if user_file is None:
            raise NotFoundError(f"Файл с id {user_file_id} не найден")
        if user_file.user_id != user_id:
            raise ForbiddenError("У вас нет доступа к этому файлу")
        return user_file

    async def _get_content_file_entity(self, file_id: int) -> FileEntity:
        content_file = await self.file_repository.get_by_id(file_id)
        if content_file is None:
            raise NotFoundError(f"Файл с id {file_id} не найден")
        return content_file




    def _build_file_info(self, user_file: UserFileEntity, content_file: FileEntity) -> FileInfo:
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

    def _build_version_info(self, user_file: UserFileEntity, content_file: FileEntity) -> FileVersionInfo:
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

    async def _build_command_payload(self, user_file: UserFileEntity, content_file: FileEntity) -> dict:
        resolved_filename = self._ensure_filename_extension(
            user_file.custom_title or ((content_file.media_metadata or {}).get("title") if content_file else None),
            content_file,
        )
        payload = {
            "user_file_id": user_file.id,
            "vault_relative_path": user_file.vault_relative_path,
            "namespace_id": user_file.namespace_id,
            "filename": resolved_filename,
            "updated_at": (user_file.updated_at or datetime.utcnow()).isoformat(),
            "last_update_source": user_file.last_update_source,
            "content_hash": content_file.content_hash,
        }
        payload["content"] = await self.file_content_service.get_text_content(
            content_file,
            strict=False,
        )
        if content_file.file_path:
            try:
                binary = await self.storage.download_file(content_file.file_path)
                payload["content_base64"] = base64.b64encode(binary).decode("ascii")
                payload["content_encoding"] = "base64"
            except Exception:
                logger.warning("[SyncService] Failed to add binary payload for user_file=%s", user_file.id)
        return payload


    def _build_user_file_touch_updates(
        self,
        *,
        source: str,
        vault_relative_path: Optional[str] = None,
    ) -> dict:
        now = datetime.utcnow()
        updates = {
            "updated_at": now,
            "last_update_source": source,
        }
        if source == "app":
            updates["app_updated_at"] = now
        elif source == "desktop":
            updates["desktop_updated_at"] = now
        if vault_relative_path is not None:
            updates["vault_relative_path"] = vault_relative_path
        return updates

    async def _build_namespace_delete_payload(self, namespace: NamespaceEntity) -> dict:
        parts: list[str] = []
        current: Optional[NamespaceEntity] = namespace
        vault_name = self.vault_root_namespace_name

        while current is not None:
            if current.kind == self.vault_root_namespace_kind:
                vault_name = current.name
                break

            parts.append(current.name)
            if current.parent_id is None:
                break
            current = await self.namespace_repository.get_by_id(current.parent_id)

        return {
            "namespace_id": namespace.id,
            "target_type": "namespace",
            "vault_name": vault_name,
            "relative_path": "/".join(reversed(parts)),
        }

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
        user_file = await self.user_file_repository.update_sync_metadata(
            user_file.id,
            **self._build_user_file_touch_updates(
                source="app",
                vault_relative_path=vault_relative_path,
            ),
        )
        if not user_file:
            raise NotFoundError(f"Файл с id {user_file_id} не найден")
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
        if user_file.user_id != user_id:
            raise ForbiddenError("У вас нет доступа к этому файлу")

        trash_namespace_id = await self._get_or_create_trash_namespace_id(user_id=user_id)
        existing_in_trash = await self.user_file_repository.find_by_user_and_file(
            user_id,
            user_file.file_id,
            trash_namespace_id,
        )

        if existing_in_trash is not None and existing_in_trash.id != user_file.id:
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
            await self.user_file_repository.delete(user_file.id)
            return command

        moved_user_file = await self.user_file_repository.update_namespace(user_file_id, trash_namespace_id)
        if not moved_user_file:
            raise NotFoundError(f"Файл с id {user_file_id} не найден")
        payload = {
            "user_file_id": moved_user_file.id,
            "namespace_id": trash_namespace_id,
        }
        command = await self.sync_repository.create_command(
            user_id=user_id,
            user_file_id=moved_user_file.id,
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
        content_file = await self.file_repository.get_by_id(user_file.file_id)
        if not content_file:
            raise NotFoundError(f"Файл с id {user_file.file_id} не найден")
        user_file = await self.user_file_repository.update_sync_metadata(
            user_file.id,
            custom_title=new_title,
            **self._build_user_file_touch_updates(source="app"),
        )
        if not user_file:
            raise NotFoundError(f"Файл с id {user_file_id} не найден")
        payload = await self._build_command_payload(user_file, content_file)
        command = await self.sync_repository.create_command(
            user_id=user_id,
            user_file_id=user_file.id,
            command_type=CommandType.RENAME.value,
            payload_json=payload,
            status=CommandStatus.PENDING.value,
        )
        return command

    async def add_delete_namespace_command_to_queue(
        self,
        *,
        namespace_id: int,
        user_id: int,
    ) -> Optional[SyncCommandEntity]:
        namespace = await self.namespace_repository.get_by_id(namespace_id)
        if not namespace:
            raise NotFoundError(f"Пространство с id {namespace_id} не найдено")
        if namespace.user_id != user_id:
            raise ForbiddenError("У вас нет доступа к этому пространству")

        payload = await self._build_namespace_delete_payload(namespace)
        command = await self.sync_repository.create_command(
            user_id=user_id,
            user_file_id=None,
            command_type=CommandType.DELETE.value,
            payload_json=payload,
            status=CommandStatus.PENDING.value,
        )
        return command

    async def list_pending_commands(self, *, user_id: int, limit: int = 100) -> list[SyncCommandItem]:
        result = await self.db.execute(
            select(SyncCommand)
            .where(SyncCommand.user_id == user_id, SyncCommand.status == CommandStatus.PENDING.value)
            .order_by(SyncCommand.created_at.asc())
            .limit(limit)
        )
        commands = result.scalars().all()
        return [
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

    async def apply_desktop_delete_namespace(
        self,
        *,
        user_id: int,
        namespace_id: int,
    ) -> None:
        """
        Полное удаление namespace-дерева по запросу watcher'а.
        Удаляет только user_files и namespaces пользователя; общие files не трогает.
        """
        namespace = await self.namespace_repository.get_by_id(namespace_id)
        if not namespace:
            raise NotFoundError(f"Пространство с id {namespace_id} не найдено")
        if namespace.user_id != user_id:
            raise ForbiddenError("У вас нет доступа к этому пространству")
        if namespace.kind == self.inbox_namespace_kind or namespace.kind == self.trash_namespace_kind:
            raise ValidationError(f"Пространство {namespace.kind} нельзя удалить через sync")

        descendant_ids = await self.namespace_repository.get_descendant_ids(
            user_id=user_id,
            namespace_id=namespace_id,
        )
        if not descendant_ids:
            descendant_ids = [namespace_id]

        user_file_ids = await self.user_file_repository.list_ids_by_user_and_namespace(
            user_id=user_id,
            namespace_id=namespace_id,
        )
        for user_file_id in user_file_ids:
            await self.user_file_repository.delete(user_file_id)

        subtree_namespaces: dict[int, NamespaceEntity] = {}
        for descendant_id in descendant_ids:
            descendant = await self.namespace_repository.get_by_id(descendant_id)
            if descendant is not None:
                subtree_namespaces[descendant_id] = descendant

        def _depth(ns_id: int) -> int:
            depth = 0
            current = subtree_namespaces.get(ns_id)
            while current is not None and current.parent_id in subtree_namespaces:
                depth += 1
                current = subtree_namespaces.get(current.parent_id)
            return depth

        for descendant_id in sorted(subtree_namespaces.keys(), key=_depth, reverse=True):
            await self.namespace_repository.delete(descendant_id)

        await self.db.commit()

    async def create_desktop_namespace(
        self,
        *,
        user_id: int,
        name: Optional[str] = None,
        vault_name: Optional[str] = None,
        parent_id: Optional[int] = None,
        description: Optional[str] = None,
        relative_path: Optional[str] = None,
    ) -> NamespaceStructureItem:
        if relative_path:
            effective_vault_name = " ".join((vault_name or "").split()).strip() or self.vault_root_namespace_name
            normalized_relative_path = relative_path.replace("\\", "/").strip("/")
            raw_parts = normalized_relative_path.split("/")
            normalized_parts = [" ".join(part.split()).strip() for part in raw_parts if part.strip()]
            if any(part == ".." for part in normalized_parts):
                raise ValidationError("relative_path не должен содержать '..'")
            if not normalized_parts:
                raise ValidationError("relative_path не может быть пустым")

            normalized_name = normalized_parts[-1]
            provided_name = " ".join((name or "").split()).strip()
            if provided_name and provided_name != normalized_name:
                raise ValidationError("name должен совпадать с последним сегментом relative_path")
            name = normalized_name

            if len(normalized_parts) == 1:
                parent_id = await self._resolve_namespace_from_path(
                    user_id=user_id,
                    vault_name=effective_vault_name,
                    folder_parts=[],
                )
            else:
                parent_id = await self._resolve_namespace_from_path(
                    user_id=user_id,
                    vault_name=effective_vault_name,
                    folder_parts=normalized_parts[:-1],
                )

        normalized_name = " ".join((name or "").split()).strip()
        if not normalized_name:
            raise ValidationError("Нужно передать name или relative_path")
        if parent_id is None and normalized_name == self.trash_namespace_name:
            raise ValidationError("Имя 'Trash' зарезервировано для системного пространства")

        if parent_id is not None:
            parent = await self.namespace_repository.get_by_id(parent_id)
            if not parent:
                raise NotFoundError(f"Пространство с id {parent_id} не найдено")
            if parent.user_id != user_id:
                raise ForbiddenError("У вас нет доступа к родительскому пространству")
            if parent.kind == self.trash_namespace_kind:
                raise ValidationError("В пространстве Trash нельзя создавать папки через sync")

        existing = await self.namespace_repository.get_by_name_and_parent(
            user_id=user_id,
            parent_id=parent_id,
            name=normalized_name,
        )
        if existing is not None:
            return NamespaceStructureItem(
                id=existing.id,
                name=existing.name,
                parent_id=existing.parent_id,
                kind="regular" if existing.kind == self.vault_root_namespace_kind else existing.kind,
                files=[],
            )

        created = await self.namespace_repository.create(
            name=normalized_name,
            user_id=user_id,
            parent_id=parent_id,
            kind=self.regular_namespace_kind,
            description=description,
        )
        await self.db.commit()
        return NamespaceStructureItem(
            id=created.id,
            name=created.name,
            parent_id=created.parent_id,
            kind=created.kind,
            files=[],
        )

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
        user_file: UserFileEntity,
        content_file: FileEntity,
        file_bytes: bytes,
        content_text: str,
        content_hash: str,
        filename: str,
        vault_relative_path: str,
    ) -> FileInfo:
        touch_updates = self._build_user_file_touch_updates(
            source="desktop",
            vault_relative_path=vault_relative_path,
        )
        file_ext = self._file_ext(filename)
        updated_meta = {
            **(content_file.media_metadata or {}),
            "title": filename,
            "file_type": file_ext,
            "file_size": len(file_bytes),
        }
        content_changed = content_file.content_hash != content_hash

        if not content_changed:
            updated_user_file = await self.user_file_repository.update_sync_metadata(
                user_file.id,
                custom_title=filename,
                **touch_updates,
            )
            if not updated_user_file:
                raise NotFoundError(f"Файл с id {user_file.id} не найден")
            return self._build_file_info(updated_user_file, content_file)

        existing_content_file = await self.file_repository.get_by_content_hash(content_hash)
        if existing_content_file is not None and existing_content_file.id != content_file.id:
            content_file = existing_content_file
            updated_user_file = await self.user_file_repository.update_sync_metadata(
                user_file.id,
                file_id=existing_content_file.id,
                custom_title=filename,
                **touch_updates,
            )
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
                content_file = new_content_file
                updated_user_file = await self.user_file_repository.update_sync_metadata(
                    user_file.id,
                    file_id=new_content_file.id,
                    custom_title=filename,
                    **touch_updates,
                )
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
                content_file = await self.file_repository.get_by_id(content_file.id)
                if not content_file:
                    raise NotFoundError(f"Файл с id {user_file.file_id} не найден")
                updated_user_file = await self.user_file_repository.update_sync_metadata(
                    user_file.id,
                    custom_title=filename,
                    **touch_updates,
                )
                await self._invalidate_related_artifacts(
                    content_file_id=content_file.id,
                    text=content_text,
                    namespace_id=user_file.namespace_id,
                    filename=filename,
                    user_file_id=user_file.id,
                )

        if not updated_user_file:
            raise NotFoundError(f"Файл с id {user_file.id} не найден")
        return self._build_file_info(updated_user_file, content_file)

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
        incoming_text = self.file_content_service.extract_text(
            incoming_bytes,
            filename=safe_filename,
            strict=False,
        )
        incoming_hash = self._compute_hash(incoming_bytes)

        if upload_request.user_file_id is not None:
            user_file = await self._get_user_file_entity(upload_request.user_file_id, user_id)
            content_file = await self._get_content_file_entity(user_file.file_id)

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

        vault_name = upload_request.vault_name or self.vault_root_namespace_name
        vault_path = upload_request.vault_relative_path.replace("\\", "/")
        parts = vault_path.split("/")
        # убираем имя файла, оставляем только директории
        folder_parts = parts[:-1] if len(parts) > 1 else []

        namespace_id = await self._resolve_namespace_from_path(
            user_id=user_id,
            vault_name=vault_name,
            folder_parts=folder_parts,
        )

        content_file_entity = await self.file_repository.get_by_content_hash(incoming_hash)
        if content_file_entity is None:
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
        now = datetime.utcnow()
        existing_user_file = await self.user_file_repository.find_by_user_and_file(
            user_id=user_id,
            file_id=content_file_entity.id,
            namespace_id=namespace_id,
        )
        if existing_user_file is not None:
            user_file = await self.user_file_repository.update_sync_metadata(
                existing_user_file.id,
                custom_title=safe_filename,
                vault_relative_path=upload_request.vault_relative_path,
                updated_at=now,
                desktop_updated_at=now,
                last_update_source="desktop",
            )
            if not user_file:
                raise NotFoundError(f"Файл с id {existing_user_file.id} не найден")
            created = False
        else:
            user_file = await self.user_file_repository.create(
                user_id=user_id,
                file_id=content_file_entity.id,
                namespace_id=namespace_id,
                custom_title=safe_filename,
                vault_relative_path=upload_request.vault_relative_path,
                updated_at=now,
                desktop_updated_at=now,
                last_update_source="desktop",
            )
            created = True
        await self.db.commit()
        return SyncUploadResponse(
            file=self._build_file_info(user_file, content_file_entity),
            created=created,
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