from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.protocols import NamespaceRepository, UserFileRepository, FileSyncNotifier
from app.core.exceptions import NotFoundError, ValidationError, ForbiddenError
from app.core.namespace_constants import (
    INBOX_NAMESPACE_KIND,
    INBOX_NAMESPACE_NAME,
    REGULAR_NAMESPACE_KIND,
    TRASH_NAMESPACE_KIND,
    TRASH_NAMESPACE_NAME,
    VAULT_ROOT_NAMESPACE_KIND,
    VAULT_ROOT_NAMESPACE_NAME,
)
from app.domain.entities.namespace import NamespaceEntity, NamespaceFileItem
from app.schemas.file import FileStructureItem
from app.schemas.namespace import NamespaceStructureItem

class NamespaceService:
    """Сервис для работы с namespace"""

    def __init__(
        self,
        namespace_repository: NamespaceRepository,
        db: AsyncSession,
        user_file_repository: Optional[UserFileRepository] = None,
        sync_notifier: Optional[FileSyncNotifier] = None,
    ):
        """
        Args:
            namespace_repository: Репозиторий для работы с namespace
            db: Сессия БД
        """
        self.namespace_repository = namespace_repository
        self.user_file_repository = user_file_repository
        self.sync_notifier = sync_notifier
        self.db = db

    async def create_namespace(
        self,
        name: str,
        user_id: int,
        description: Optional[str],
        parent_id: Optional[int] = None,
        kind: str = "regular",
    ) -> NamespaceEntity:
        """Создает новый namespace для пользователя.
        Args:
            name: Имя namespace
            user_id: ID пользователя
            description: Описание namespace

        Returns:
            NamespaceEntity.
        """

        if kind == TRASH_NAMESPACE_KIND:
            raise ValidationError("Системное пространство Trash нельзя создать вручную")
        if kind == VAULT_ROOT_NAMESPACE_KIND:
            raise ValidationError("Системное пространство Vault нельзя создать вручную")
        if kind == REGULAR_NAMESPACE_KIND and parent_id is None:
            parent_id = await self.get_or_create_vault_root_namespace(user_id=user_id)
        if name == TRASH_NAMESPACE_NAME:
            raise ValidationError("Имя 'Trash' зарезервировано для системного пространства")
        existing = await self.namespace_repository.get_by_name_and_parent(
            user_id=user_id,
            parent_id=parent_id,
            name=name,
        )
        if existing:
            raise ValidationError(
                f"Namespace с именем '{name}' уже существует на этом уровне"
            )
        namespace = await self.namespace_repository.create(
            name=name,
            user_id=user_id,
            parent_id=parent_id,
            kind=kind,
            description=description,
        )
        await self.db.commit()
        return namespace

    async def get_or_create_trash(self, user_id: int) -> int:
        """
        Возвращает ID пространства «Trash» пользователя. Создаёт его, если ещё нет.
        Args:
            user_id: ID пользователя
        Returns:
            ID пространства «Trash».
        """
        vault_root_id = await self.get_or_create_vault_root_namespace(user_id=user_id)
        existing = await self.namespace_repository.get_by_name_and_parent(
            user_id=user_id,
            parent_id=vault_root_id,
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
            parent_id=vault_root_id,
            kind=TRASH_NAMESPACE_KIND,
            description=None,
        )
        await self.db.commit()
        return namespace.id

    async def get_or_create_inbox(self, user_id: int) -> int:
        """
        Возвращает ID пространства «Inbox» пользователя. Создаёт его, если ещё нет.
        Используется при загрузке файлов без указания пространства.

        Args:
            user_id: ID пользователя

        Returns:
            ID пространства «Inbox».
        """
        vault_root_id = await self.get_or_create_vault_root_namespace(user_id=user_id)
        existing = await self.namespace_repository.get_by_name_and_parent(
            user_id=user_id,
            parent_id=vault_root_id,
            name=INBOX_NAMESPACE_NAME,
        )
        if existing:
            if existing.kind == INBOX_NAMESPACE_KIND:
                return existing.id
            existing.kind = INBOX_NAMESPACE_KIND
            updated = await self.namespace_repository.update(existing)
            await self.db.commit()
            return updated.id

        namespace = await self.namespace_repository.create(
            name=INBOX_NAMESPACE_NAME,
            user_id=user_id,
            parent_id=vault_root_id,
            kind=INBOX_NAMESPACE_KIND,
            description=None,
        )
        await self.db.commit()
        return namespace.id

    async def get_or_create_vault_root_namespace(self, user_id: int) -> int:
        existing = await self.namespace_repository.get_by_name_and_parent(
            user_id=user_id,
            parent_id=None,
            name=VAULT_ROOT_NAMESPACE_NAME,
        )
        if existing:
            if existing.kind != VAULT_ROOT_NAMESPACE_KIND:
                existing.kind = VAULT_ROOT_NAMESPACE_KIND
                updated = await self.namespace_repository.update(existing)
                await self.db.commit()
                return updated.id
            return existing.id

        namespace = await self.namespace_repository.create(
            name=VAULT_ROOT_NAMESPACE_NAME,
            user_id=user_id,
            parent_id=None,
            kind=VAULT_ROOT_NAMESPACE_KIND,
            description=None,
        )
        await self.db.commit()
        return namespace.id

    
    async def get_descendant_ids(self, *, user_id: int, namespace_id: int) -> list[int]:
        namespace = await self.get_namespace(namespace_id=namespace_id, user_id=user_id)
        ids = await self.namespace_repository.get_descendant_ids(
            user_id=user_id,
            namespace_id=namespace.id,
        )
        return ids

    async def get_namespace(
        self,
        namespace_id: int,
        user_id: int,
    ) -> NamespaceEntity:
        """
        Получает namespace по ID с проверкой доступа.

        Args:
            namespace_id: ID namespace
            user_id: ID пользователя

        Returns:
            NamespaceEntity.
        """
        namespace = await self.namespace_repository.get_by_id(namespace_id)
        if not namespace:
            raise NotFoundError(f"Namespace с ID {namespace_id} не найден")
        if namespace.user_id != user_id:
            raise ForbiddenError("Нет доступа к данному namespace")
        return namespace

    async def get_namespace_files(
        self,
        namespace_id: int,
    ) -> list[NamespaceFileItem]:
        """Список файлов в пространстве (с путём и метаданными для URL/удаления)."""
        return await self.namespace_repository.get_files_by_namespace_id(namespace_id)

    async def get_namespace_file_paths(self, namespace_id: int) -> list[str]:
        """Пути в хранилище (MinIO) для файлов пространства — для удаления при удалении namespace."""
        files = await self.namespace_repository.get_files_by_namespace_id(namespace_id)
        return [f.file_path for f in files if f.file_path]

    async def get_user_namespaces_with_files(self, user_id: int) -> list[NamespaceEntity]:
        """Список namespace пользователя с файлами.
        Args:
            user_id: ID пользователя

        Returns:
            list[NamespaceEntity].
        """
        namespaces = await self.namespace_repository.get_by_user_with_files(user_id)
        return namespaces

    async def get_user_namespaces(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[NamespaceEntity], int]:
        """Список namespace пользователя с пагинацией."""
        namespaces_with_count, total = await self.namespace_repository.get_by_user(
            user_id=user_id, skip=skip, limit=limit
        )
        return namespaces_with_count, total

    async def update_namespace(
        self,
        namespace_id: int,
        user_id: int,
        name: Optional[str],
        description: Optional[str],
        notify_watcher: bool = True,
    ) -> NamespaceEntity:
        """Обновляет namespace.
        Args:
            namespace_id: ID namespace
            user_id: ID пользователя
            name: Имя namespace
            description: Описание namespace

        Returns:
            NamespaceEntity.
        """
        namespace = await self.namespace_repository.get_by_id(namespace_id)
        if not namespace:
            raise NotFoundError(f"Namespace с ID {namespace_id} не найден")
        if namespace.user_id != user_id:
            raise ForbiddenError("Нет доступа к данному namespace")
        if namespace.kind == VAULT_ROOT_NAMESPACE_KIND:
            raise ValidationError("Системное пространство Vault нельзя изменять")
        if namespace.kind == TRASH_NAMESPACE_KIND:
            raise ValidationError("Системное пространство Trash нельзя изменять")
        old_name = namespace.name
        name_changed = False
        if name and name != namespace.name:
            existing = await self.namespace_repository.get_by_name_and_parent(
                user_id=user_id,
                parent_id=namespace.parent_id,
                name=name,
            )
            if existing and existing.id != namespace_id:
                raise ValidationError(
                    f"Namespace с именем '{name}' уже существует на этом уровне"
                )
            namespace.name = name
            name_changed = True
        if description is not None:
            namespace.description = description
        entity = await self.namespace_repository.update(namespace)
        if self.sync_notifier and notify_watcher and name_changed and entity.name != old_name:
            await self.sync_notifier.add_rename_namespace_command_to_queue(
                namespace_id=entity.id,
                user_id=user_id,
                new_name=entity.name,
            )
        await self.db.commit()
        return entity

    async def move_namespace(
        self,
        *,
        namespace_id: int,
        user_id: int,
        target_parent_id: int,
        notify_watcher: bool = True,
    ) -> NamespaceEntity:
        """Перемещает namespace под нового родителя."""
        namespace = await self.namespace_repository.get_by_id(namespace_id)
        if not namespace:
            raise NotFoundError(f"Namespace с ID {namespace_id} не найден")
        if namespace.user_id != user_id:
            raise ForbiddenError("Нет доступа к данному namespace")
        if namespace.kind == VAULT_ROOT_NAMESPACE_KIND:
            raise ValidationError("Системное пространство Vault нельзя перемещать")
        if namespace.kind == TRASH_NAMESPACE_KIND:
            raise ValidationError("Системное пространство Trash нельзя перемещать")
        if namespace.kind == INBOX_NAMESPACE_KIND:
            raise ValidationError("Системное пространство Inbox нельзя перемещать")
        if namespace.id == target_parent_id:
            raise ValidationError("Нельзя переместить пространство в само себя")

        target_parent = await self.namespace_repository.get_by_id(target_parent_id)
        if not target_parent:
            raise NotFoundError(f"Родительское пространство с ID {target_parent_id} не найдено")
        if target_parent.user_id != user_id:
            raise ForbiddenError("Нет доступа к целевому пространству")
        if target_parent.kind == TRASH_NAMESPACE_KIND:
            raise ValidationError("Нельзя переместить пространство внутрь Trash")
        if target_parent.kind == INBOX_NAMESPACE_KIND:
            raise ValidationError("Нельзя переместить пространство внутрь Inbox")

        descendant_ids = await self.namespace_repository.get_descendant_ids(
            user_id=user_id,
            namespace_id=namespace_id,
        )
        if target_parent_id in descendant_ids:
            raise ValidationError("Нельзя переместить пространство внутрь собственного потомка")

        existing = await self.namespace_repository.get_by_name_and_parent(
            user_id=user_id,
            parent_id=target_parent_id,
            name=namespace.name,
        )
        if existing and existing.id != namespace.id:
            raise ValidationError(
                f"Namespace с именем '{namespace.name}' уже существует на этом уровне"
            )

        if namespace.parent_id == target_parent_id:
            return namespace

        namespace.parent_id = target_parent_id
        entity = await self.namespace_repository.update(namespace)
        if self.sync_notifier and notify_watcher:
            await self.sync_notifier.add_move_namespace_command_to_queue(
                namespace_id=entity.id,
                user_id=user_id,
                target_parent_id=target_parent_id,
            )
        await self.db.commit()
        return entity

    async def delete_namespace(
        self,
        namespace_id: int,
        user_id: int,
    ) -> NamespaceEntity:
        """Удаляет namespace. Пространства Inbox и Trash удалять нельзя."""
        namespace = await self.namespace_repository.get_by_id(namespace_id)
        if not namespace:
            raise NotFoundError(f"Namespace с ID {namespace_id} не найден")
        if namespace.user_id != user_id:
            raise ForbiddenError("Нет доступа к данному namespace")
        if namespace.kind == INBOX_NAMESPACE_KIND:
            raise ValidationError("Пространство «Inbox» нельзя удалить")
        if namespace.kind == TRASH_NAMESPACE_KIND:
            raise ValidationError("Пространство «Trash» нельзя удалить")
        if namespace.kind == VAULT_ROOT_NAMESPACE_KIND:
            raise ValidationError("Пространство «Vault» нельзя удалить")

        if self.user_file_repository:
            trash_namespace_id = await self.get_or_create_trash(user_id)
            file_ids = await self.user_file_repository.list_ids_by_user_and_namespace(
                user_id=user_id,
                namespace_id=namespace_id,
            )
            for file_id in file_ids:
                if self.sync_notifier:
                    await self.sync_notifier.add_trash_command_to_queue(
                        user_file_id=file_id,
                        user_id=user_id,
                    )
                else:
                    await self.user_file_repository.update_namespace(
                        file_id,
                        trash_namespace_id,
                    )

        if self.sync_notifier:
            await self.sync_notifier.add_delete_namespace_command_to_queue(
                namespace_id=namespace_id,
                user_id=user_id,
            )

        await self.namespace_repository.delete(namespace_id)
        await self.db.commit()
        return namespace

    
    
    
    async def get_namespaces_with_files(self, *, user_id: int) -> list[NamespaceStructureItem]:
        """
        Получение списка пространств с файлами пользователя
        Args:
            user_id: ID пользователя
        Returns:
            list[NamespaceStructureItem]: Список пространств с файлами пользователя
        """
        namespaces = await self.namespace_repository.get_namespaces_with_files(user_id)
        items: list[NamespaceStructureItem] = []
        for ns in namespaces:
            files: list[FileStructureItem] = []
            for uf in ns.user_files:
                if uf.is_conflict_copy:
                    continue
                files.append(
                    FileStructureItem(
                        id=uf.id,
                        filename=uf.filename or "document",
                        file_size=uf.file_size,
                        updated_at=uf.updated_at or uf.created_at or datetime.utcnow(),
                        content_hash=uf.content_hash,
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
        return items