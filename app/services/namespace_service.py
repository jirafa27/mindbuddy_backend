from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.protocols import NamespaceRepository, UserFileRepository, FileSyncNotifier
from app.core.exceptions import NotFoundError, ValidationError, ForbiddenError
from app.domain.entities.namespace import NamespaceEntity, NamespaceFileItem

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
        self.inbox_namespace_name = "Inbox"
        self.inbox_namespace_kind = "inbox"
        self.trash_namespace_name = "Trash"
        self.trash_namespace_kind = "trash"
        self.vault_root_namespace_kind = "vault_root"
        self.regular_namespace_kind = "regular"
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

        if kind == self.trash_namespace_kind:
            raise ValidationError("Системное пространство Trash нельзя создать вручную")
        if parent_id is None and name == self.trash_namespace_name:
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
        existing = await self.namespace_repository.get_by_name_and_user(
            name=self.trash_namespace_name,
            user_id=user_id,
        )
        if existing and existing.parent_id is None:
            if existing.kind != self.trash_namespace_kind:
                existing.kind = self.trash_namespace_kind
                updated = await self.namespace_repository.update(existing)
                await self.db.commit()
                return updated.id
            return existing.id

        namespace = await self.namespace_repository.create(
            name=self.trash_namespace_name,
            user_id=user_id,
            parent_id=None,
            kind=self.trash_namespace_kind,
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
        existing = await self.namespace_repository.get_by_name_and_user(
            name=self.inbox_namespace_name,
            user_id=user_id,
        )
        if existing and existing.kind == self.inbox_namespace_kind and existing.parent_id is None:
            return existing.id
        namespace = await self.namespace_repository.create(
            name=self.inbox_namespace_name,
            user_id=user_id,
            parent_id=None,
            kind=self.inbox_namespace_kind,
            description=None,
        )
        await self.db.commit()
        return namespace.id

    async def get_or_create_vault_root_namespace(self, user_id: int, vault_name: str) -> int:
        existing = await self.namespace_repository.get_by_name_and_parent(
            user_id=user_id,
            parent_id=None,
            name=vault_name,
        )
        if existing:
            if existing.kind != self.vault_root_namespace_kind:
                existing.kind = self.vault_root_namespace_kind
                updated = await self.namespace_repository.update(existing)
                await self.db.commit()
                return updated.id
            return existing.id

        namespace = await self.namespace_repository.create(
            name=vault_name,
            user_id=user_id,
            parent_id=None,
            kind=self.vault_root_namespace_kind,
            description=None,
        )
        await self.db.commit()
        return namespace.id

    async def get_or_create_namespace_path(
        self,
        user_id: int,
        vault_name: str,
        parts: list[str],
    ) -> int:
        current_id = await self.get_or_create_vault_root_namespace(user_id=user_id, vault_name=vault_name)
        for part in parts:
            existing = await self.namespace_repository.get_by_name_and_parent(
                user_id=user_id,
                parent_id=current_id,
                name=part,
            )
            if existing:
                current_id = existing.id
                continue
            created = await self.namespace_repository.create(
                name=part,
                user_id=user_id,
                parent_id=current_id,
                kind=self.regular_namespace_kind,
                description=None,
            )
            current_id = created.id
        await self.db.commit()
        return current_id

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
        if namespace.kind == self.trash_namespace_kind:
            raise ValidationError("Системное пространство Trash нельзя изменять")
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
        if description is not None:
            namespace.description = description
        entity = await self.namespace_repository.update(namespace)
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
        if namespace.kind == self.inbox_namespace_kind:
            raise ValidationError("Пространство «Inbox» нельзя удалить")
        if namespace.kind == self.trash_namespace_kind:
            raise ValidationError("Пространство «Trash» нельзя удалить")

        if self.user_file_repository:
            trash_namespace_id = await self.get_or_create_trash(user_id)
            file_ids = await self.user_file_repository.list_ids_by_user_and_namespace(
                user_id=user_id,
                namespace_id=namespace_id,
            )
            for file_id in file_ids:
                updated = await self.user_file_repository.update_namespace(
                    file_id,
                    trash_namespace_id,
                )
                if updated and self.sync_notifier:
                    await self.sync_notifier.add_trash_command_to_queue(
                        user_file_id=updated.id,
                        user_id=user_id,
                    )

        await self.namespace_repository.delete(namespace_id)
        await self.db.commit()
        return namespace
