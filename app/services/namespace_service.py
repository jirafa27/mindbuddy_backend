from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.protocols import NamespaceRepository
from app.core.exceptions import NotFoundError, ValidationError, ForbiddenError
from app.domain.entities.namespace import NamespaceEntity, NamespaceFileItem


class NamespaceService:
    """Сервис для работы с namespace"""

    def __init__(self, namespace_repository: NamespaceRepository, db: AsyncSession):
        """
        Args:
            namespace_repository: Репозиторий для работы с namespace
            db: Сессия БД
        """
        self.inbox_namespace_name = "Inbox"
        self.namespace_repository = namespace_repository
        self.db = db

    async def create_namespace(self, name: str, user_id: int, description: Optional[str]) -> NamespaceEntity:
        """Создает новый namespace для пользователя.
        Args:
            name: Имя namespace
            user_id: ID пользователя
            description: Описание namespace

        Returns:
            NamespaceEntity.
        """

        existing = await self.namespace_repository.get_by_name_and_user(name=name, user_id=user_id)
        if existing:
            raise ValidationError(
                f"Namespace с именем '{name}' уже существует для данного пользователя"
            )
        namespace = await self.namespace_repository.create(
            name=name,
            user_id=user_id,
            description=description,
        )
        await self.db.commit()
        return namespace

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
        if existing:
            return existing.id
        namespace = await self.namespace_repository.create(
            name=self.inbox_namespace_name,
            user_id=user_id,
            description=None,
        )
        await self.db.commit()
        return namespace.id

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
        if name and name != namespace.name:
            existing = await self.namespace_repository.get_by_name_and_user(name=name, user_id=user_id)
            if existing and existing.id != namespace_id:
                raise ValidationError(
                    f"Namespace с именем '{name}' уже существует для данного пользователя"
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
        """Удаляет namespace. Пространство Inbox удалять нельзя."""
        namespace = await self.namespace_repository.get_by_id(namespace_id)
        if not namespace:
            raise NotFoundError(f"Namespace с ID {namespace_id} не найден")
        if namespace.user_id != user_id:
            raise ForbiddenError("Нет доступа к данному namespace")
        if namespace.name == self.inbox_namespace_name:
            raise ValidationError("Пространство «Inbox» нельзя удалить")

        await self.namespace_repository.delete(namespace_id)
        await self.db.commit()
        return namespace
