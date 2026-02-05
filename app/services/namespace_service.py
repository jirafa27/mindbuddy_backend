from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.protocols import NamespaceRepository
from app.infrastructure.db.models import Namespace
from app.core.exceptions import NotFoundError, ValidationError, ForbiddenError
from app.schemas.namespace import NamespaceResponse, NamespaceListItem
from app.schemas.file import FileInNamespace, FileStructureItem, NamespaceStructureItem


class NamespaceService:
    """Сервис для работы с namespace (бизнес-логика)"""

    def __init__(self, repository: NamespaceRepository, db: AsyncSession):
        self.repository = repository
        self.db = db

    async def create_namespace(self, name: str, user_id: int, description: Optional[str]) -> NamespaceResponse:
        """
        Создает новый namespace для пользователя.

        Returns:
            NamespaceResponse (без files).
        """
        existing = await self.repository.get_by_name_and_user(name=name, user_id=user_id)
        if existing:
            raise ValidationError(
                f"Namespace с именем '{name}' уже существует для данного пользователя"
            )
        namespace = await self.repository.create(
            name=name,
            user_id=user_id,
            description=description,
        )
        await self.db.commit()
        await self.db.refresh(namespace)
        return NamespaceResponse(
            id=namespace.id,
            user_id=namespace.user_id,
            name=namespace.name,
            description=namespace.description,
            created_at=namespace.created_at,
            files=[],
        )

    async def get_namespace(
        self,
        namespace_id: int,
        user_id: int,
    ) -> NamespaceResponse:
        """
        Получает namespace по ID с проверкой доступа.

        Returns:
            NamespaceResponse (без files; файлы — через get_namespace_files).
        """
        namespace = await self.repository.get_by_id(namespace_id)
        if not namespace:
            raise NotFoundError(f"Namespace с ID {namespace_id} не найден")
        if namespace.user_id != user_id:
            raise ForbiddenError("Нет доступа к данному namespace")
        return NamespaceResponse(
            id=namespace.id,
            user_id=namespace.user_id,
            name=namespace.name,
            description=namespace.description,
            created_at=namespace.created_at,
            files=[],
        )

    async def get_namespace_files(
        self,
        namespace_id: int,
    ) -> list[FileInNamespace]:
        """Список файлов namespace (file_path для генерации URL в API)."""
        files = await self.repository.get_files_by_namespace_id(namespace_id)
        return [
            FileInNamespace(
                id=f.id,
                filename=f.filename,
                file_type=f.file_type,
                file_size=f.file_size,
                created_at=f.created_at,
                file_path=f.file_path or "",
            )
            for f in files
        ]

    async def get_user_namespaces_with_files(self, user_id: int) -> list[NamespaceStructureItem]:
        """Список namespace пользователя с файлами (для структуры Watcher)."""
        namespaces = await self.repository.get_by_user_with_files(user_id)
        return [
            NamespaceStructureItem(
                id=ns.id,
                name=ns.name,
                files=[
                    FileStructureItem(
                        id=f.id,
                        filename=f.filename,
                        file_size=f.file_size,
                        updated_at=f.updated_at,
                    )
                    for f in ns.files
                ],
            )
            for ns in namespaces
        ]

    async def get_user_namespaces(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[NamespaceListItem], int]:
        """Список namespace пользователя с пагинацией."""
        namespaces_with_count, total = await self.repository.get_by_user(
            user_id=user_id, skip=skip, limit=limit
        )
        items = [
            NamespaceListItem(
                id=ns.id,
                user_id=ns.user_id,
                name=ns.name,
                description=ns.description,
                created_at=ns.created_at,
                files_count=count,
            )
            for ns, count in namespaces_with_count
        ]
        return items, total

    async def update_namespace(
        self,
        namespace_id: int,
        user_id: int,
        name: Optional[str],
        description: Optional[str],
    ) -> NamespaceResponse:
        """Обновляет namespace. Возвращает NamespaceResponse (без files)."""
        namespace = await self.repository.get_by_id(namespace_id)
        if not namespace:
            raise NotFoundError(f"Namespace с ID {namespace_id} не найден")
        if namespace.user_id != user_id:
            raise ForbiddenError("Нет доступа к данному namespace")
        if name and name != namespace.name:
            existing = await self.repository.get_by_name_and_user(name=name, user_id=user_id)
            if existing and existing.id != namespace_id:
                raise ValidationError(
                    f"Namespace с именем '{name}' уже существует для данного пользователя"
                )
            namespace.name = name
        if description is not None:
            namespace.description = description
        await self.repository.update(namespace)
        await self.db.commit()
        await self.db.refresh(namespace)
        return NamespaceResponse(
            id=namespace.id,
            user_id=namespace.user_id,
            name=namespace.name,
            description=namespace.description,
            created_at=namespace.created_at,
            files=[],
        )

    async def delete_namespace(
        self,
        namespace_id: int,
        user_id: int,
    ) -> list[str]:
        """Удаляет namespace. Возвращает список file_path для удаления из MinIO (в API)."""
        await self.get_namespace(namespace_id, user_id)
        files = await self.repository.get_files_by_namespace_id(namespace_id)
        file_paths = [f.file_path for f in files if f.file_path]
        namespace = await self.repository.get_by_id(namespace_id)
        if namespace:
            await self.repository.delete(namespace)
        await self.db.commit()
        return file_paths
