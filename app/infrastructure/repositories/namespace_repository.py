from typing import List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload
from app.infrastructure.db.models import Namespace, File


class NamespaceRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, namespace_id: int) -> Optional[Namespace]:
        result = await self.db.execute(
            select(Namespace).where(Namespace.id == namespace_id)
        )
        return result.scalar_one_or_none()


    async def get_files_by_namespace_id(self, namespace_id: int) -> List[File]:
        result = await self.db.execute(
            select(File)
            .where(File.namespace_id == namespace_id)
            .order_by(File.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_by_name_and_user(self, name: str, user_id: int) -> Optional[Namespace]:
        result = await self.db.execute(
            select(Namespace).where(
                Namespace.name == name,
                Namespace.user_id == user_id
            )
        )
        return result.scalar_one_or_none()

    async def get_by_user(
        self, user_id: int, skip: int = 0, limit: int = 100
    ) -> Tuple[List[Tuple[Namespace, int]], int]:
        files_count_subq = (
            select(func.count(File.id))
            .where(File.namespace_id == Namespace.id)
            .correlate(Namespace)
            .scalar_subquery()
        )
        result = await self.db.execute(
            select(Namespace, files_count_subq.label("files_count"))
            .where(Namespace.user_id == user_id)
            .order_by(Namespace.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        rows = result.all()
        namespaces_with_count = [(row[0], row[1] or 0) for row in rows]
        count_result = await self.db.execute(
            select(func.count(Namespace.id)).where(Namespace.user_id == user_id)
        )
        total = count_result.scalar()
        return namespaces_with_count, total or 0

    async def get_by_user_with_files(self, user_id: int) -> List[Namespace]:
        """Namespace'ы пользователя с файлами (selectinload + принудительная материализация в рамках сессии)."""
        result = await self.db.execute(
            select(Namespace)
            .where(Namespace.user_id == user_id)
            .order_by(Namespace.created_at.desc())
            .options(selectinload(Namespace.files))
        )
        namespaces = list(result.scalars().unique().all())
        return namespaces

    async def create(
        self, name: str, user_id: int, description: Optional[str] = None
    ) -> Namespace:
        namespace = Namespace(
            user_id=user_id,
            name=name,
            description=description,
        )
        self.db.add(namespace)
        return namespace

    async def update(self, namespace: Namespace) -> Namespace:
        self.db.add(namespace)
        return namespace

    async def delete(self, namespace: Namespace) -> None:
        await self.db.delete(namespace)
