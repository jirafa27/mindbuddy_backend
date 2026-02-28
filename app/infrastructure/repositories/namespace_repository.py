from typing import List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select, delete
from sqlalchemy.orm import selectinload
from app.infrastructure.db.models import Namespace, UserFile
from app.domain.entities import NamespaceEntity, NamespaceFileItem, UserFileEntity


class PgNamespaceRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    def _to_entity(
        self,
        model: Namespace,
        user_files: Optional[List[UserFileEntity]] = None,
    ) -> NamespaceEntity:
        if user_files is None:
            user_files = [
                UserFileEntity(
                    id=uf.id,
                    user_id=uf.user_id,
                    file_id=uf.file_id,
                    namespace_id=uf.namespace_id,
                    custom_title=uf.custom_title,
                )
                for uf in model.user_files
            ]
        return NamespaceEntity(
            id=model.id,
            user_id=model.user_id,
            name=model.name,
            description=model.description,
            created_at=model.created_at,
            user_files=user_files,
        )

    def _user_file_to_entity(self, model: UserFile) -> UserFileEntity:
        return UserFileEntity(
            id=model.id,
            user_id=model.user_id,
            file_id=model.file_id,
            namespace_id=model.namespace_id,
            custom_title=model.custom_title,
        )

    async def get_by_id(self, namespace_id: int) -> Optional[NamespaceEntity]:
        result = await self.db.execute(
            select(Namespace)
            .where(Namespace.id == namespace_id)
            .options(selectinload(Namespace.user_files))
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_entity(row)

    async def _get_model_by_id(self, namespace_id: int) -> Optional[Namespace]:
        """Модель Namespace по id (внутренний метод для update)."""
        result = await self.db.execute(
            select(Namespace).where(Namespace.id == namespace_id)
        )
        return result.scalar_one_or_none()

    async def get_files_by_namespace_id(self, namespace_id: int) -> List[NamespaceFileItem]:
        result = await self.db.execute(
            select(UserFile)
            .where(UserFile.namespace_id == namespace_id)
            .order_by(UserFile.created_at.desc())
            .options(selectinload(UserFile.file))
        )
        rows = result.scalars().all()
        out = []
        for uf in rows:
            cf = uf.file
            meta = (cf.media_metadata or {}) if cf else {}
            out.append(
                NamespaceFileItem(
                    id=uf.id,
                    file_path=cf.file_path if cf else None,
                    filename=(uf.custom_title or meta.get("title") or "document") or "document",
                    file_type=meta.get("file_type") or "md",
                    file_size=0,
                    created_at=uf.created_at,
                )
            )
        return out

    async def get_by_name_and_user(self, name: str, user_id: int) -> Optional[NamespaceEntity]:
        result = await self.db.execute(
            select(Namespace)
            .where(
                Namespace.name == name,
                Namespace.user_id == user_id
            )
            .options(selectinload(Namespace.user_files))
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_entity(row)

    async def get_by_user(
        self, user_id: int, skip: int = 0, limit: int = 100
    ) -> Tuple[List[Tuple[NamespaceEntity, int]], int]:
        files_count_subq = (
            select(func.count(UserFile.id))
            .where(UserFile.namespace_id == Namespace.id)
            .correlate(Namespace)
            .scalar_subquery()
        )
        result = await self.db.execute(
            select(Namespace, files_count_subq.label("files_count"))
            .where(Namespace.user_id == user_id)
            .order_by(Namespace.created_at.desc())
            .offset(skip)
            .limit(limit)
            .options(selectinload(Namespace.user_files))
        )
        rows = result.all()
        namespaces_with_count = [(self._to_entity(row[0]), row[1] or 0) for row in rows]
        count_result = await self.db.execute(
            select(func.count(Namespace.id)).where(Namespace.user_id == user_id)
        )
        total = count_result.scalar()
        return namespaces_with_count, total or 0

    async def get_by_user_with_files(self, user_id: int) -> List[NamespaceEntity]:
        result = await self.db.execute(
            select(Namespace)
            .where(Namespace.user_id == user_id)
            .order_by(Namespace.created_at.desc())
            .options(selectinload(Namespace.user_files))
        )
        namespaces = list(result.scalars().unique().all())
        return [self._to_entity(namespace) for namespace in namespaces]

    async def create(
        self, name: str, user_id: int, description: Optional[str] = None
    ) -> NamespaceEntity:
        namespace = Namespace(
            user_id=user_id,
            name=name,
            description=description,
        )
        self.db.add(namespace)
        await self.db.flush()
        return self._to_entity(namespace, user_files=[])

    async def update(self, namespace: NamespaceEntity) -> NamespaceEntity:
        model = await self._get_model_by_id(namespace.id)
        if model is None:
            raise ValueError(f"Namespace id={namespace.id} not found")
        model.name = namespace.name
        model.description = namespace.description
        self.db.add(model)
        await self.db.flush()
        return self._to_entity(model, user_files=[])

    async def delete(self, id: int) -> None:
        await self.db.execute(delete(Namespace).where(Namespace.id == id))
        await self.db.flush()