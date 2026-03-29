from typing import Optional, Sequence

from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.db.models import UserFile, File
from app.domain.entities import UserFileEntity


class PgUserFileRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    def _to_entity(self, model: UserFile) -> UserFileEntity:
        return UserFileEntity(
            id=model.id,
            user_id=model.user_id,
            file_id=model.file_id,
            namespace_id=model.namespace_id,
            custom_title=model.custom_title,
        )

    async def get_by_id(self, user_file_id: int) -> Optional[UserFileEntity]:
        result = await self.db.execute(select(UserFile).where(UserFile.id == user_file_id))
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_entity(row)

    async def list_ids_by_user_and_namespace(
        self, user_id: int, namespace_id: int
    ) -> Sequence[int]:
        """user_files.id пользователя в пространстве, по возрастанию created_at."""
        result = await self.db.execute(
            select(UserFile.id)
            .where(UserFile.user_id == user_id, UserFile.namespace_id == namespace_id)
            .order_by(UserFile.created_at.asc())
        )
        return [row[0] for row in result.fetchall()]

    async def find_by_user_and_file(
        self, user_id: int, file_id: int, namespace_id: Optional[int] = None
    ) -> Optional[UserFileEntity]:
        result = await self.db.execute(
            select(UserFile).where(
                UserFile.user_id == user_id,
                UserFile.file_id == file_id,
                UserFile.namespace_id == namespace_id,
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_entity(row)

    async def find_by_source_url(self, source_url: str, user_id: int) -> Optional[UserFileEntity]:
        """Найти user_file по source_url контент-файла и user_id."""
        result = await self.db.execute(
            select(UserFile)
            .join(File, UserFile.file_id == File.id)
            .where(File.source_url == source_url, UserFile.user_id == user_id)
            .limit(1)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_entity(row)

    async def find_by_content_hash(self, content_hash: str, user_id: int) -> Optional[UserFileEntity]:
        """Найти user_file по content_hash контент-файла и user_id."""
        result = await self.db.execute(
            select(UserFile)
            .join(File, UserFile.file_id == File.id)
            .where(File.content_hash == content_hash, UserFile.user_id == user_id)
            .limit(1)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_entity(row)


    async def create(
        self,
        user_id: int,
        file_id: int,
        namespace_id: Optional[int] = None,
        custom_title: Optional[str] = None,
    ) -> UserFileEntity:
        row = UserFile(
            user_id=user_id,
            file_id=file_id,
            namespace_id=namespace_id,
            custom_title=custom_title,
        )
        self.db.add(row)
        await self.db.flush()
        return self._to_entity(row)

    async def delete(self, user_file_id: int) -> None:
        await self.db.execute(delete(UserFile).where(UserFile.id == user_file_id))
        await self.db.flush()

    async def update_namespace(self, user_file_id: int, namespace_id: Optional[int]) -> Optional[UserFileEntity]:
        result = await self.db.execute(select(UserFile).where(UserFile.id == user_file_id))
        row = result.scalar_one_or_none()
        if row is None:
            return None
        row.namespace_id = namespace_id
        await self.db.flush()
        return self._to_entity(row)

    async def count_by_file_id(self, file_id: int) -> int:
        """Количество UserFile, ссылающихся на данный File (content file)."""
        result = await self.db.execute(
            select(func.count()).select_from(UserFile).where(UserFile.file_id == file_id)
        )
        return result.scalar_one()

    async def update_file_id(self, user_file_id: int, new_file_id: int) -> Optional[UserFileEntity]:
        """Переключает UserFile на другой File (для Copy-on-Write при редактировании)."""
        result = await self.db.execute(select(UserFile).where(UserFile.id == user_file_id))
        row = result.scalar_one_or_none()
        if row is None:
            return None
        row.file_id = new_file_id
        await self.db.flush()
        return self._to_entity(row)

    async def update_custom_title(self, user_file_id: int, new_title: str) -> Optional[UserFileEntity]:
        """Обновляет отображаемое имя файла (custom_title)."""
        result = await self.db.execute(select(UserFile).where(UserFile.id == user_file_id))
        row = result.scalar_one_or_none()
        if row is None:
            return None
        row.custom_title = new_title
        await self.db.flush()
        return self._to_entity(row)
