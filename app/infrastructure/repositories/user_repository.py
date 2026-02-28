from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.domain.entities import UserEntity
from app.infrastructure.db.models import User


class PgUserRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    def _to_entity(self, model: User) -> UserEntity:
        return UserEntity(
            id=model.id,
            email=model.email or "",
            username=model.username,
            full_name=model.full_name,
            is_active=model.is_active,
            watcher_token=model.watcher_token,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    async def get_by_id(self, user_id: int) -> Optional[UserEntity]:
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_entity(row)

    async def get_by_email(self, email: str) -> Optional[UserEntity]:
        result = await self.db.execute(
            select(User).where(User.email == email)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_entity(row)

    async def get_by_watcher_token(self, token: str) -> Optional[UserEntity]:
        result = await self.db.execute(
            select(User).where(User.watcher_token == token)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_entity(row)

    async def create(
        self,
        email: str,
        password_hash: str,
        full_name: Optional[str] = None,
        watcher_token: Optional[str] = None,
    ) -> UserEntity:
        user = User(
            email=email,
            password_hash=password_hash,
            full_name=full_name,
            watcher_token=watcher_token,
        )
        self.db.add(user)
        await self.db.flush()
        return self._to_entity(user)
