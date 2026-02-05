from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.infrastructure.db.models import User


class UserRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, user_id: int) -> Optional[User]:
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_by_telegram_id(self, telegram_id: int) -> Optional[User]:
        result = await self.db.execute(
            select(User).where(User.telegram_id == telegram_id)
        )
        return result.scalar_one_or_none()

    async def get_by_watcher_token(self, token: str) -> Optional[User]:
        result = await self.db.execute(
            select(User).where(User.watcher_token == token)
        )
        return result.scalar_one_or_none()

    async def create(
        self,
        telegram_id: Optional[int] = None,
        username: Optional[str] = None,
        full_name: Optional[str] = None,
    ) -> User:
        user = User(
            telegram_id=telegram_id,
            username=username,
            full_name=full_name,
        )
        self.db.add(user)
        return user
