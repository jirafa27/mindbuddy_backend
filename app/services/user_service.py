from typing import Optional
import secrets

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.protocols import UserRepository
from app.infrastructure.db.models import User
from app.core.exceptions import ValidationError, NotFoundError
from app.schemas.user import UserResponse


class UserService:
    """Сервис для работы с пользователями"""

    def __init__(self, repository: UserRepository, db: AsyncSession):
        self.repository = repository
        self.db = db

    async def create_user(
        self,
        telegram_id: int,
        username: Optional[str],
        full_name: Optional[str],
    ) -> UserResponse:
        """Создает нового пользователя из Telegram. Возвращает UserResponse."""
        existing = await self.repository.get_by_telegram_id(telegram_id)
        if existing:
            raise ValidationError(
                f"Пользователь с Telegram ID {telegram_id} уже существует"
            )
        watcher_token = self._generate_watcher_token()
        user = await self.repository.create(
            telegram_id=telegram_id,
            username=username,
            full_name=full_name,
        )
        user.watcher_token = watcher_token
        await self.db.commit()
        await self.db.refresh(user)
        return UserResponse(
                            id=user.id,
                            telegram_id=user.telegram_id,
                            username=user.username,
                            full_name=user.full_name,
                            is_active=user.is_active,
                            watcher_token=user.watcher_token,
                            created_at=user.created_at,
                            updated_at=user.updated_at,
                        )


    async def get_or_create_user(
        self,
        telegram_id: int,
        username: Optional[str],
        full_name: Optional[str],
    ) -> tuple[UserResponse, bool]:
        """Находит пользователя по telegram_id или создает нового. Возвращает (UserResponse, created)."""
        existing = await self.repository.get_by_telegram_id(telegram_id)
        if existing:
            return UserResponse(id=existing.id,
                                telegram_id=existing.telegram_id,
                                username=existing.username,
                                full_name=existing.full_name,
                                is_active=existing.is_active,
                                watcher_token=existing.watcher_token,
                                created_at=existing.created_at,
                                updated_at=existing.updated_at,
                                ), False
        watcher_token = self._generate_watcher_token()
        user = await self.repository.create(
            telegram_id=telegram_id,
            username=username,
            full_name=full_name,
        )
        user.watcher_token = watcher_token
        await self.db.commit()
        await self.db.refresh(user)
        return UserResponse(
            id=user.id,
            telegram_id=user.telegram_id,
            username=user.username,
            full_name=user.full_name,
            is_active=user.is_active,
            watcher_token=user.watcher_token,
            created_at=user.created_at,
            updated_at=user.updated_at,
        ), True

    async def get_user(
        self,
        user_id: int,
    ) -> Optional[UserResponse]:
        """Пользователь по внутреннему ID или None."""
        user = await self.repository.get_by_id(user_id)
        if not user:
            return None
        return UserResponse(
            id=user.id,
            telegram_id=user.telegram_id,
            username=user.username,
            full_name=user.full_name,
            is_active=user.is_active,
            watcher_token=user.watcher_token,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )

    async def get_user_by_telegram_id(
        self,
        telegram_id: int,
    ) -> UserResponse:
        """Пользователь по Telegram ID. NotFoundError если не найден."""
        user = await self.repository.get_by_telegram_id(telegram_id)
        if not user:
            raise NotFoundError(
                f"Пользователь с Telegram ID {telegram_id} не найден"
            )
        return UserResponse(
            id=user.id,
            telegram_id=user.telegram_id,
            username=user.username,
            full_name=user.full_name,
            is_active=user.is_active,
            watcher_token=user.watcher_token,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )

    async def get_user_by_watcher_token(
        self,
        token: str,
    ) -> UserResponse:
        """Пользователь по токену Watcher. NotFoundError если не найден."""
        user = await self.repository.get_by_watcher_token(token)
        if not user:
            raise NotFoundError("Недействительный токен Watcher")
        return UserResponse(
            id=user.id,
            telegram_id=user.telegram_id,
            username=user.username,
            full_name=user.full_name,
            is_active=user.is_active,
            watcher_token=user.watcher_token,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )

    async def regenerate_watcher_token(
        self,
        user_id: int,
    ) -> str:
        """
        Перегенерирует токен Watcher для пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Новый токен
            
        Raises:
            NotFoundError: Если пользователь не найден
        """
        user = await self.repository.get_by_id(user_id)
        
        if not user:
            raise NotFoundError(f"Пользователь с ID {user_id} не найден")
        
        new_token = self._generate_watcher_token()
        user.watcher_token = new_token
        
        await self.db.commit()
        await self.db.refresh(user)
        
        return new_token

    def _generate_watcher_token(self) -> str:
        """Генерирует уникальный токен для Watcher"""
        return secrets.token_urlsafe(32)
