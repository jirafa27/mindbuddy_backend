from typing import Optional
import secrets

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.protocols import UserRepository, NamespaceRepository
from app.core.exceptions import ValidationError, NotFoundError, UnauthorizedError
from app.core.namespace_constants import (
    INBOX_NAMESPACE_KIND,
    INBOX_NAMESPACE_NAME,
    TRASH_NAMESPACE_KIND,
    TRASH_NAMESPACE_NAME,
    VAULT_ROOT_NAMESPACE_KIND,
    VAULT_ROOT_NAMESPACE_NAME,
)
from app.schemas.user import UserResponse
from app.core.security import hash_password, verify_password


def _entity_to_response(user) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        is_active=user.is_active,
        watcher_token=user.watcher_token,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


class UserService:
    """Сервис для работы с пользователями"""

    def __init__(
        self,
        repository: UserRepository,
        db: AsyncSession,
        namespace_repository: Optional[NamespaceRepository] = None,
    ):
        self.repository = repository
        self.db = db
        self.namespace_repository = namespace_repository

    async def create_user(
        self,
        email: str,
        password: str,
        full_name: Optional[str] = None,
    ) -> UserResponse:
        """Регистрация пользователя по email и паролю. Возвращает UserResponse."""
        existing = await self.repository.get_by_email(email)
        if existing:
            raise ValidationError(f"Пользователь с email {email} уже существует")
        watcher_token = secrets.token_urlsafe(32)
        user = await self.repository.create(
            email=email.strip().lower(),
            password_hash=hash_password(password),
            full_name=full_name,
            watcher_token=watcher_token,
        )
        await self.db.commit()
        if self.namespace_repository:
            await self._get_or_create_vault_root_for_user(user.id)
            await self._get_or_create_inbox_for_user(user.id)
            await self._get_or_create_trash_for_user(user.id)
        return _entity_to_response(user)

    async def get_user(
        self,
        user_id: int,
    ) -> Optional[UserResponse]:
        """Пользователь по внутреннему ID или None."""
        user = await self.repository.get_by_id(user_id)
        if not user:
            return None
        return _entity_to_response(user)

    async def login(self, email: str, password: str) -> UserResponse:
        """Проверка пароля и возврат пользователя. NotFoundError или ValidationError при ошибке."""
        user = await self.repository.get_by_email(email.strip().lower())
        if not user:
            raise NotFoundError("Неверный email или пароль")
        from sqlalchemy import select
        from app.infrastructure.db.models import User
        result = await self.db.execute(select(User).where(User.id == user.id))
        model = result.scalar_one_or_none()
        if not model or not model.password_hash or not verify_password(password, model.password_hash):
            raise ValidationError("Неверный email или пароль")
        return _entity_to_response(user)

    async def get_user_by_watcher_token(
        self,
        token: str,
    ) -> UserResponse:
        """Пользователь по токену Desktop Watcher. UnauthorizedError если не найден."""
        user = await self.repository.get_by_watcher_token(token)
        if not user:
            raise UnauthorizedError("Недействительный токен Desktop Watcher")
        return _entity_to_response(user)

    async def _get_or_create_vault_root_for_user(self, user_id: int) -> int:
        """Получает или создаёт пространство Vault для пользователя."""
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

    async def _get_or_create_inbox_for_user(self, user_id: int) -> None:
        """Создаёт пространство Inbox для пользователя, если его ещё нет."""
        vault_root_id = await self._get_or_create_vault_root_for_user(user_id)
        existing = await self.namespace_repository.get_by_name_and_parent(
            user_id=user_id,
            parent_id=vault_root_id,
            name=INBOX_NAMESPACE_NAME,
        )
        if existing:
            if existing.kind == INBOX_NAMESPACE_KIND:
                return
            existing.kind = INBOX_NAMESPACE_KIND
            await self.namespace_repository.update(existing)
            await self.db.commit()
            return
        await self.namespace_repository.create(
            name=INBOX_NAMESPACE_NAME,
            user_id=user_id,
            parent_id=vault_root_id,
            kind=INBOX_NAMESPACE_KIND,
            description=None,
        )
        await self.db.commit()

    async def _get_or_create_trash_for_user(self, user_id: int) -> None:
        """Создаёт пространство Trash для пользователя, если его ещё нет."""
        vault_root_id = await self._get_or_create_vault_root_for_user(user_id)
        existing = await self.namespace_repository.get_by_name_and_parent(
            user_id=user_id,
            parent_id=vault_root_id,
            name=TRASH_NAMESPACE_NAME,
        )
        if existing:
            if existing.kind == TRASH_NAMESPACE_KIND:
                return
            existing.kind = TRASH_NAMESPACE_KIND
            await self.namespace_repository.update(existing)
            await self.db.commit()
            return
        await self.namespace_repository.create(
            name=TRASH_NAMESPACE_NAME,
            user_id=user_id,
            parent_id=vault_root_id,
            kind=TRASH_NAMESPACE_KIND,
            description=None,
        )
        await self.db.commit()
