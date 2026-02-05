from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=False)
class UserEntity:
    """Пользователь системы"""
    id: int
    telegram_id: Optional[int] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    is_active: bool = True
    watcher_token: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
