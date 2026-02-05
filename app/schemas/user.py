from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class UserCreate(BaseModel):
    """Схема создания пользователя"""
    telegram_id: int = Field(..., description="Telegram ID пользователя (обязательно)")
    username: Optional[str] = Field(None, min_length=1, max_length=255, description="Telegram username (@username)")
    full_name: Optional[str] = Field(None, max_length=255, description="Полное имя пользователя")


class UserResponse(BaseModel):
    """Схема ответа с информацией о пользователе"""
    id: int
    telegram_id: Optional[int]
    username: Optional[str]
    full_name: Optional[str]
    is_active: bool
    watcher_token: Optional[str] = Field(None, description="Токен для аутентификации Desktop Watcher")
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

