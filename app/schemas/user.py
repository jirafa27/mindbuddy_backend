from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class UserCreate(BaseModel):
    """Схема регистрации пользователя"""
    email: str = Field(..., min_length=1, max_length=255, description="Email")
    password: str = Field(..., min_length=6, max_length=255, description="Пароль")
    full_name: Optional[str] = Field(None, max_length=255, description="Полное имя")


class UserResponse(BaseModel):
    """Схема ответа с информацией о пользователе"""
    id: int
    email: str
    username: Optional[str] = None
    full_name: Optional[str] = None
    is_active: bool
    watcher_token: Optional[str] = Field(None, description="Токен для аутентификации Desktop Watcher")
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    """Схема входа"""
    email: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=1, max_length=255)


class TokenResponse(BaseModel):
    """Ответ с access token и данными пользователя"""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class AccessTokenResponse(BaseModel):
    """Краткий OAuth2-ответ для Swagger."""
    access_token: str
    token_type: str = "bearer"

