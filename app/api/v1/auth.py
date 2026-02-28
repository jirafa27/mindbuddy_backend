"""Эндпоинты аутентификации: логин."""
from fastapi import APIRouter, Depends

from app.schemas.user import LoginRequest, TokenResponse
from app.schemas.base import ResponseMessage
from app.services.user_service import UserService
from app.core.dependencies import get_user_service
from app.core.security import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=ResponseMessage[TokenResponse])
async def login(
    body: LoginRequest,
    user_service: UserService = Depends(get_user_service),
) -> ResponseMessage[TokenResponse]:
    """Вход по email и паролю. Возвращает access_token и данные пользователя."""
    user = await user_service.login(email=body.email, password=body.password)
    token = create_access_token(subject=user.id)
    return ResponseMessage(
        data=TokenResponse(access_token=token, user=user),
    )
