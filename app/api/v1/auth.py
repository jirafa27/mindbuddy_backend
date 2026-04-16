"""Эндпоинты аутентификации: логин."""
from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm

from app.schemas.user import AccessTokenResponse, LoginRequest, TokenResponse
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


@router.post("/token", response_model=AccessTokenResponse)
async def login_for_swagger(
    form_data: OAuth2PasswordRequestForm = Depends(),
    user_service: UserService = Depends(get_user_service),
) -> AccessTokenResponse:
    """OAuth2 password flow для Swagger. В поле username нужно передавать email."""
    user = await user_service.login(
        email=form_data.username,
        password=form_data.password,
    )
    token = create_access_token(subject=user.id)
    return AccessTokenResponse(access_token=token)
