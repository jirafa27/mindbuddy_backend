from fastapi import APIRouter, Depends, status

from app.schemas import UserCreate, UserResponse, ResponseMessage
from app.schemas.user import TokenResponse
from app.services.user_service import UserService
from app.core.dependencies import get_user_service, get_current_user
from app.core.security import create_access_token
from app.core.exceptions import NotFoundError

router = APIRouter()


@router.post(
    "/",
    response_model=ResponseMessage[TokenResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Регистрация пользователя",
)
async def register(
    user: UserCreate,
    service: UserService = Depends(get_user_service),
):
    """
    Регистрация по email и паролю.
    Возвращает пользователя и access_token для последующих запросов.
    """
    created = await service.create_user(
        email=user.email,
        password=user.password,
        full_name=user.full_name,
    )
    token = create_access_token(subject=created.id)
    return ResponseMessage(data=TokenResponse(access_token=token, user=created))


@router.get(
    "/me",
    response_model=ResponseMessage[UserResponse],
    summary="Текущий пользователь",
)
async def get_me(
    user: UserResponse = Depends(get_current_user),
):
    """Данные текущего пользователя (по JWT)."""
    return ResponseMessage(data=user)


@router.get(
    "/{user_id}",
    response_model=ResponseMessage[UserResponse],
    summary="Получить пользователя по ID",
)
async def get_user(
    user_id: int,
    user_service: UserService = Depends(get_user_service),
):
    """Получить пользователя по внутреннему ID."""
    user = await user_service.get_user(user_id=user_id)
    if not user:
        raise NotFoundError(f"Пользователь с ID {user_id} не найден")
    return ResponseMessage(data=user)
