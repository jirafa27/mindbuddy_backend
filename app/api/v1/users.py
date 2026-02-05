from fastapi import APIRouter, Depends, status

from app.schemas import (
    UserCreate,
    UserResponse,
    ResponseMessage,
)
from app.services.user_service import UserService
from app.core.dependencies import get_user_service
from app.core.exceptions import NotFoundError

router = APIRouter()


@router.post(
    "/",
    response_model=ResponseMessage[UserResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Создать пользователя из Telegram",
)
async def create_user(
    user: UserCreate,
    service: UserService = Depends(get_user_service),
):
    """
    Создает нового пользователя из Telegram.
    
    Args:
        user: Данные из Telegram (telegram_id обязателен)
        
    Returns:
        ID созданного пользователя
        
    Raises:
        ValidationError: Если пользователь с таким telegram_id уже существует
    """
    created = await service.create_user(
        telegram_id=user.telegram_id,
        username=user.username,
        full_name=user.full_name,
    )
    return ResponseMessage(data=created)

@router.get(
    "/telegram/{telegram_id}",
    response_model=ResponseMessage[UserResponse],
    summary="Получить пользователя по Telegram ID",
)
async def get_user_by_telegram_id(
    telegram_id: int,
    user_service: UserService = Depends(get_user_service),
):
    """
    Получает информацию о пользователе по Telegram ID.
    
    **Используется для Telegram ботов:**
    Удобный способ получить пользователя по его Telegram ID.
    
    Args:
        telegram_id: Telegram ID пользователя
        
    Returns:
        Информация о пользователе
        
    Raises:
        NotFoundError: Если пользователь не найден
    """
    user = await user_service.get_user_by_telegram_id(telegram_id=telegram_id)
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
    """
    Получает информацию о пользователе по внутреннему ID.
    
    Args:
        user_id: Внутренний ID пользователя
        
    Returns:
        Информация о пользователе
        
    Raises:
        NotFoundError: Если пользователь не найден
    """
    user = await user_service.get_user(user_id=user_id)
    if not user:
        raise NotFoundError(f"Пользователь с ID {user_id} не найден")
    return ResponseMessage(data=user)
