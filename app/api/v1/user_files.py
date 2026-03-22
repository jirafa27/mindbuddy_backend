"""Эндпоинты для управления пользовательскими файлами (user_files)."""
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.base import ResponseMessage
from app.schemas.content import AttachFileResponse, UserFileCreateRequest
from app.services.file_service import FileService
from app.services.namespace_service import NamespaceService
from app.core.dependencies import get_current_user, get_file_service, get_namespace_service
from app.core.exceptions import NotFoundError, ForbiddenError, ValidationError
from app.schemas.user import UserResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "",
    response_model=ResponseMessage[AttachFileResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Привязать файл к пространству",
)
async def create_user_file(
    body: UserFileCreateRequest,
    user: UserResponse = Depends(get_current_user),
    file_service: FileService = Depends(get_file_service),
    namespace_service: NamespaceService = Depends(get_namespace_service),
) -> ResponseMessage[AttachFileResponse]:
    """
    Привязывает уже загруженный контент-файл к пространству пользователя.

    `file_id` — ID записи в таблице `files` (content_file_id), возвращённый из
    `POST /content/extract` или `POST /files/upload`.

    Если `namespace_id` не передан (null) — файл помещается в Inbox пользователя.
    Если файл уже привязан к другому пространству — namespace_id обновляется.
    """
    namespace_id = body.namespace_id
    if namespace_id is None:
        namespace_id = await namespace_service.get_or_create_inbox(user.id)

    try:
        user_file = await file_service.attach_file_to_namespace(
            content_file_id=body.file_id,
            user_id=user.id,
            namespace_id=namespace_id,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ForbiddenError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception(
            "[API/UserFiles] Failed to attach file_id=%d for user=%d",
            body.file_id,
            user.id,
        )
        raise HTTPException(status_code=500, detail="Ошибка привязки файла к пространству")

    return ResponseMessage[AttachFileResponse](
        data=AttachFileResponse(
            user_file_id=user_file.id,
            content_file_id=body.file_id,
            namespace_id=namespace_id,
            filename=user_file.custom_title,
        )
    )
