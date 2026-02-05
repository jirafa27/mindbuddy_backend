from fastapi import APIRouter, Depends, status, Query
import logging

from app.utils.file import decode_filename
from app.schemas import (
    NamespaceCreate,
    NamespaceUpdate,
    NamespaceResponse,
    NamespaceListItem,
    FileWithUrl,
    PaginationInfo,
    ResponseMessage,
    ListResponseData,
)
from app.domain.protocols import FileStorage
from app.services.namespace_service import NamespaceService
from app.core.dependencies import get_namespace_service, get_storage_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/",
    response_model=ResponseMessage[NamespaceResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Создать namespace",
)
async def create_namespace(
    user_id: int,
    namespace: NamespaceCreate,
    service: NamespaceService = Depends(get_namespace_service),
):
    """
    Создает новый namespace для пользователя.
    
    Args:
        user_id: ID пользователя
        namespace: Данные для создания
        
    Returns:
        ID созданного namespace
        
    Raises:
        ValidationError: Если namespace с таким именем уже существует
    """
    created = await service.create_namespace(
        name=namespace.name,
        user_id=user_id,
        description=namespace.description,
    )
    return ResponseMessage(data=created)


@router.get(
    "/{namespace_id}",
    response_model=ResponseMessage[NamespaceResponse],
    summary="Получить namespace",
)
async def get_namespace(
    namespace_id: int,
    user_id: int,
    service: NamespaceService = Depends(get_namespace_service),
    storage_service: FileStorage = Depends(get_storage_service),
):
    """
    Получает информацию о namespace со списком файлов и ссылками на скачивание.
    
    Args:
        namespace_id: ID namespace
        user_id: ID пользователя
        
    Returns:
        Информация о namespace с файлами и presigned URL
        
    Raises:
        NotFoundError: Если namespace не найден
        ForbiddenError: Если нет доступа
    """
    ns_data = await service.get_namespace(
        namespace_id=namespace_id,
        user_id=user_id,
    )
    files = await service.get_namespace_files(namespace_id)
    files_with_urls = []
    for f in files:
        if f.file_path:
            try:
                download_url = storage_service.get_file_url(
                    object_name=f.file_path,
                    expires_in=86400,
                )
                filename = decode_filename(f.filename)
                files_with_urls.append(
                    FileWithUrl(
                        id=f.id,
                        filename=filename,
                        file_type=f.file_type,
                        file_size=f.file_size,
                        download_url=download_url,
                        created_at=f.created_at,
                    )
                )
            except Exception as e:
                logger.warning("Failed to generate download URL for file: %s", e)
                continue
    ns_data = NamespaceResponse(
        id=ns_data.id,
        user_id=ns_data.user_id,
        name=ns_data.name,
        description=ns_data.description,
        created_at=ns_data.created_at,
        files=files_with_urls,
    )
    return ResponseMessage(data=ns_data)


@router.get(
    "/",
    response_model=ResponseMessage[ListResponseData[NamespaceListItem]],
    summary="Список namespace пользователя",
)
async def list_namespaces(
    user_id: int,
    page: int = Query(1, ge=1, description="Номер страницы"),
    page_size: int = Query(20, ge=1, le=100, description="Размер страницы"),
    service: NamespaceService = Depends(get_namespace_service),
):
    """
    Получает список namespace пользователя.
    
    Args:
        user_id: ID пользователя
        page: Номер страницы (начиная с 1)
        page_size: Размер страницы
        
    Returns:
        Список namespace с пагинацией (без файлов, с количеством файлов)
    """
    skip = (page - 1) * page_size
    items, total = await service.get_user_namespaces(
        user_id=user_id,
        skip=skip,
        limit=page_size,
    )
    pagination = PaginationInfo(total=total, page=page, page_size=page_size)
    return ResponseMessage(data=ListResponseData(items=items, pagination=pagination))


@router.patch(
    "/{namespace_id}",
    response_model=ResponseMessage[NamespaceResponse],
    summary="Обновить namespace",
)
async def update_namespace(
    namespace_id: int,
    user_id: int,
    namespace: NamespaceUpdate,
    service: NamespaceService = Depends(get_namespace_service),
):
    """
    Обновляет namespace.
    
    Args:
        namespace_id: ID namespace
        user_id: ID пользователя
        namespace: Данные для обновления
        
    Returns:
        ID обновленного namespace
        
    Raises:
        NotFoundError: Если namespace не найден
        ForbiddenError: Если нет доступа
        ValidationError: Если новое имя уже занято
    """
    updated = await service.update_namespace(
        namespace_id=namespace_id,
        user_id=user_id,
        name=namespace.name,
        description=namespace.description,
    )
    return ResponseMessage(data=updated)


@router.delete(
    "/{namespace_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить namespace",
)
async def delete_namespace(
    namespace_id: int,
    user_id: int,
    service: NamespaceService = Depends(get_namespace_service),
    storage_service: FileStorage = Depends(get_storage_service),
):
    """
    Удаляет namespace.
    
    Args:
        namespace_id: ID namespace
        user_id: ID пользователя
        
    Raises:
        NotFoundError: Если namespace не найден
        ForbiddenError: Если нет доступа
    """
    file_paths = await service.delete_namespace(
        namespace_id=namespace_id,
        user_id=user_id,
    )
    for path in file_paths:
        try:
            await storage_service.delete_file(path)
        except Exception as e:
            logger.warning("Failed to delete file from MinIO: %s", e)
    # 204 No Content - тело ответа не возвращается