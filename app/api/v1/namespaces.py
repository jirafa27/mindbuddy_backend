from datetime import datetime
from fastapi import APIRouter, Depends, status, Query
import logging

from app.utils.file import decode_filename
from app.schemas import (
    NamespaceCreate,
    NamespaceUpdate,
    NamespaceResponse,
    NamespaceListItem,
    NamespaceTreeItem,
    FileWithUrl,
    PaginationInfo,
    ResponseMessage,
    ListResponseData,
)
from app.core.namespace_constants import (
    INBOX_NAMESPACE_KIND,
    TRASH_NAMESPACE_KIND,
    VAULT_ROOT_NAMESPACE_KIND,
)
from app.domain.protocols import FileStorage
from app.services.namespace_service import NamespaceService
from app.core.dependencies import get_namespace_service, get_storage_service, get_current_user
from app.schemas.user import UserResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_namespace_tree(items: list[NamespaceListItem]) -> list[NamespaceTreeItem]:
    nodes = {
        item.id: NamespaceTreeItem(
            id=item.id,
            user_id=item.user_id,
            name=item.name,
            parent_id=item.parent_id,
            kind=item.kind,
            description=item.description,
            created_at=item.created_at,
            files_count=item.files_count,
            children=[],
        )
        for item in items
    }
    roots: list[NamespaceTreeItem] = []
    for item in items:
        node = nodes[item.id]
        if item.parent_id is not None and item.parent_id in nodes:
            nodes[item.parent_id].children.append(node)
        else:
            roots.append(node)
    roots.sort(
        key=lambda node: (
            0 if node.kind == VAULT_ROOT_NAMESPACE_KIND else 1 if node.kind == INBOX_NAMESPACE_KIND else 3 if node.kind == TRASH_NAMESPACE_KIND else 2,
            node.name.lower(),
        )
    )
    return roots


@router.post(
    "/",
    response_model=ResponseMessage[NamespaceResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Создать namespace",
)
async def create_namespace(
    namespace: NamespaceCreate,
    user: UserResponse = Depends(get_current_user),
    service: NamespaceService = Depends(get_namespace_service),
):
    """
    Создает новый namespace для пользователя. Аутентификация: JWT.
    """
    created = await service.create_namespace(
        name=namespace.name,
        user_id=user.id,
        description=namespace.description,
        parent_id=namespace.parent_id,
    )
    return ResponseMessage(data=created)


@router.get(
    "/tree",
    response_model=ResponseMessage[list[NamespaceTreeItem]],
    summary="Дерево namespace пользователя",
)
async def list_namespaces_tree(
    user: UserResponse = Depends(get_current_user),
    service: NamespaceService = Depends(get_namespace_service),
):
    namespaces_with_count, _ = await service.get_user_namespaces(
        user_id=user.id,
        skip=0,
        limit=1000,
    )
    items = [
        NamespaceListItem(
            id=ns.id,
            user_id=ns.user_id,
            name=ns.name,
            parent_id=ns.parent_id,
            kind=ns.kind,
            description=ns.description,
            created_at=ns.created_at,
            files_count=count,
        )
        for ns, count in namespaces_with_count
    ]
    return ResponseMessage(data=_build_namespace_tree(items))


@router.get(
    "/{namespace_id}",
    response_model=ResponseMessage[NamespaceResponse],
    summary="Получить namespace",
)
async def get_namespace(
    namespace_id: int,
    user: UserResponse = Depends(get_current_user),
    service: NamespaceService = Depends(get_namespace_service),
    storage_service: FileStorage = Depends(get_storage_service),
):
    """Получает информацию о namespace. Аутентификация: JWT."""
    ns_data = await service.get_namespace(
        namespace_id=namespace_id,
        user_id=user.id,
    )
    files = await service.get_namespace_files(namespace_id)
    files_with_urls = []
    for f in files:
        if not f.file_path:
            continue
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
                    created_at=f.created_at or datetime.utcnow(),
                )
            )
        except Exception as e:
            logger.warning("Failed to generate download URL for file: %s", e)
    ns_data = NamespaceResponse(
        id=ns_data.id,
        user_id=ns_data.user_id,
        name=ns_data.name,
        parent_id=ns_data.parent_id,
        kind=ns_data.kind,
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
    page: int = Query(1, ge=1, description="Номер страницы"),
    page_size: int = Query(20, ge=1, le=100, description="Размер страницы"),
    user: UserResponse = Depends(get_current_user),
    service: NamespaceService = Depends(get_namespace_service),
):
    """Список namespace текущего пользователя. Аутентификация: JWT."""
    skip = (page - 1) * page_size
    namespaces_with_count, total = await service.get_user_namespaces(
        user_id=user.id,
        skip=skip,
        limit=page_size,
    )
    items = [
        NamespaceListItem(
            id=ns.id,
            user_id=ns.user_id,
            name=ns.name,
            parent_id=ns.parent_id,
            kind=ns.kind,
            description=ns.description,
            created_at=ns.created_at,
            files_count=count,
        )
        for ns, count in namespaces_with_count
    ]
    pagination = PaginationInfo(total=total, page=page, page_size=page_size)
    return ResponseMessage(data=ListResponseData(items=items, pagination=pagination))


@router.patch(
    "/{namespace_id}",
    response_model=ResponseMessage[NamespaceResponse],
    summary="Обновить namespace",
)
async def update_namespace(
    namespace_id: int,
    namespace: NamespaceUpdate,
    user: UserResponse = Depends(get_current_user),
    service: NamespaceService = Depends(get_namespace_service),
):
    """Обновляет namespace. Аутентификация: JWT."""
    updated = await service.update_namespace(
        namespace_id=namespace_id,
        user_id=user.id,
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
    user: UserResponse = Depends(get_current_user),
    namespace_service: NamespaceService = Depends(get_namespace_service),
):
    """Удаляет namespace. Файлы перемещаются в корзину. Аутентификация: JWT."""
    await namespace_service.delete_namespace(namespace_id=namespace_id, user_id=user.id)