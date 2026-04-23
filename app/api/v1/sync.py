from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.core.dependencies import (
    get_sync_service,
    get_user_by_watcher_token,
    get_namespace_service,
)
from app.services.namespace_service import NamespaceService
from app.schemas.base import ResponseMessage
from app.schemas.file import (
    SyncAckRequest,
    SyncAckResponse,
    SyncCommandItem,
    SyncUploadRequest,
    SyncUploadResponse,
)
from app.schemas.namespace import (
    NamespaceMoveRequest,
    NamespaceRenameRequest,
    SyncNamespaceCreate,
    NamespaceStructureItem,
)
from app.schemas.user import UserResponse
from app.services.sync_service import SyncService

router = APIRouter()


@router.post("/upload", response_model=ResponseMessage[SyncUploadResponse])
async def sync_upload(
    user_file_id: Optional[int] = Form(None),
    namespace_id: Optional[int] = Form(None),
    file: UploadFile = File(...),
    user: UserResponse = Depends(get_user_by_watcher_token),
    sync_service: SyncService = Depends(get_sync_service),
) -> ResponseMessage[SyncUploadResponse]:
    """
    Загрузка локальной версии файла на сервер.
    Вотчер отправляет file (бинарный) или content (текст).
    """
    upload_bytes = await file.read()
    filename = file.filename or "unnamed_file"
    body = SyncUploadRequest(
        user_file_id=user_file_id,
        namespace_id=namespace_id,
        filename=filename,
        file_bytes=upload_bytes,
    )
    result = await sync_service.upload_file(
        user_id=user.id,
        upload_request=body,
    )
    return ResponseMessage[SyncUploadResponse](data=result)


@router.get("/commands", response_model=ResponseMessage[list[SyncCommandItem]])
async def get_sync_commands(
    limit: int = 100,
    user: UserResponse = Depends(get_user_by_watcher_token),
    sync_service: SyncService = Depends(get_sync_service),
) -> ResponseMessage[list[SyncCommandItem]]:
    """
    Получение списка команд синхронизации, ожидающих выполнения
    """
    result = await sync_service.list_pending_commands(user_id=user.id, limit=limit)
    return ResponseMessage[list[SyncCommandItem]](data=result)


@router.post("/commands/ack", response_model=ResponseMessage[SyncAckResponse])
async def ack_sync_command(
    body: SyncAckRequest,
    user: UserResponse = Depends(get_user_by_watcher_token),
    sync_service: SyncService = Depends(get_sync_service),
) -> ResponseMessage[SyncAckResponse]:
    """
    Подтверждение выполнения команды синхронизации
    """
    result = await sync_service.ack_command(user_id=user.id, command_id=body.command_id, status=body.status)
    return ResponseMessage[SyncAckResponse](data=result)


@router.get("/structure", response_model=ResponseMessage[list[NamespaceStructureItem]])
async def get_sync_structure(
    user: UserResponse = Depends(get_user_by_watcher_token),
    namespace_service: NamespaceService = Depends(get_namespace_service),
) -> ResponseMessage[list[NamespaceStructureItem]]:
    """
    Получение списка пространств с файлами пользователя
    """
    result = await namespace_service.get_namespaces_with_files(user_id=user.id)
    return ResponseMessage[list[NamespaceStructureItem]](data=result)

@router.post(
    "/namespaces",
    response_model=ResponseMessage[NamespaceStructureItem],
    status_code=status.HTTP_201_CREATED,
)
async def create_sync_namespace(
    body: SyncNamespaceCreate,
    user: UserResponse = Depends(get_user_by_watcher_token),
    sync_service: SyncService = Depends(get_sync_service),
) -> ResponseMessage[NamespaceStructureItem]:
    """
    Создание пространства на сервере по запросу watcher'а.
    """
    result = await sync_service.create_desktop_namespace(
        user_id=user.id,
        name=body.name,
        vault_name=body.vault_name,
        parent_id=body.parent_id,
        description=body.description,
        relative_path=body.relative_path,
    )
    return ResponseMessage[NamespaceStructureItem](data=result)


@router.delete("/files/{user_file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_sync_file(
    user_file_id: int,
    user: UserResponse = Depends(get_user_by_watcher_token),
    sync_service: SyncService = Depends(get_sync_service),
):
    """
    Удаление файла на сервере
    """

    await sync_service.apply_desktop_delete(
        user_id=user.id,
        user_file_id=user_file_id,
    )


@router.delete("/namespaces/{namespace_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_sync_namespace(
    namespace_id: int,
    user: UserResponse = Depends(get_user_by_watcher_token),
    sync_service: SyncService = Depends(get_sync_service),
) -> None:
    """
    Полное удаление пространства и всех пользовательских файлов в нём на сервере.
    Используется watcher'ом при удалении папки на ПК.
    """
    await sync_service.apply_desktop_delete_namespace(
        user_id=user.id,
        namespace_id=namespace_id,
    )


@router.put(
    "/namespaces/{namespace_id}/move",
    response_model=ResponseMessage[NamespaceStructureItem],
)
async def move_sync_namespace(
    namespace_id: int,
    body: NamespaceMoveRequest,
    user: UserResponse = Depends(get_user_by_watcher_token),
    namespace_service: NamespaceService = Depends(get_namespace_service),
) -> ResponseMessage[NamespaceStructureItem]:
    """
    Перемещение пространства на сервере по запросу watcher'а.
    """
    result = await namespace_service.move_namespace(
        namespace_id=namespace_id,
        user_id=user.id,
        target_parent_id=body.target_parent_id,
        notify_watcher=False,
    )
    return ResponseMessage[NamespaceStructureItem](data=result)


@router.put("/namespaces/{namespace_id}/rename", status_code=status.HTTP_204_NO_CONTENT)
async def rename_sync_namespace(
    namespace_id: int,
    body: NamespaceRenameRequest,
    user: UserResponse = Depends(get_user_by_watcher_token),
    namespace_service: NamespaceService = Depends(get_namespace_service),
) -> None:
    """
    Переименование пространства на сервере по запросу watcher'а.
    """
    await namespace_service.update_namespace(
        namespace_id=namespace_id,
        user_id=user.id,
        name=body.new_name,
        description=None,
        notify_watcher=False,
    )


@router.put("/files/{user_file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def rename_sync_file(
    user_file_id: int,
    new_name: str,
    user: UserResponse = Depends(get_user_by_watcher_token),
    sync_service: SyncService = Depends(get_sync_service),
) -> None:
    """
    Переименование файла на сервере
    """

    await sync_service.apply_desktop_rename(
        user_id=user.id,
        user_file_id=user_file_id,
        new_name=new_name,
    )


@router.put("/files/{user_file_id}/move", status_code=status.HTTP_204_NO_CONTENT)
async def move_sync_file(
    user_file_id: int,
    namespace_id: int,
    user: UserResponse = Depends(get_user_by_watcher_token),
    sync_service: SyncService = Depends(get_sync_service),
) -> None:
    """
    Перемещение файла на сервере в указанное пространство
    """

    await sync_service.apply_desktop_move(
        user_id=user.id,
        user_file_id=user_file_id,
        namespace_id=namespace_id,
    )

