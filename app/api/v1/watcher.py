"""API для Desktop Watcher: WebSocket, структура файлов, задачи."""
import json
import logging

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from app.infrastructure.db.base import AsyncSessionLocal
from app.schemas.user import UserResponse
from app.infrastructure.repositories import UserRepository
from app.schemas.base import ResponseMessage
from app.schemas.file import StructureResponse, NamespaceStructureItem, FileStructureItem
from app.services.namespace_service import NamespaceService
from app.services.websocket_manager import WebSocketManager
from app.core.dependencies import get_websocket_manager, get_user_by_watcher_token, get_namespace_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/structure", response_model=ResponseMessage[StructureResponse])
async def get_files_structure(
    user: UserResponse = Depends(get_user_by_watcher_token),
    service: NamespaceService = Depends(get_namespace_service),
):
    """
    Возвращает полную структуру файлов и папок для Desktop Watcher.

    Аутентификация: query-параметр `token` — Watcher токен пользователя.

    Формат: список пространств (namespace = папка на диске), в каждом — список файлов
    с id, filename, file_size, updated_at. Watcher сравнивает с локальной ФС и создаёт/
    удаляет/обновляет файлы при расхождении.
    """
    namespaces = await service.get_user_namespaces_with_files(user.id)
    items = []
    for ns in namespaces:
        files_sorted = sorted(ns.files, key=lambda f: f.filename)
        items.append(
            NamespaceStructureItem(
                id=ns.id,
                name=ns.name,
                files=[
                    FileStructureItem(
                        id=f.id,
                        filename=f.filename,
                        file_size=f.file_size,
                        updated_at=f.updated_at,
                    )
                    for f in files_sorted
                ],
            )
        )
    return ResponseMessage(data=StructureResponse(namespaces=items))


@router.websocket("/ws")
async def watcher_websocket(
    websocket: WebSocket,
    token: str = Query(..., description="Токен аутентификации Watcher"),
    websocket_manager: WebSocketManager = Depends(get_websocket_manager),
):
    """
    WebSocket соединение для получения задач Watcher в реальном времени.

    Использование:
    - Watcher подключается к этому эндпоинту через WebSocket
    - Сервер автоматически отправляет задачи как только они появляются в очереди
    - Соединение остаётся открытым до отключения клиента

    Аутентификация: По токену Watcher (query параметр `token`)
    - Токен выдаётся при регистрации пользователя
    - Можно получить токен через GET /api/v1/users/telegram/{telegram_id}

    Формат сообщений:
    - Сервер → Клиент: JSON с задачей (WatcherTaskResponse)
    - Клиент → Сервер: можно отправлять heartbeat/ping сообщения (опционально)

    Пример подключения:
    ```python
    import websockets

    uri = f"ws://api.example.com/api/v1/watcher/ws?token={WATCHER_TOKEN}"
    async with websockets.connect(uri) as ws:
        async for message in ws:
            task = json.loads(message)
            # Обработать задачу
    ```
    """
    async with AsyncSessionLocal() as db:
        repository = UserRepository(db)
        user = await repository.get_by_watcher_token(token)

        if not user:
            await websocket.close(code=1008, reason="Invalid token")
            return

    await websocket_manager.connect_websocket(websocket, user.id)

    try:
        await websocket.send_json({
            "type": "connected",
            "message": "WebSocket connected successfully",
            "user_id": user.id,
        })

        while True:
            try:
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except json.JSONDecodeError:
                    pass
            except WebSocketDisconnect:
                break
    except Exception as e:
        logger.error("WebSocket error for user %s: %s", user.id, e)
    finally:
        await websocket_manager.disconnect_websocket(user.id)
