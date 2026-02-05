from fastapi import APIRouter, Depends, UploadFile, File, status, Response, Query
from fastapi.responses import StreamingResponse
import io
import logging

import uuid

from app.utils.file import decode_filename, encode_filename_for_header
from app.schemas.base import ResponseMessage
from app.schemas.file import (
    FileResponse,
    SyncToLocalRequest,
    SyncToLocalResponse,
)
from app.domain.protocols import WatcherTaskPublisher, FileStorage
from app.services.file_service import FileService
from app.core.dependencies import (
    get_file_service,
    get_rabbitmq_service,
    get_storage_service,
    get_user_by_telegram_id,
)
from app.infrastructure.workers.file_processing import process_file_embeddings
from app.schemas.user import UserResponse
from app.core.exceptions import NotFoundError, ForbiddenError, ValidationError, FileProcessingError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload", response_model=ResponseMessage[FileResponse], status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    namespace_id: int = Query(...),
    user_id: int = Query(...),
    file_service: FileService = Depends(get_file_service)
):
    """
    Загружает файл на сервер и запускает векторизацию.
    
    **Используется для:**
    - Загрузки из Telegram бота
    - Загрузки из Desktop Watcher
    
    **Процесс:**
    1. Сохраняет файл в MinIO
    2. Создает запись в БД
    3. Запускает задачу векторизации (RabbitMQ → Celery)
    
    **Для Telegram бота:** После этого эндпоинта вызовите `/sync-to-local` 
    чтобы файл скачался на компьютер пользователя через Desktop Watcher.
    
    Returns:
        file_id: ID файла в БД
        task_id: ID задачи векторизации
        status: "processing"
    """
    file_content = await file.read()
    
    # Декодируем имя файла, если оно пришло в URL-encoded виде
    # FastAPI может передавать имена файлов с кириллицей в закодированном виде
    filename = decode_filename(file.filename or "unnamed_file")
    
    # Создаем файл в БД и MinIO
    file_created = await file_service.upload_file(
        file_content=file_content,
        filename=filename,
        namespace_id=namespace_id,
        user_id=user_id,
    )
    
    # Отправляем задачу на векторизацию в RabbitMQ
    task = process_file_embeddings.delay(
        file_id=file_created.file_id,
        text=file_created.text,
        namespace_id=namespace_id,
        filename=file_created.filename,
    )

    file_response = FileResponse(
        file_id=file_created.file_id,
        filename=file_created.filename,
        task_id=task.id,
        status="processing"
    )
    return ResponseMessage[FileResponse](data=file_response)


@router.post("/sync-to-local", response_model=ResponseMessage[SyncToLocalResponse], status_code=status.HTTP_202_ACCEPTED)
async def sync_to_local(
    request: SyncToLocalRequest,
    watcher_publisher: WatcherTaskPublisher = Depends(get_rabbitmq_service),
    storage_service: FileStorage = Depends(get_storage_service),
    file_service: FileService = Depends(get_file_service),
):
    """
    Отправляет задачу Desktop Watcher'у для скачивания файла на локальный диск.
    
    **Используется только для Telegram бота:**
    Когда файл загружен через Telegram, вызовите этот эндпоинт чтобы 
    Desktop Watcher скачал файл на компьютер пользователя и начал мониторинг изменений.
    
    **Desktop Watcher должен:**
    1. Получить задачу из RabbitMQ очереди "watcher_tasks"
    2. Скачать файл по presigned URL из MinIO
    3. Сохранить на диск пользователя
    4. Начать мониторинг изменений
    
    **Для Desktop Watcher:** Этот эндпоинт НЕ нужен, т.к. файл уже на диске.
    
    Args:
        file_id: ID файла на сервере
        user_id: ID пользователя
        local_path: Желаемый путь (опционально, watcher может выбрать сам)
    
    Returns:
        task_id: Уникальный ID задачи для отслеживания
        status: "pending"
    """
    # Получаем информацию о файле из БД
    file = await file_service.get_file(file_id=request.file_id, user_id=request.user_id)
    
    # Генерируем presigned URL для скачивания (действует 24 часа)
    try:
        download_url = storage_service.get_file_url(
            object_name=file.file_path,
            expires_in=86400,
        )
    except Exception as e:
        raise FileProcessingError(f"Не удалось сгенерировать ссылку для скачивания: {str(e)}")
    
    # Декодируем имя файла, если оно URL-encoded (для старых записей)
    filename = decode_filename(file.filename)
    
    # Отправляем задачу в RabbitMQ напрямую (без Celery)
    try:
        watcher_publisher.send_watcher_task(
            file_id=file.id,
            user_id=file.user_id,
            filename=filename,
            file_type=file.file_type,
            file_size=file.file_size,
            download_url=download_url,
            local_path=request.local_path,
        )
    except Exception as e:
        raise FileProcessingError(f"Не удалось отправить задачу в RabbitMQ: {str(e)}")
    
    # Генерируем уникальный task_id для отслеживания
    task_id = str(uuid.uuid4())
    
    sync_to_local_response = SyncToLocalResponse(
        file_id=request.file_id,
        task_id=task_id,
        status="pending",
        message="Задача отправлена Desktop Watcher'у в очередь 'watcher_tasks'"
    )
    return ResponseMessage[SyncToLocalResponse](data=sync_to_local_response)


@router.get("/download/{file_id}")
async def download_file(
    file_id: int,
    user: UserResponse = Depends(get_user_by_telegram_id),
    file_service: FileService = Depends(get_file_service),
):
    """
    Скачивает файл из хранилища.
    
    **Аутентификация:** По Telegram ID (query параметр `telegram_id`)
    
    Args:
        file_id: ID файла
        telegram_id: Telegram ID пользователя (query параметр)
    
    Returns:
        Файл для скачивания (бинарный поток)
        
    Raises:
        404: Если пользователь или файл не найден
        403: Если нет доступа к файлу
    """
    file_content, filename, content_type = await file_service.download_file(
        file_id=file_id,
        user_id=user.id,
    )
    
    # Декодируем имя файла, если оно URL-encoded (для старых записей)
    filename = decode_filename(filename)
    
    # StreamingResponse возвращается напрямую (не оборачивается в ResponseMessage)
    return StreamingResponse(
        io.BytesIO(file_content),
        media_type=content_type,
        headers={
            "Content-Disposition": encode_filename_for_header(filename)
        }
    )


@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    file_id: int,
    user: UserResponse = Depends(get_user_by_telegram_id),
    file_service: FileService = Depends(get_file_service),
):
    """
    Удаляет файл из хранилища и базы данных.
    
    **Аутентификация:** По Telegram ID (query параметр `telegram_id`)
    
    Args:
        file_id: ID файла
        telegram_id: Telegram ID пользователя (query параметр)
        
    Raises:
        404: Если пользователь или файл не найден
        403: Если нет доступа к файлу
    """
    await file_service.delete_file(
        file_id=file_id,
        user_id=user.id,
    )