from fastapi import APIRouter, Depends, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas import FileResponse
from app.services.file_service import FileService
from app.core.dependencies import get_file_service
from app.tasks.file_processing import process_file_embeddings


router = APIRouter()


@router.post("/upload", response_model=FileResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    namespace_id: int = Form(...),
    user_id: int = Form(...),
    db: AsyncSession = Depends(get_db),
    file_service: FileService = Depends(get_file_service),
):
    """
    Загружает файл, валидирует и сохраняет в БД.
    Разбиение на чанки и векторизация выполняются асинхронно через RabbitMQ.
    Возвращает file_id и task_id для отслеживания статуса обработки.
    """
    file_content = await file.read()
    
    # Создаем файл в БД
    file_created = await file_service.create_file(
        file_content=file_content,
        filename=file.filename,
        namespace_id=namespace_id,
        user_id=user_id,
        db=db,
    )
    
    # Отправляем задачу на обработку (разбиение на чанки и векторизация)
    task = process_file_embeddings.delay(
        file_id=file_created.file_id,
        text=file_created.text,
        namespace_id=namespace_id,
    )
    
    return FileResponse(
        file_id=file_created.file_id,
        filename=file_created.filename,
        task_id=task.id,
        status="processing"
    )

