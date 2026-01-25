from pydantic import BaseModel, Field
from datetime import datetime


class FileUploadRequest(BaseModel):
    """Схема запроса на загрузку файла"""
    namespace_id: int = Field(..., description="ID пространства знаний")
    user_id: int = Field(..., description="ID пользователя")


class ChunkResponse(BaseModel):
    """Схема ответа с информацией о чанке"""
    chunk_index: int
    text_preview: str = Field(..., max_length=200)

    class Config:
        from_attributes = True


class FileResponse(BaseModel):
    """Схема ответа после загрузки файла"""
    file_id: int
    filename: str
    task_id: str
    status: str = "processing"

    class Config:
        from_attributes = True


class FileInfo(BaseModel):
    """Полная информация о файле"""
    id: int
    user_id: int
    namespace_id: int
    filename: str
    file_type: str
    file_size: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class FileProcessingResult(BaseModel):
    """Результат обработки файла"""
    file_id: int
    chunks_count: int
    status: str = "success"


class FileCreated(BaseModel):
    """Данные о созданном файле (без task_id)"""
    file_id: int
    filename: str
    text: str  # Текст для передачи в Celery задачу
