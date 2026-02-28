from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class FileUploadRequest(BaseModel):
    """Схема запроса на загрузку файла"""
    namespace_id: int = Field(..., description="ID пространства знаний")
    user_id: int = Field(..., description="ID пользователя")


class SyncToLocalRequest(BaseModel):
    """Схема запроса на синхронизацию файла с локальным компьютером"""
    file_id: int = Field(..., description="ID файла на сервере")
    user_id: int = Field(..., description="ID пользователя")
    local_path: Optional[str] = Field(None, description="Желаемый путь на диске (опционально)")


class SyncToLocalResponse(BaseModel):
    """Схема ответа на запрос синхронизации"""
    file_id: int
    task_id: str
    status: str = "pending"
    message: str = "Задача отправлена watcher'у"


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
    message: Optional[str] = None  # Например: "Не удалось получить текст видео, сохранены только метаданные"

    class Config:
        from_attributes = True


class FileInfo(BaseModel):
    """Полная информация о файле"""
    user_file_id: int
    content_file_id: int 
    user_id: int
    namespace_id: Optional[int] = None
    filename: str
    file_type: str
    file_size: int
    created_at: datetime
    updated_at: datetime
    file_path: Optional[str] = None

    class Config:
        from_attributes = True


class FileProcessingResult(BaseModel):
    """Результат обработки файла"""
    file_id: int
    chunks_count: int
    status: str = "success"


class DeduplicationResult(BaseModel):
    """Результат проверки дедупликации."""
    is_duplicate: bool
    existing_file_id: Optional[int] = None


class FileCreated(BaseModel):
    """Данные о созданном файле"""
    file_id: int  # UserFile.id (для ответа и задач)
    content_file_id: int  # File.id (для эмбеддингов и суммаризации)
    filename: str
    text: str  # Текст для передачи в Celery задачу


class IngestUrlResult(BaseModel):
    """Результат индексации URL (без суммаризации)."""
    file_id: int
    content_file_id: int
    filename: str
    text: str
    is_duplicate: bool = False
    message: Optional[str] = None


class ProcessUserLinkResult(BaseModel):
    """Результат добавления ссылки пользователем (content deduplication flow)."""
    user_file_id: int
    content_file_id: int
    custom_title: str
    is_cache_hit: bool
    processing_status: str
    text: Optional[str] = None


class FileInNamespace(BaseModel):
    """Файл в namespace (без URL; URL добавляется в API при отдаче клиенту)."""
    id: int
    filename: str
    file_type: str
    file_size: int
    created_at: datetime
    file_path: str = Field(..., description="Путь в хранилище (для генерации URL или удаления)")

    class Config:
        from_attributes = True


class FileWithUrl(BaseModel):
    """Информация о файле с presigned URL для скачивания"""
    id: int
    filename: str
    file_type: str
    file_size: int
    download_url: str = Field(..., description="Presigned URL для скачивания (действует 24 часа)")
    created_at: datetime

    class Config:
        from_attributes = True

class FileStructureItem(BaseModel):
    """Элемент структуры: один файл (для watcher)"""
    id: int = Field(..., description="ID файла")
    filename: str = Field(..., description="Имя файла")
    file_size: int = Field(..., description="Размер в байтах")
    updated_at: datetime = Field(..., description="Время последнего обновления для сравнения с локальным")

    class Config:
        from_attributes = True


class NamespaceStructureItem(BaseModel):
    """Элемент структуры: namespace как «папка» с файлами"""
    id: int = Field(..., description="ID пространства знаний")
    name: str = Field(..., description="Имя пространства (имя папки на диске)")
    files: list[FileStructureItem] = Field(default_factory=list, description="Файлы в этом пространстве")

    class Config:
        from_attributes = True


class StructureResponse(BaseModel):
    """Полная структура файлов и папок для watcher (все namespace пользователя)"""
    namespaces: list[NamespaceStructureItem] = Field(
        default_factory=list,
        description="Список пространств (папок), в каждом — список файлов",
    )
