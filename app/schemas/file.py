from pydantic import BaseModel, Field, model_validator
from datetime import datetime
from typing import Any, Literal, Optional, TypedDict
from enum import Enum


class RawFileUpload(TypedDict, total=False):
    """Один загруженный файл из HTTP-запроса до сохранения в BlobStorage."""
    content: bytes
    filename: str
    content_type: Optional[str]
    size: int


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
    desktop_updated_at: Optional[datetime] = None
    app_updated_at: Optional[datetime] = None
    last_update_source: Optional[str] = None
    content_hash: str
    is_conflict_copy: bool = False

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
    is_new_file: bool = True  # False если File уже существовал (дедупликация по содержимому)
    is_new_user_file: bool = True  # False если UserFile для этого user+namespace уже существовал


class FileRenameRequest(BaseModel):
    """Запрос на переименование файла."""
    new_title: str = Field(..., min_length=1, max_length=255, description="Новое имя файла")


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
    content_hash: str = Field(..., description="SHA256 содержимого файла для сравнения")

    class Config:
        from_attributes = True


class FileVersionInfo(BaseModel):
    """Минимальная информация о версии файла для pre-save check."""
    user_file_id: int
    content_file_id: int
    updated_at: datetime
    desktop_updated_at: Optional[datetime] = None
    app_updated_at: Optional[datetime] = None
    last_update_source: Optional[str] = None
    content_hash: str


class SyncConflictInfo(BaseModel):
    """Структура конфликта версий для UI."""
    message: str
    server: FileVersionInfo


class SyncUploadRequest(BaseModel):
    """Запрос от Desktop Watcher на отправку локальной версии файла."""
    user_file_id: Optional[int] = Field(None, description="ID user_files, если уже известен watcher")
    namespace_id: Optional[int] = Field(None, description="ID пространства для нового файла")
    filename: str = Field(..., description="Имя файла")
    file_bytes: bytes = Field(..., description="Содержимое файла в виде bytes")

    @model_validator(mode="after")
    def validate_target(self) -> "SyncUploadRequest":
        if self.user_file_id is not None or self.namespace_id is not None:
            return self
        raise ValueError("Нужно передать либо user_file_id, либо namespace_id")

class SyncUploadResponse(BaseModel):
    """Результат принятия desktop upload."""
    file: FileInfo
    conflict_copy_file_id: Optional[int] = None
    created: bool = False
    applied_as: Literal["desktop"] = "desktop"


class CommandStatus(str, Enum):
    """Статус команды."""
    ACKED = "acked"
    FAILED = "failed"
    PENDING = "pending"

class CommandType(str, Enum):
    """Тип команды."""
    UPSERT_FILE = "upsert_file"
    MOVE_FILE = "move_file"
    RENAME_FILE = "rename_file"
    TRASH_FILE = "trash_file"
    DELETE_FILE = "delete_file"
    MOVE_NAMESPACE = "move_namespace"
    RENAME_NAMESPACE = "rename_namespace"
    TRASH_NAMESPACE = "trash_namespace"
    DELETE_NAMESPACE = "delete_namespace"

class SyncCommandItem(BaseModel):
    """Команда для Desktop Watcher."""
    id: int
    user_file_id: Optional[int]
    command_type: CommandType
    payload: dict[str, Any]
    status: CommandStatus
    created_at: datetime


class SyncAckRequest(BaseModel):
    """Подтверждение выполнения команд watcher'ом."""
    command_id: int = Field(..., description="ID команды")
    status: CommandStatus = Field(..., description="Статус команды")


class SyncAckResponse(BaseModel):
    """Результат подтверждения команды."""
    acked_at: datetime
    command_id: int
    command_type: CommandType
    status: CommandStatus

