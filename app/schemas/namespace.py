from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

from app.schemas.file import FileWithUrl


class NamespaceCreate(BaseModel):
    """Схема создания namespace"""
    name: str = Field(..., min_length=1, max_length=255, description="Название namespace")
    description: Optional[str] = Field(None, max_length=1000, description="Описание namespace")


class NamespaceUpdate(BaseModel):
    """Схема обновления namespace"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)


class NamespaceListItem(BaseModel):
    """Схема namespace для списка (без файлов, с количеством)"""
    id: int
    user_id: int
    name: str
    description: Optional[str]
    created_at: datetime
    files_count: int = Field(0, description="Количество файлов в namespace")

    class Config:
        from_attributes = True


class NamespaceResponse(BaseModel):
    """Схема ответа с детальной информацией о namespace (с файлами)"""
    id: int
    user_id: int
    name: str
    description: Optional[str]
    created_at: datetime
    files: List[FileWithUrl] = Field(default_factory=list, description="Список файлов с ссылками на скачивание")

    class Config:
        from_attributes = True
