"""Схемы для извлечения контента и привязки файлов."""
from typing import Optional

from pydantic import BaseModel, Field


class ContentExtractResponse(BaseModel):
    """Ответ после извлечения контента по URL (без суммаризации)."""
    file_id: int = Field(..., description="ID контент-файла (files.id) для передачи в /summary и /attach")
    title: Optional[str] = Field(None, description="Заголовок страницы или видео")
    parsed_content: Optional[str] = Field(None, description="Распарсенный текст (markdown)")
    source_url: Optional[str] = Field(None, description="Исходный URL")


class AttachFileRequest(BaseModel):
    """Запрос на привязку контент-файла к пространству пользователя."""
    namespace_id: int = Field(..., description="ID пространства, к которому привязывается файл")


class AttachFileResponse(BaseModel):
    """Ответ после привязки файла к пространству."""
    user_file_id: int = Field(..., description="ID записи user_files (для работы с файлом в пространстве)")
    content_file_id: int = Field(..., description="ID контент-файла")
    namespace_id: int
    filename: Optional[str] = None
