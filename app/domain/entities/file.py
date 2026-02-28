from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class FileEntity:
    """Файл в хранилище"""
    id: int
    content_hash: str
    source_url: Optional[str]
    transcript_text: str
    file_path: str
    media_metadata: dict
    processing_status: str
    created_at: datetime


@dataclass
class UserFileEntity:
    """Файл в пространстве пользователя"""
    id: int
    user_id: int
    file_id: int
    namespace_id: Optional[int] = None
    custom_title: Optional[str] = None

