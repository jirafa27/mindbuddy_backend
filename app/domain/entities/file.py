from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=False)
class FileEntity:
    """Метаданные файла"""
    id: int
    namespace_id: int
    user_id: int
    filename: str
    file_path: Optional[str]
    file_type: str
    file_size: int
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
