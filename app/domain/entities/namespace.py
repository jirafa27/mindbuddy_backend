from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List


@dataclass
class NamespaceFileItem:
    """Файл в пространстве: данные для отображения и удаления из хранилища"""
    id: int
    content_hash: str
    file_path: Optional[str] = None
    filename: str = ""
    file_type: str = "md"
    file_size: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_conflict_copy: bool = False


@dataclass
class NamespaceEntity:
    """Пространство знаний"""
    id: int
    user_id: int
    name: str
    parent_id: Optional[int] = None
    kind: str = "regular"
    description: Optional[str] = None
    user_files: List[NamespaceFileItem] = field(default_factory=list)
    created_at: Optional[datetime] = None
