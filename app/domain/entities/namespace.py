from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=False)
class NamespaceEntity:
    """Пространство знаний"""
    id: int
    user_id: int
    name: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
