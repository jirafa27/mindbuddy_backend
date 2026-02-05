from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass(frozen=False)
class ChunkEntity:
    """Текстовый чанк с эмбеддингом"""
    id: int
    file_id: int
    namespace_id: int
    chunk_index: int
    chunk_text: str
    embedding: List[float]
    created_at: Optional[datetime] = None
