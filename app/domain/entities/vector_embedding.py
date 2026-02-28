from dataclasses import dataclass
from typing import List
from datetime import datetime

@dataclass
class VectorEmbeddingEntity:
    id: int
    file_id: int
    chunk_index: int
    chunk_text: str
    embedding: List[float]
    created_at: datetime