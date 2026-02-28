"""Сущность результата парсинга контента."""
import hashlib
from dataclasses import dataclass
from enum import Enum

class ContentType(Enum):
    YOUTUBE = "youtube"
    HTML = "html"
    FILE = "file"


@dataclass
class ParsedContent:
    """Результат парсинга контента (YouTube, HTML, файл)."""
    text: str
    title: str
    source_url: str
    content_hash: str
    content_type: ContentType
    file_size: int
    fallback_used: bool = False
    
    @staticmethod
    def compute_hash(content: str) -> str:
        """Вычисляет SHA-256 хэш контента."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
