"""Обратная совместимость: реэкспорт из app.infrastructure.parsers."""
from app.infrastructure.parsers import (
    YouTubeParser,
    HTMLParser,
    ParsedContent,
    ContentParser,
)

__all__ = ["YouTubeParser", "HTMLParser", "ContentParser", "ParsedContent"]
