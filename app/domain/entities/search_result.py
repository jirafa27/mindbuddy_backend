"""Типы результатов поиска (value objects)."""
from typing import TypedDict


class SearchResultRow(TypedDict):
    """Строка результата векторного поиска."""
    chunk_text: str
    filename: str
    relevance: float
