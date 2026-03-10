"""Типы результатов поиска (value objects)."""
from typing import TypedDict


class SearchResultRow(TypedDict, total=False):
    """
    Строка результата поиска.

    Семантический поиск: chunk_text, filename, relevance.
    Структурный запрос (list_files): filename, namespace_name, created_at.
    """
    chunk_text: str
    filename: str
    relevance: float
    namespace_name: str
    created_at: str
