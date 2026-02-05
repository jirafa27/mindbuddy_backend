"""Состояние графа для RAG-пайплайна POST /ask."""
from typing import TypedDict, Optional, Any


class AskState(TypedDict, total=False):
    """Единое состояние графа LangGraph для /ask."""

    question: str
    namespace_id: int
    user_id: int

    # Файл (опционально)
    file_content: bytes
    filename: str


    blob_key: Optional[str]

    # После сохранения в DBAgent
    file_id: Optional[int]

    # Поиск
    query_embedding: list[float]
    sql_query: str
    search_result: list[dict[str, Any]]  # chunk_text, filename, relevance
    db_error: Optional[str]
    retry_count: int

    # Ответ
    agent_steps: list[str]
    answer: str
    sources: list[dict[str, Any]]  # filename, relevance
