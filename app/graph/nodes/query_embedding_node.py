"""Узел вычисления query_embedding по question (для векторного поиска)."""
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.graph.state import AskState
from app.domain.protocols import EmbeddingProvider


class QueryEmbeddingNode:
    """Вычисляет эмбеддинг запроса (question) один раз для векторного поиска."""

    def __init__(self, *, embedding_service: EmbeddingProvider):
        self.embedding_service = embedding_service

    async def run(self, state: AskState, config: RunnableConfig) -> dict[str, Any]:
        # Если пользователь выбрал конкретные файлы, используем полный вопрос —
        # search_query вроде "файлы" слишком короткий и даёт нерелевантные результаты.
        # В остальных случаях search_query имеет приоритет как очищенный LLM-термин.
        search_query = state.get("search_query") or ""
        question = state.get("question") or ""
        search_file_ids = state.get("search_file_ids")
        if search_file_ids and len(search_query.split()) <= 2:
            text_to_embed = question or search_query
        else:
            text_to_embed = search_query or question
        if not text_to_embed.strip():
            return {}

        try:
            query_embedding = await self.embedding_service.generate_query_embedding(text_to_embed)
            return {"query_embedding": query_embedding}
        except Exception:
            return {"query_embedding": []}
