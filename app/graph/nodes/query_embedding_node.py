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
        question = state.get("question") or ""
        if not question.strip():
            return {}

        try:
            query_embedding = await self.embedding_service.generate_query_embedding(question)
            return {"query_embedding": query_embedding}
        except Exception:
            return {"query_embedding": []}
