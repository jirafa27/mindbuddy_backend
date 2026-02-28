"""Сервис векторного поиска: доменный search() и выполнение SQL для агента."""
from typing import List, Optional

from app.domain.entities import SearchResultRow
from app.domain.protocols import VectorRepository


class SearchService:
    def __init__(self, vector_repository: VectorRepository):
        self._repo = vector_repository

    async def execute_search_sql(
        self,
        sql: str,
        query_embedding: Optional[List[float]],
        user_id: int,
        limit: int = 5,
        namespace_id: Optional[int] = None,
    ) -> List[SearchResultRow]:
        return await self._repo.search(
            query_embedding=query_embedding,
            user_id=user_id,
            limit=limit,
            namespace_id=namespace_id,
            sql=sql,
        )
