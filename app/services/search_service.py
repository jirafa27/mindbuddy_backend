"""Сервис векторного поиска по SQL (делегирует AsyncVectorEmbeddingRepository)."""
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import SearchResultRow
from app.infrastructure.repositories import AsyncVectorEmbeddingRepository


class SearchService:
    def __init__(self, db: AsyncSession):
        self._repo = AsyncVectorEmbeddingRepository(db)

    async def execute_search_sql(
        self,
        sql: str,
        query_embedding: Optional[List[float]],
        user_id: int,
        limit: int = 5,
        namespace_id: Optional[int] = None,
    ) -> List[SearchResultRow]:
        return await self._repo.execute_search_sql(
            sql=sql,
            query_embedding=query_embedding,
            user_id=user_id,
            limit=limit,
            namespace_id=namespace_id,
        )
