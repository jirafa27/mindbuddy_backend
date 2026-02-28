import logging
from typing import List, Mapping, Optional, Any
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import ChunkEntity, SearchResultRow
from app.infrastructure.db.models import VectorEmbedding
from app.infrastructure.repositories.vector_queries import VECTOR_SEARCH_SQL


logger = logging.getLogger(__name__)


class PgVectorRepository:
    """Реализация векторного репозитория."""

    def __init__(self, db: AsyncSession):
        self.db = db

    def _row_to_result(self, row: Mapping[str, Any]) -> SearchResultRow:
        return SearchResultRow(
            chunk_text=row["chunk_text"],
            filename=row["filename"],
            relevance=float(row["relevance"]),
        )

    async def create_batch(
        self,
        file_id: int,
        chunks: List[str],
        embeddings: List[List[float]],
        namespace_id: Optional[int] = None,
    ) -> List[ChunkEntity]:
        models = [
            VectorEmbedding(
                file_id=file_id,
                chunk_index=idx,
                chunk_text=chunk_text,
                embedding=embedding,
            )
            for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings))
        ]
        self.db.add_all(models)
        await self.db.flush()
        ns = namespace_id if namespace_id is not None else 0
        return [
            ChunkEntity(
                id=m.id,
                file_id=m.file_id,
                namespace_id=ns,
                chunk_index=m.chunk_index,
                chunk_text=m.chunk_text,
                embedding=list(m.embedding),
                created_at=m.created_at,
            )
            for m in models
        ]

    async def search(
        self,
        query_embedding: Optional[List[float]] = None,
        user_id: int = 0,
        limit: int = 5,
        namespace_id: Optional[int] = None,
        *,
        sql: Optional[str] = None,
    ) -> List[SearchResultRow]:
        """
        Семантический поиск (sql=None, используется VECTOR_SEARCH_SQL) или выполнение переданного SQL (агент).
        """
        actual_sql = sql if sql is not None else VECTOR_SEARCH_SQL
        return await self._execute_search_sql(
            actual_sql,
            query_embedding=query_embedding,
            user_id=user_id,
            limit=limit,
            namespace_id=namespace_id,
        )

    async def _execute_search_sql(
        self,
        sql: str,
        query_embedding: Optional[List[float]],
        user_id: int,
        limit: int = 5,
        namespace_id: Optional[int] = None,
    ) -> List[SearchResultRow]:
        sql = sql.replace(":query_embedding::vector", "CAST(:query_embedding AS vector)")
        is_semantic = ":query_embedding" in sql
        if is_semantic:
            if not query_embedding:
                logger.warning("Semantic query requires embedding, using fallback SQL")
                sql = VECTOR_SEARCH_SQL
            required = [":query_embedding", ":user_id", ":limit"]
            if not all(p in sql for p in required):
                logger.warning("Using fallback SQL. Missing: %s", [p for p in required if p not in sql])
                sql = VECTOR_SEARCH_SQL
        else:
            logger.info("Structure query (no embedding), executing metadata SQL")
        params: dict = {"user_id": user_id, "limit": limit}
        if ":query_embedding" in sql and query_embedding:
            params["query_embedding"] = "[" + ",".join(str(x) for x in query_embedding) + "]"
        if ":namespace_id" in sql:
            params["namespace_id"] = namespace_id
        stmt = text(sql).bindparams(**params)
        result = await self.db.execute(stmt)
        rows = result.mappings().all()
        return [self._row_to_result(row) for row in rows]
