import logging
from typing import List, Mapping, Optional, Any
from sqlalchemy import text, delete
from sqlalchemy.dialects.postgresql import insert as pg_insert
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
        result: SearchResultRow = {"filename": row.get("filename", "Document")}
        if row.get("chunk_text") is not None:
            result["chunk_text"] = row["chunk_text"]
        if row.get("relevance") is not None:
            result["relevance"] = float(row["relevance"])
        if row.get("namespace_name") is not None:
            result["namespace_name"] = row["namespace_name"]
        if row.get("created_at") is not None:
            result["created_at"] = str(row["created_at"])
        return result

    async def create_batch(
        self,
        file_id: int,
        chunks: List[str],
        embeddings: List[List[float]],
        namespace_id: Optional[int] = None,
    ) -> List[ChunkEntity]:
        rows = [
            {
                "file_id": file_id,
                "chunk_index": idx,
                "chunk_text": chunk_text,
                "embedding": embedding,
            }
            for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings))
        ]
        stmt = (
            pg_insert(VectorEmbedding)
            .values(rows)
            .on_conflict_do_update(
                constraint="uq_vector_embeddings_file_chunk",
                set_={
                    "chunk_text": pg_insert(VectorEmbedding).excluded.chunk_text,
                    "embedding": pg_insert(VectorEmbedding).excluded.embedding,
                },
            )
            .returning(
                VectorEmbedding.id,
                VectorEmbedding.file_id,
                VectorEmbedding.chunk_index,
                VectorEmbedding.chunk_text,
                VectorEmbedding.created_at,
            )
        )
        result = await self.db.execute(stmt)
        ns = namespace_id if namespace_id is not None else 0
        return [
            ChunkEntity(
                id=row.id,
                file_id=row.file_id,
                namespace_id=ns,
                chunk_index=row.chunk_index,
                chunk_text=row.chunk_text,
                embedding=rows[row.chunk_index]["embedding"],
                created_at=row.created_at,
            )
            for row in result.fetchall()
        ]

    async def delete_by_file_id(self, file_id: int) -> int:
        """Удаляет все эмбеддинги для файла. Возвращает количество удалённых записей."""
        result = await self.db.execute(delete(VectorEmbedding).where(VectorEmbedding.file_id == file_id))
        return result.rowcount or 0

    async def search(
        self,
        query_embedding: Optional[List[float]] = None,
        user_id: int = 0,
        limit: int = 5,
        namespace_id: Optional[int] = None,
        file_ids: Optional[List[int]] = None,
        *,
        sql: Optional[str] = None,
    ) -> List[SearchResultRow]:
        """Семантический поиск или выполнение переданного SQL."""
        return await self._execute_search_sql(
            sql or VECTOR_SEARCH_SQL,
            query_embedding=query_embedding,
            user_id=user_id,
            limit=limit,
            namespace_id=namespace_id,
            file_ids=file_ids,
        )

    async def _execute_search_sql(
        self,
        sql: str,
        query_embedding: Optional[List[float]],
        user_id: int,
        limit: int = 5,
        namespace_id: Optional[int] = None,
        file_ids: Optional[List[int]] = None,
    ) -> List[SearchResultRow]:
        sql = sql.replace(":query_embedding::vector", "CAST(:query_embedding AS vector)")
        sql = sql.replace(":namespace_id::integer", "CAST(:namespace_id AS integer)")
        # Подставляем список file_ids напрямую (безопасно — только целые числа)
        if file_ids and "__FILE_IDS__" in sql:
            sql = sql.replace("__FILE_IDS__", ",".join(str(fid) for fid in file_ids))
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
        try:
            stmt = text(sql).bindparams(**params)
            result = await self.db.execute(stmt)
            rows = result.mappings().all()
            return [self._row_to_result(row) for row in rows]
        except Exception:
            await self.db.rollback()
            raise
