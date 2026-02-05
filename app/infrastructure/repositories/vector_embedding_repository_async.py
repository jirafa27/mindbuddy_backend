import logging
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import SearchResultRow
from app.infrastructure.repositories.vector_queries import VECTOR_SEARCH_SQL

logger = logging.getLogger(__name__)


class AsyncVectorEmbeddingRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def execute_search_sql(
        self,
        sql: str,
        query_embedding: Optional[List[float]],
        user_id: int,
        limit: int = 5,
        namespace_id: Optional[int] = None,
    ) -> List[SearchResultRow]:
        # Санитизация: SQLAlchemy text() не понимает :param::type
        sql = sql.replace(":query_embedding::vector", "CAST(:query_embedding AS vector)")

        # Определяем тип запроса: семантический (с embedding) или структурный (без)
        is_semantic_query = ":query_embedding" in sql
        
        if is_semantic_query:
            # Для семантических запросов нужен embedding
            if not query_embedding:
                logger.warning("Semantic query requires embedding, using fallback SQL")
                sql = VECTOR_SEARCH_SQL
            
            # Проверяем обязательные параметры для семантического поиска
            required_params = [":query_embedding", ":user_id", ":limit"]
            if not all(p in sql for p in required_params):
                logger.warning(
                    "Using fallback SQL. Missing params: %s",
                    [p for p in required_params if p not in sql],
                )
                sql = VECTOR_SEARCH_SQL
            
            # Если namespace_id=None но SQL использует :namespace_id — PostgreSQL не сможет
            # определить тип. Используем fallback.
            if namespace_id is None and ":namespace_id" in sql:
                logger.info("namespace_id is None, using fallback SQL without namespace filter")
                sql = VECTOR_SEARCH_SQL
        else:
            # Структурные запросы (списки файлов, папок) — embedding не нужен
            logger.info("Structure query detected (no embedding), executing metadata SQL")
        
        # Базовые параметры
        params: dict = {
            "user_id": user_id,
            "limit": limit,
        }
        
        # Добавляем query_embedding только если он используется в SQL
        if ":query_embedding" in sql and query_embedding:
            vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            params["query_embedding"] = vec_str
        
        # Добавляем namespace_id только если используется в SQL и не None
        if ":namespace_id" in sql and namespace_id is not None:
            params["namespace_id"] = namespace_id
        
        stmt = text(sql).bindparams(**params)
        logger.info("execute_search_sql: executing SQL for user_id=%s namespace_id=%s", user_id, namespace_id)
        result = await self.db.execute(stmt)
        rows = result.mappings().all()
        logger.info("execute_search_sql: got %d rows", len(rows))
        return [dict(row) for row in rows]
