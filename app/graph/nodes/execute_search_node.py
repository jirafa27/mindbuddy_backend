"""ExecuteSearchNode: выполнение векторного (или гибридного) поиска."""
import logging
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from app.graph.state import AskState
from app.services.search_service import SearchService
from app.infrastructure.repositories.vector_queries import (
    VECTOR_SEARCH_SQL,
    VECTOR_SEARCH_BY_FILES_SQL,
    HYBRID_SEARCH_SQL,
    HYBRID_SEARCH_BY_FILES_SQL,
)

logger = logging.getLogger(__name__)


class ExecuteSearchNode:
    """
    Выполняет SQL-поиск с эмбеддингами; при ошибке возвращает db_error.
    SearchService берётся из config.

    Стратегия выбора SQL:
    - Если sql_query задан явно (SQLAgent) — использует его (чистый векторный).
    - Если search_file_ids задан и есть search_query → гибридный поиск по файлам.
    - Если search_file_ids задан без search_query → векторный поиск по файлам.
    - Иначе → гибридный или векторный глобальный поиск.
    """

    async def run(self, state: AskState, config: RunnableConfig) -> dict[str, Any]:
        configurable = config.get("configurable") or {}
        search_service: Optional[SearchService] = configurable.get("search_service")

        agent_steps = list(state.get("agent_steps") or []) + ["ExecuteSearchNode"]

        sql_query = state.get("sql_query")
        user_id = state.get("user_id")
        namespace_id = state.get("namespace_id")
        query_embedding = state.get("query_embedding")
        search_file_ids = state.get("search_file_ids")
        search_query = state.get("search_query")
        limit = state.get("search_limit") or 10

        if user_id is None:
            return {"agent_steps": agent_steps}

        # Выбираем SQL и fts_query
        fts_query: Optional[str] = None
        if not sql_query:
            if search_file_ids and search_query:
                # Гибридный поиск по конкретным файлам
                sql_query = HYBRID_SEARCH_BY_FILES_SQL
                fts_query = search_query
            elif search_file_ids:
                # Только векторный поиск по файлам (нет текстового запроса)
                sql_query = VECTOR_SEARCH_BY_FILES_SQL
            elif search_query:
                # Гибридный глобальный поиск
                sql_query = HYBRID_SEARCH_SQL
                fts_query = search_query
            else:
                sql_query = VECTOR_SEARCH_SQL

        if search_service is None:
            return {
                "search_result": [],
                "db_error": "search_service not in config",
                "agent_steps": agent_steps,
            }

        try:
            rows = await search_service.execute_search_sql(
                sql=sql_query,
                query_embedding=query_embedding,
                user_id=user_id,
                limit=limit,
                namespace_id=namespace_id,
                file_ids=search_file_ids,
                fts_query=fts_query,
            )
            return {
                "search_result": rows,
                "db_error": None,
                "agent_steps": agent_steps,
            }
        except Exception as e:
            logger.warning("ExecuteSearchNode failed: %s", e)
            return {
                "search_result": [],
                "db_error": str(e),
                "agent_steps": agent_steps,
            }
