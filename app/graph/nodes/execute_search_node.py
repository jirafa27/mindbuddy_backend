"""ExecuteSearchNode: выполнение векторного поиска по SQL от SQLAgent."""
import logging
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from app.graph.state import AskState
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)


class ExecuteSearchNode:
    """Выполняет SQL-поиск с эмбеддингами; при ошибке возвращает db_error. SearchService берётся из config."""

    async def run(self, state: AskState, config: RunnableConfig) -> dict[str, Any]:
        configurable = config.get("configurable") or {}
        search_service: Optional[SearchService] = configurable.get("search_service")

        agent_steps = list(state.get("agent_steps") or []) + ["ExecuteSearchNode"]

        sql_query = state.get("sql_query")
        user_id = state.get("user_id")
        namespace_id = state.get("namespace_id")
        query_embedding = state.get("query_embedding")
        limit = 5

        if not sql_query or user_id is None:
            return {"agent_steps": agent_steps}

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
