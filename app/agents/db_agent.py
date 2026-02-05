"""DBAgent: сохранение файла и эмбеддингов; выполнение векторного поиска по SQL."""
import asyncio
import logging
from typing import Any, Callable, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.state import AskState
from app.domain.protocols import BlobStorage, VectorRepository
from app.infrastructure.db.sync_session import get_sync_db
from app.infrastructure.repositories import PgVectorRepository
from app.services.file_service import FileService
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)


class DBAgent:
    """
    Два режима:
    1) Save: по blob_key забирает chunks/embeddings из MinIO, сохраняет файл в MinIO + File в БД + эмбеддинги, удаляет blob.
    2) Execute: выполнение SQL от SQLAgent с параметрами; при ошибке — db_error, иначе search_result.
    """

    def __init__(
        self,
        *,
        file_service: FileService,
        search_service_factory: Callable[[AsyncSession], SearchService],
        blob_storage: BlobStorage,
    ):
        self.file_service = file_service
        self.search_service_factory = search_service_factory
        self.blob_storage = blob_storage

    @staticmethod
    def _sync_create_batch(
        file_id: int,
        namespace_id: Optional[int],
        chunks: List[str],
        embeddings: List[List[float]],
    ) -> None:
        """Синхронная вставка эмбеддингов (вызывается из потока)."""
        with get_sync_db() as db:
            repo: VectorRepository = PgVectorRepository(db)
            repo.create_batch(
                file_id=file_id,
                chunks=chunks,
                embeddings=embeddings,
                namespace_id=namespace_id,
            )

    async def run(self, state: AskState, config: dict | None = None) -> dict[str, Any]:
        configurable = (config or {}).get("configurable") or {}
        async_db = configurable.get("async_db")
        file_repository = configurable.get("file_repository")
        if not async_db:
            return {"agent_steps": list(state.get("agent_steps") or []) + ["DBAgent"]}

        agent_steps = list(state.get("agent_steps") or [])
        namespace_id = state.get("namespace_id")
        user_id = state.get("user_id")

        # --- Save: файл + эмбеддинги (данные из MinIO по blob_key) ---
        blob_key = state.get("blob_key")
        file_content = state.get("file_content")
        filename = state.get("filename")

        if blob_key and file_content and filename and user_id is not None and file_repository is not None:
            try:
                bucket = self.blob_storage.blob_bucket_name
                logger.info("DBAgent: fetching blob from MinIO key=%s bucket=%s", blob_key, bucket)

                # get_blob уже имеет встроенный retry (5 попыток с интервалом 1с)
                payload = await self.blob_storage.get_blob(blob_key)
                if payload is None:
                    raise RuntimeError(f"get_blob returned None for key={blob_key}")
                chunks = payload.get("chunks")
                embeddings = payload.get("embeddings")
                if not chunks or not embeddings:
                    return {
                        "db_error": "Invalid blob: missing chunks or embeddings",
                        "agent_steps": agent_steps + ["DBAgent"],
                    }
                file_created = await self.file_service.upload_file(
                    file_content=file_content,
                    filename=filename,
                    namespace_id=namespace_id,
                    user_id=user_id,
                    db=async_db,
                    file_repository=file_repository,
                )
                await async_db.commit()
                await asyncio.to_thread(
                    self._sync_create_batch,
                    file_created.file_id,
                    namespace_id,
                    chunks,
                    embeddings,
                )
                await self.blob_storage.delete_blob(blob_key)
                return {
                    "file_id": file_created.file_id,
                    "blob_key": None,  # Очищаем, чтобы при повторном вызове DBAgent не пытался снова загрузить
                    "agent_steps": agent_steps + ["DBAgent"],
                }
            except Exception as e:
                logger.exception("DBAgent save failed")
                return {
                    "db_error": str(e),
                    "agent_steps": agent_steps + ["DBAgent"],
                }

        # --- Execute: SQL от SQLAgent (семантический поиск или запрос о структуре) ---
        sql_query = state.get("sql_query")
        query_embedding = state.get("query_embedding")  # Может быть None для структурных запросов
        limit = 5

        if not sql_query or user_id is None:
            return {"agent_steps": agent_steps + ["DBAgent"]}

        try:
            search_service = self.search_service_factory(async_db)
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
                "agent_steps": agent_steps + ["DBAgent"],
            }
        except Exception as e:
            logger.warning("DBAgent execute failed: %s", e)
            return {
                "search_result": [],
                "db_error": str(e),
                "agent_steps": agent_steps + ["DBAgent"],
            }
