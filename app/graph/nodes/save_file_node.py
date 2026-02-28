"""SaveFileNode: сохранение файла и эмбеддингов по blob_key из MinIO."""
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.state import AskState
from app.domain.protocols import BlobStorage, VectorRepository
from app.services.file_service import FileService

logger = logging.getLogger(__name__)


class SaveFileNode:
    """
    Сохраняет файл по blob_key: забирает chunks/embeddings из MinIO,
    сохраняет файл и эмбеддинги в БД, удаляет blob.
    """

    def __init__(
        self,
        *,
        file_service: FileService,
        blob_storage: BlobStorage,
    ):
        self.file_service = file_service
        self.blob_storage = blob_storage

    async def run(self, state: AskState, config: RunnableConfig) -> dict[str, Any]:
        configurable = config.get("configurable") or {}
        async_db: AsyncSession | None = configurable.get("async_db")
        file_repository = configurable.get("file_repository")
        vector_repository: VectorRepository | None = configurable.get("vector_repository")

        agent_steps = list(state.get("agent_steps") or []) + ["SaveFileNode"]
        if not async_db:
            return {"agent_steps": agent_steps}

        blob_key = state.get("blob_key")
        file_content = state.get("file_content")
        filename = state.get("filename")
        namespace_id = state.get("namespace_id")
        user_id = state.get("user_id")

        if not (blob_key and file_content and filename and user_id is not None and file_repository is not None):
            return {"agent_steps": agent_steps}

        try:
            bucket = self.blob_storage.blob_bucket_name
            logger.info("SaveFileNode: fetching blob from MinIO key=%s bucket=%s", blob_key, bucket)

            payload = await self.blob_storage.get_blob(blob_key)
            if payload is None:
                raise RuntimeError(f"get_blob returned None for key={blob_key}")
            chunks = payload.get("chunks")
            embeddings = payload.get("embeddings")
            if not chunks or not embeddings:
                return {
                    "db_error": "Invalid blob: missing chunks or embeddings",
                    "agent_steps": agent_steps,
                }

            file_created = await self.file_service.upload_file(
                file_content=file_content,
                filename=filename,
                namespace_id=namespace_id,
                user_id=user_id,
            )

            if vector_repository is None:
                return {
                    "db_error": "vector_repository not in config",
                    "agent_steps": agent_steps,
                }

            await vector_repository.create_batch(
                file_id=file_created.content_file_id,
                chunks=chunks,
                embeddings=embeddings,
                namespace_id=namespace_id,
            )
            if async_db:
                await async_db.commit()
            await self.blob_storage.delete_blob(blob_key)

            return {
                "file_id": file_created.file_id,
                "blob_key": None,
                "agent_steps": agent_steps,
            }
        except Exception as e:
            logger.exception("SaveFileNode failed")
            return {
                "db_error": str(e),
                "agent_steps": agent_steps,
            }
