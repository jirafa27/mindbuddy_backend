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
        search_query = state.get("search_query")
        content_already_indexed = state.get("content_already_indexed", False)

        if not (file_content and filename and user_id is not None and file_repository is not None):
            return {"agent_steps": agent_steps}

        # Если blob_key нет и контент не проиндексирован — нечего делать
        if not blob_key and not content_already_indexed:
            return {"agent_steps": agent_steps}

        try:
            chunks = None
            embeddings = None

            if blob_key:
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

            content_hash = self.file_service.compute_content_hash(file_content)
            file_created = await self.file_service.upload_file(
                file_content=file_content,
                filename=filename,
                namespace_id=namespace_id,
                user_id=user_id,
                content_hash=content_hash,
            )

            ns_hint = state.get("namespace_name_hint")
            namespace_created = state.get("namespace_created", False)
            ns_label = ns_hint or namespace_id
            if namespace_created and ns_label:
                ns_part = f" в новое пространство «{ns_label}»"
            elif ns_label:
                ns_part = f" в пространство «{ns_label}»"
            else:
                ns_part = ""

            if not file_created.is_new_file:
                if not file_created.is_new_user_file:
                    # user_file уже существует в этом namespace — полный дубликат
                    logger.info(
                        "SaveFileNode: full duplicate (user_file_id=%d, content_file_id=%d), skipping",
                        file_created.file_id, file_created.content_file_id,
                    )
                    if blob_key:
                        await self.blob_storage.delete_blob(blob_key)
                    ns_label_str = ns_hint or (str(namespace_id) if namespace_id else None)
                    already_msg = (
                        f"Файл «{filename}» уже есть в пространстве «{ns_label_str}»."
                        if ns_label_str else
                        f"Файл «{filename}» уже есть в базе знаний."
                    )
                    result: dict = {
                        "file_id": file_created.file_id,
                        "search_file_ids": [file_created.file_id],
                        "blob_key": None,
                        "agent_steps": agent_steps,
                    }
                    if search_query:
                        result["file_save_notice"] = already_msg
                    else:
                        result["answer"] = already_msg
                    return result
                # Новый user_file, но контент уже существует — эмбеддинги уже есть в БД
                logger.info(
                    "SaveFileNode: new UserFile for existing content (user_file_id=%d, content_file_id=%d), skipping embeddings (already indexed)",
                    file_created.file_id, file_created.content_file_id,
                )
                if blob_key:
                    await self.blob_storage.delete_blob(blob_key)
                if namespace_created and ns_label:
                    existing_msg = f"Создал пространство «{ns_label}» и добавил туда файл «{filename}» (файл уже был в базе знаний)."
                elif namespace_id:
                    existing_msg = f"Файл «{filename}» уже есть в базе знаний и был добавлен в пространство «{ns_hint or namespace_id}»."
                else:
                    existing_msg = f"Файл «{filename}» уже есть в базе знаний."
                result2: dict = {
                    "file_id": file_created.file_id,
                    "search_file_ids": [file_created.file_id],
                    "blob_key": None,
                    "agent_steps": agent_steps,
                }
                if search_query:
                    result2["file_save_notice"] = existing_msg
                else:
                    result2["answer"] = existing_msg
                return result2

            if vector_repository is None or not chunks or not embeddings:
                return {
                    "db_error": "vector_repository not in config or missing chunks/embeddings",
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
            if blob_key:
                await self.blob_storage.delete_blob(blob_key)

            if namespace_created and ns_label:
                saved_msg = f"Создал пространство «{ns_label}» и добавил туда файл «{filename}»."
            else:
                saved_msg = f"Файл «{filename}» сохранён{ns_part}."
            result3: dict = {
                "file_id": file_created.file_id,
                "search_file_ids": [file_created.file_id],
                "blob_key": None,
                "agent_steps": agent_steps,
            }
            if search_query:
                result3["file_save_notice"] = saved_msg
            else:
                result3["answer"] = saved_msg
            return result3
        except Exception as e:
            logger.exception("SaveFileNode failed")
            return {
                "db_error": str(e),
                "agent_steps": agent_steps,
            }
