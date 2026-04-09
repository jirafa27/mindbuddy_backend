"""SaveFileNode: сохранение файлов по blobs из MinIO."""
import logging
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.state import AskState
from app.domain.protocols import BlobStorage, VectorRepository
from app.services.file_service import FileService

logger = logging.getLogger(__name__)


class SaveFileNode:
    """
    Сохраняет все файлы из state["blobs"]: забирает chunks/embeddings из MinIO,
    сохраняет файлы и эмбеддинги в БД, удаляет blobs.

    Файлы обрабатываются последовательно — они разделяют одну DB-сессию,
    параллельный flush вызывает "Session is already flushing".
    Ответы агрегируются через "\\n", file_ids накапливаются в список.
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

        blobs: List[dict] = state.get("blobs") or []
        if not blobs:
            return {"agent_steps": agent_steps}

        user_id = state.get("user_id")
        namespace_id = state.get("namespace_id")
        search_query = state.get("search_query")

        if user_id is None or file_repository is None:
            return {"agent_steps": agent_steps}

        all_answers: List[str] = []
        all_file_ids: List[int] = []
        all_search_file_ids: List[int] = []
        db_error: Optional[str] = None

        for blob in blobs:
            try:
                result = await self._save_single_blob(
                    blob=blob,
                    namespace_id=namespace_id,
                    user_id=user_id,
                    state=state,
                    async_db=async_db,
                    vector_repository=vector_repository,
                )
            except Exception as e:
                logger.exception("SaveFileNode: failed for file=%s", blob.get("filename"))
                db_error = str(e)
                continue

            if result.get("db_error"):
                db_error = result["db_error"]
                continue

            fid = result.get("file_id")
            if fid is not None:
                all_file_ids.append(fid)
                all_search_file_ids.extend(result.get("search_file_ids") or [fid])

            ans = result.get("answer") or ""
            if ans:
                all_answers.append(ans)

        out: dict[str, Any] = {
            "blobs": None,
            "agent_steps": agent_steps,
        }
        if db_error:
            out["db_error"] = db_error
        if all_file_ids:
            out["file_ids"] = all_file_ids
            out["file_id"] = all_file_ids[0]
            out["search_file_ids"] = all_search_file_ids
        if all_answers:
            combined = "\n".join(all_answers)
            if search_query:
                out["file_save_notice"] = combined
            else:
                out["answer"] = combined

        return out

    async def _save_single_blob(
        self,
        blob: dict,
        *,
        namespace_id: Optional[int],
        user_id: int,
        state: AskState,
        async_db: AsyncSession,
        vector_repository: Optional[VectorRepository],
    ) -> dict:
        """
        Сохраняет один файл по данным из blob-dict.

        Возвращает dict с полями: answer, file_id, search_file_ids, db_error.
        """
        filename = blob.get("filename")
        file_blob_key = blob.get("file_blob_key")
        blob_key = blob.get("blob_key")
        content_hash = blob.get("content_hash")
        content_already_indexed = blob.get("content_already_indexed", False)

        # Полный дубликат, обнаруженный FileAgent ещё до генерации эмбеддингов
        if blob.get("early_duplicate"):
            return {
                "file_id": blob.get("file_id"),
                "answer": blob.get("answer", ""),
                "search_file_ids": blob.get("search_file_ids") or [],
            }

        # Нечитаемый файл — пропускаем тихо
        if blob.get("parse_error"):
            return {}

        if not (file_blob_key and filename):
            return {}

        # Нет ни blob_key ни флага already_indexed — нечего делать
        if not blob_key and not content_already_indexed:
            await self.blob_storage.delete_blob(file_blob_key)
            return {}

        chunks = None
        embeddings = None

        if blob_key:
            bucket = self.blob_storage.blob_bucket_name
            logger.info(
                "SaveFileNode: fetching embeddings blob from MinIO key=%s bucket=%s", blob_key, bucket
            )
            payload = await self.blob_storage.get_blob(blob_key)
            if payload is None:
                await self.blob_storage.delete_blob(file_blob_key)
                raise RuntimeError(f"get_blob returned None for key={blob_key}")
            chunks = payload.get("chunks")
            embeddings = payload.get("embeddings")
            if not chunks or not embeddings:
                await self.blob_storage.delete_blob(file_blob_key)
                return {"db_error": "Invalid blob: missing chunks or embeddings"}

        # Получаем сырые байты файла из MinIO
        file_payload = await self.blob_storage.get_blob(file_blob_key)
        if file_payload is None:
            raise RuntimeError(f"get_blob returned None for file_blob_key={file_blob_key}")
        file_content: bytes = file_payload["raw"]

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

        if not file_created.is_new_file:
            if not file_created.is_new_user_file:
                # user_file уже существует в этом namespace — полный дубликат
                logger.info(
                    "SaveFileNode: full duplicate (user_file_id=%d, content_file_id=%d), skipping",
                    file_created.file_id, file_created.content_file_id,
                )
                if blob_key:
                    await self.blob_storage.delete_blob(blob_key)
                await self.blob_storage.delete_blob(file_blob_key)
                ns_label_str = ns_hint or (str(namespace_id) if namespace_id else None)
                already_msg = (
                    f"Файл «{filename}» уже есть в пространстве «{ns_label_str}»."
                    if ns_label_str else
                    f"Файл «{filename}» уже есть в базе знаний."
                )
                return {
                    "file_id": file_created.file_id,
                    "search_file_ids": [file_created.file_id],
                    "answer": already_msg,
                }

            # Новый user_file, но контент уже существует — эмбеддинги уже есть в БД
            logger.info(
                "SaveFileNode: new UserFile for existing content (user_file_id=%d, content_file_id=%d), skipping embeddings",
                file_created.file_id, file_created.content_file_id,
            )
            if blob_key:
                await self.blob_storage.delete_blob(blob_key)
            await self.blob_storage.delete_blob(file_blob_key)
            if namespace_created and ns_label:
                existing_msg = f"Создал пространство «{ns_label}» и добавил туда файл «{filename}» (файл уже был в базе знаний)."
            elif namespace_id:
                existing_msg = f"Файл «{filename}» уже есть в базе знаний и был добавлен в пространство «{ns_hint or namespace_id}»."
            else:
                existing_msg = f"Файл «{filename}» уже есть в базе знаний."
            return {
                "file_id": file_created.file_id,
                "search_file_ids": [file_created.file_id],
                "answer": existing_msg,
            }

        if vector_repository is None or not chunks or not embeddings:
            await self.blob_storage.delete_blob(file_blob_key)
            return {"db_error": "vector_repository not in config or missing chunks/embeddings"}

        await vector_repository.create_batch(
            file_id=file_created.content_file_id,
            chunks=chunks,
            embeddings=embeddings,
            namespace_id=namespace_id,
        )
        await async_db.commit()
        if blob_key:
            await self.blob_storage.delete_blob(blob_key)
        await self.blob_storage.delete_blob(file_blob_key)

        if namespace_created and ns_label:
            saved_msg = f"Создал пространство «{ns_label}» и добавил туда файл «{filename}»."
        else:
            ns_part = ""
            if namespace_created and ns_label:
                ns_part = f" в новое пространство «{ns_label}»"
            elif ns_label:
                ns_part = f" в пространство «{ns_label}»"
            saved_msg = f"Файл «{filename}» сохранён{ns_part}."

        return {
            "file_id": file_created.file_id,
            "search_file_ids": [file_created.file_id],
            "answer": saved_msg,
        }
