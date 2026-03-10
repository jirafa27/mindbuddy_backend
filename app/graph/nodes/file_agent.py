import hashlib
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.graph.state import AskState
from app.domain.protocols import BlobStorage, EmbeddingProvider
from app.services.text_chunker import TextChunkerService
from app.utils.file_readers import FileReaderFactory
from app.utils.file import decode_filename

logger = logging.getLogger(__name__)


class FileAgent:
    """Парсит файл, разбивает на чанки, генерирует эмбеддинги; сохраняет данные в BlobStorage"""

    def __init__(
        self,
        *,
        file_reader_factory: FileReaderFactory,
        text_chunker: TextChunkerService,
        embedding_service: EmbeddingProvider,
        blob_storage: BlobStorage,
    ):
        self.file_reader_factory = file_reader_factory
        self.text_chunker = text_chunker
        self.embedding_service = embedding_service
        self.blob_storage = blob_storage

    async def run(self, state: AskState, config: RunnableConfig) -> dict[str, Any]:
        """
        Парсит файл, разбивает на чанки, генерирует эмбеддинги.
        Не сохраняет в БД — это делает SaveFileNode.

        До любых тяжёлых операций проверяет, не является ли файл полным дубликатом
        (тот же контент уже загружен этим пользователем в этот namespace). Если дубликат —
        возвращает готовый ответ без единого вызова embedding-модели.
        """
        file_content = state.get("file_content")
        filename = state.get("filename")
        agent_steps = list(state.get("agent_steps") or [])

        if not file_content or not filename:
            return {"agent_steps": agent_steps + ["FileAgent"]}

        filename = decode_filename(filename)
        if "." in filename:
            file_ext = filename.split(".")[-1].lower()
        else:
            file_ext = "txt"

        # --- Ранняя проверка дубликата (до парсинга и эмбеддингов) ---
        configurable = config.get("configurable") or {}
        file_repository = configurable.get("file_repository")
        user_file_repository = configurable.get("user_file_repository")
        user_id = state.get("user_id")
        namespace_id = state.get("namespace_id")
        search_query = state.get("search_query")
        namespace_name_hint = state.get("namespace_name_hint")

        if file_repository is not None and user_file_repository is not None and user_id is not None:
            content_hash = hashlib.sha256(file_content).hexdigest()
            existing_content = await file_repository.get_by_content_hash(content_hash)
            if existing_content is not None:
                existing_user_file = await user_file_repository.find_by_user_and_file(
                    user_id, existing_content.id, namespace_id
                )
                if existing_user_file is not None:
                    logger.info(
                        "FileAgent: full duplicate detected early (user_file_id=%d, content_file_id=%d), skipping embeddings",
                        existing_user_file.id,
                        existing_content.id,
                    )
                    already_msg = (
                        f"Файл «{filename}» уже есть в пространстве «{namespace_name_hint or namespace_id}»."
                        if namespace_id
                        else f"Файл «{filename}» уже есть в базе знаний."
                    )
                    result: dict = {
                        "file_id": existing_user_file.id,
                        "search_file_ids": [existing_user_file.id],
                        "agent_steps": agent_steps + ["FileAgent"],
                    }
                    if search_query:
                        result["file_save_notice"] = already_msg
                    else:
                        result["answer"] = already_msg
                    return result
                else:
                    # Контент уже проиндексирован (эмбеддинги есть), но UserFile для этого namespace нет.
                    # Пропускаем генерацию эмбеддингов — SaveFileNode создаст только UserFile.
                    logger.info(
                        "FileAgent: content already indexed (content_file_id=%d), skipping embedding generation",
                        existing_content.id,
                    )
                    return {
                        "filename": filename,
                        "content_already_indexed": True,
                        "agent_steps": agent_steps + ["FileAgent"],
                    }
        # --- Конец ранней проверки ---

        try:
            reader = self.file_reader_factory.get_reader(file_ext)
            text = reader.read(file_content)
        except Exception:
            return {"agent_steps": agent_steps + ["FileAgent"]}

        if not text or not text.strip():
            return {"agent_steps": agent_steps + ["FileAgent"]}

        chunks = self.text_chunker.chunk_text(text, filename=filename)
        if not chunks:
            return {"agent_steps": agent_steps + ["FileAgent"]}

        embeddings = await self.embedding_service.generate_embeddings_batch(chunks)

        bucket = self.blob_storage.blob_bucket_name
        logger.info("FileAgent: uploading blob with %d chunks for file=%s to bucket=%s", len(chunks), filename, bucket)
        blob_key = await self.blob_storage.put_blob({"chunks": chunks, "embeddings": embeddings})
        logger.info("FileAgent: saved blob to MinIO key=%s bucket=%s", blob_key, bucket)

        return {
            "blob_key": blob_key,
            "filename": filename,
            "agent_steps": agent_steps + ["FileAgent"],
        }
