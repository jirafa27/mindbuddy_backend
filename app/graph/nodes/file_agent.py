import hashlib
import logging
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig

from app.graph.state import AskState, AttachedFile
from app.domain.protocols import BlobStorage, EmbeddingProvider
from app.services.text_chunker import TextChunkerService
from app.utils.file_readers import FileReaderFactory
from app.utils.file import decode_filename

logger = logging.getLogger(__name__)


class FileAgent:
    """
    Парсит файлы, разбивает на чанки, генерирует эмбеддинги; сохраняет данные в BlobStorage.

    Читает state["attached_file"] как список (file_blob_key, filename).
    Байты скачиваются из MinIO по ключу — в state не хранятся.
    Для каждого файла создаёт blob_key с чанками/эмбеддингами.
    file_blob_key из attached_files повторно используется SaveFileNode.
    """

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
        Парсит все файлы запроса, разбивает на чанки, генерирует эмбеддинги.
        Не сохраняет в БД — это делает SaveFileNode.

        Возвращает blobs: List[FileSaveBlob] — по одному элементу на каждый файл.
        Каждый blob содержит blob_key (чанки/эмбеддинги) и file_blob_key (ключ сырых байтов в MinIO).
        Байты через state не передаются.
        """
        agent_steps = list(state.get("agent_steps") or [])

        all_files: List[AttachedFile] = list(state.get("attached_files") or [])

        if not all_files:
            return {"agent_steps": agent_steps + ["FileAgent"]}

        configurable = config.get("configurable") or {}
        file_repository = configurable.get("file_repository")
        user_file_repository = configurable.get("user_file_repository")
        user_id = state.get("user_id")
        namespace_id = state.get("namespace_id")
        namespace_name_hint = state.get("namespace_name_hint")

        blobs: List[dict] = []
        for file_entry in all_files:
            blob = await self._process_file(
                file_blob_key=file_entry["file_blob_key"],
                filename=file_entry["filename"],
                user_id=user_id,
                namespace_id=namespace_id,
                namespace_name_hint=namespace_name_hint,
                file_repository=file_repository,
                user_file_repository=user_file_repository,
            )
            blobs.append(blob)

        return {
            "blobs": blobs,
            "agent_steps": agent_steps + ["FileAgent"],
        }

    async def _process_file(
        self,
        *,
        file_blob_key: str,
        filename: str,
        user_id: Optional[int],
        namespace_id: Optional[int],
        namespace_name_hint: Optional[str],
        file_repository,
        user_file_repository,
    ) -> dict:
        """
        Обрабатывает один файл. Возвращает FileSaveBlob для SaveFileNode:

        - early_duplicate=True  → полный дубликат (file_id и answer уже заполнены)
        - content_already_indexed=True → контент есть в БД, нужен только UserFile
        - blob_key заполнен → нормальный путь (чанки/эмбеддинги в MinIO)
        - parse_error=True → файл нечитаем, SaveFileNode его пропустит
        """
        filename = decode_filename(filename)
        file_ext = filename.split(".")[-1].lower() if "." in filename else "txt"

        # Скачиваем байты из MinIO
        file_payload = await self.blob_storage.get_blob(file_blob_key)
        if file_payload is None:
            logger.error("FileAgent: blob not found for key=%s", file_blob_key)
            return {
                "filename": filename, "content_hash": None,
                "blob_key": None, "file_blob_key": file_blob_key,
                "content_already_indexed": False, "early_duplicate": False, "parse_error": True,
            }
        file_content: bytes = file_payload["raw"]

        content_hash = hashlib.sha256(file_content).hexdigest()
        base = {"filename": filename, "content_hash": content_hash, "file_blob_key": file_blob_key}

        # --- Ранняя проверка дубликата (до парсинга и эмбеддингов) ---
        if file_repository is not None and user_file_repository is not None and user_id is not None:
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
                    # file_blob_key больше не нужен — SaveFileNode его не будет вызывать
                    await self.blob_storage.delete_blob(file_blob_key)
                    return {
                        "filename": filename, "content_hash": content_hash,
                        "blob_key": None, "file_blob_key": None,
                        "content_already_indexed": False,
                        "early_duplicate": True,
                        "file_id": existing_user_file.id,
                        "search_file_ids": [existing_user_file.id],
                    }
                else:
                    # Контент уже проиндексирован, но UserFile для этого namespace нет.
                    # SaveFileNode создаст только UserFile — байты нужны для upload_file.
                    logger.info(
                        "FileAgent: content already indexed (content_file_id=%d), skipping embedding generation",
                        existing_content.id,
                    )
                    return {
                        **base,
                        "blob_key": None,
                        "content_already_indexed": True,
                        "early_duplicate": False,
                    }
        # --- Конец ранней проверки ---

        try:
            reader = self.file_reader_factory.get_reader(file_ext)
            text = reader.read(file_content)
        except Exception:
            logger.warning("FileAgent: failed to parse file=%s", filename)
            return {**base, "blob_key": None,
                    "content_already_indexed": False, "early_duplicate": False, "parse_error": True}

        if not text or not text.strip():
            return {**base, "blob_key": None,
                    "content_already_indexed": False, "early_duplicate": False, "parse_error": True}

        chunks = self.text_chunker.chunk_text(text, filename=filename)
        if not chunks:
            return {**base, "blob_key": None,
                    "content_already_indexed": False, "early_duplicate": False, "parse_error": True}

        embeddings = await self.embedding_service.generate_embeddings_batch(chunks)

        bucket = self.blob_storage.blob_bucket_name
        logger.info(
            "FileAgent: uploading embeddings blob for file=%s to bucket=%s",
            filename, bucket,
        )
        blob_key = await self.blob_storage.put_blob({"chunks": chunks, "embeddings": embeddings})
        logger.info(
            "FileAgent: saved embeddings blob=%s (file_blob_key=%s)", blob_key, file_blob_key
        )

        return {
            **base,
            "blob_key": blob_key,
            "content_already_indexed": False,
            "early_duplicate": False,
        }
