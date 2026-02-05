"""FileAgent: парсинг файла, чанки, эмбеддинги; сохранение в Storage (Claim Check)."""
import logging
from typing import Any

from app.agents.state import AskState
from app.domain.protocols import BlobStorage, EmbeddingProvider
from app.services.text_chunker import TextChunkerService
from app.utils.file_readers import FileReaderFactory
from app.utils.file import decode_filename

logger = logging.getLogger(__name__)


class FileAgent:
    """Парсит файл, разбивает на чанки, генерирует эмбеддинги; сохраняет данные в BlobStorage (Claim Check)."""

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

    async def run(self, state: AskState, config: dict | None = None) -> dict[str, Any]:
        """
        Парсит файл, разбивает на чанки, генерирует эмбеддинги.
        Не сохраняет в БД — это делает DBAgent.
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
