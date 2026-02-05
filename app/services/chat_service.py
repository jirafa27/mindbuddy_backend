from typing import Callable, Optional, Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import build_ask_graph
from app.agents.state import AskState
from app.domain.protocols import BlobStorage, EmbeddingProvider, LLMProvider
from app.services.file_service import FileService
from app.services.search_service import SearchService
from app.services.text_chunker import TextChunkerService
from app.utils.file_readers import FileReaderFactory


class ChatService:
    def __init__(
        self,
        file_reader_factory: FileReaderFactory,
        text_chunker: TextChunkerService,
        embedding_service: EmbeddingProvider,
        file_service: FileService,
        llm_service: LLMProvider,
        search_service_factory: Callable[[AsyncSession], SearchService],
        blob_storage: BlobStorage,
    ):
        graph = build_ask_graph(
            file_reader_factory=file_reader_factory,
            text_chunker=text_chunker,
            embedding_service=embedding_service,
            file_service=file_service,
            llm_service=llm_service,
            search_service_factory=search_service_factory,
            blob_storage=blob_storage,
        )
        self.ask_graph = graph.compile()

    async def ask(
        self,
        question: str,
        user_id: int,
        namespace_id: Optional[int] = None,
        file_content: Optional[bytes] = None,
        filename: Optional[str] = None,
        async_db: Optional[AsyncSession] = None,
        file_repository: Any = None,
    ) -> str:
        state: AskState = {
            "question": question,
            "user_id": user_id,
            "namespace_id": namespace_id,
            "file_content": file_content,
            "filename": filename,
        }
        config = {"configurable": {"async_db": async_db, "file_repository": file_repository}}
        result = await self.ask_graph.ainvoke(state, config=config)
        return result.get("answer", "")