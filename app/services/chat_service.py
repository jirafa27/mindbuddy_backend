from typing import Optional, List

from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.graph import build_ask_graph
from app.graph.state import AskState
from app.graph.schemas import AskResponse, SourceItem
from app.domain.protocols import (
    BlobStorage,
    EmbeddingProvider,
    TaskPublisher,
    FileRepository,
    LLMProvider,
    VectorRepository,
)
from app.services.file_service import FileService
from app.services.search_service import SearchService
from app.services.text_chunker import TextChunkerService
from app.services.content_extractor import ContentExtractorService
from app.services.summary_service import SummaryService
from app.utils.file_readers import FileReaderFactory


class ChatService:
    def __init__(
        self,
        *,
        db: AsyncSession,
        file_repository: FileRepository,
        vector_repository: VectorRepository,
        search_service: SearchService,
        summary_service: SummaryService,
        summary_agent,
        file_reader_factory: FileReaderFactory,
        text_chunker: TextChunkerService,
        embedding_service: EmbeddingProvider,
        file_service: FileService,
        llm_service: LLMProvider,
        blob_storage: BlobStorage,
        intent_classifier,
        content_extractor: Optional[ContentExtractorService] = None,
        task_publisher: Optional[TaskPublisher] = None,
    ):
        self.db = db
        self.file_repository = file_repository
        self.vector_repository = vector_repository
        self.search_service = search_service
        self.summary_service = summary_service
        self.summary_agent = summary_agent
        self.file_service = file_service
        self.content_extractor = content_extractor

        graph = build_ask_graph(
            file_reader_factory=file_reader_factory,
            text_chunker=text_chunker,
            embedding_service=embedding_service,
            file_service=file_service,
            llm_service=llm_service,
            blob_storage=blob_storage,
            intent_classifier=intent_classifier,
            content_extractor=content_extractor,
            task_publisher=task_publisher,
        )
        self.ask_graph = graph.compile()

    async def ask(
        self,
        question: str,
        user_id: int,
        namespace_id: Optional[int] = None,
        file_content: Optional[bytes] = None,
        filename: Optional[str] = None,
        file_id: Optional[int] = None,
        history: Optional[List[dict]] = None,
        override_intent: Optional[str] = None,
    ) -> AskResponse:
        """

        """
        state: AskState = {
            "question": question,
            "user_id": user_id,
            "namespace_id": namespace_id,
            "file_content": file_content,
            "filename": filename,
            "history": history or [],
            "override_intent": override_intent,
        }
        if file_id is not None:
            state["history_file_id"] = file_id
        config = {
            "configurable": {
                "async_db": self.db,
                "file_repository": self.file_repository,
                "vector_repository": self.vector_repository,
                "search_service": self.search_service,
                "summary_service": self.summary_service,
                "summary_agent": self.summary_agent,
                "file_service": self.file_service,
                "content_extractor": self.content_extractor,
            }
        }
        result = await self.ask_graph.ainvoke(state, config=config)
        
        sources_raw: List[dict] = result.get("sources") or []
        sources = [
            SourceItem(filename=s.get("filename", "?"), relevance=s.get("relevance", 0.0))
            for s in sources_raw
        ]
        
        return AskResponse(
            answer=result.get("answer", ""),
            sources=sources,
            agent_steps=result.get("agent_steps") or [],
            file_id=result.get("file_id"),
        )