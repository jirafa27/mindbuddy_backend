"""Сборка LangGraph StateGraph для RAG-пайплайна POST /ask."""
from typing import Any, Callable

from langgraph.graph import StateGraph, START, END
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.state import AskState
from app.agents.file_agent import FileAgent
from app.agents.db_agent import DBAgent
from app.agents.sql_agent import SQLAgent
from app.agents.mind_buddy_agent import MindBuddyAgent
from app.agents.query_embedding_node import QueryEmbeddingNode
from app.domain.protocols import BlobStorage, EmbeddingProvider
from app.services.file_service import FileService
from app.services.text_chunker import TextChunkerService
from app.utils.file_readers import FileReaderFactory
from app.domain.protocols import LLMProvider


MAX_SQL_RETRIES = 2


def _route_start(state: AskState) -> str:
    """Старт: если есть файл — FileAgent, иначе — вычисление query_embedding."""
    if state.get("file_content") and state.get("filename"):
        return "file_agent"
    return "compute_query_embedding"


def _route_after_db(state: AskState) -> str:
    """
    После DBAgent: если только что сохранили файл — compute_query_embedding;
    если ошибка SQL и retry < N — sql_agent; иначе — mind_buddy_agent.
    """
    if state.get("file_id") is not None and state.get("search_result") is None and not state.get("db_error"):
        return "compute_query_embedding"
    if state.get("db_error") and (state.get("retry_count") or 0) < MAX_SQL_RETRIES:
        return "sql_agent"
    return "mind_buddy_agent"


def build_ask_graph(
    *,
    file_reader_factory: FileReaderFactory,
    text_chunker: TextChunkerService,
    embedding_service: EmbeddingProvider,
    file_service: FileService,
    llm_service: LLMProvider,
    search_service_factory: Callable[[AsyncSession], Any],
    blob_storage: BlobStorage,
) -> StateGraph:
    """Собирает граф для /ask. Перед использованием вызвать .compile()."""
    file_agent = FileAgent(
        file_reader_factory=file_reader_factory,
        text_chunker=text_chunker,
        embedding_service=embedding_service,
        blob_storage=blob_storage,
    )
    db_agent = DBAgent(
        file_service=file_service,
        search_service_factory=search_service_factory,
        blob_storage=blob_storage,
    )
    sql_agent = SQLAgent(llm_service=llm_service)
    mind_buddy_agent = MindBuddyAgent(llm_service=llm_service)
    query_embedding_node = QueryEmbeddingNode(embedding_service=embedding_service)

    graph = StateGraph(AskState)

    graph.add_node("file_agent", file_agent.run)
    graph.add_node("db_agent", db_agent.run)
    graph.add_node("compute_query_embedding", query_embedding_node.run)
    graph.add_node("sql_agent", sql_agent.run)
    graph.add_node("mind_buddy_agent", mind_buddy_agent.run)

    graph.add_conditional_edges(START, _route_start, {"file_agent": "file_agent", "compute_query_embedding": "compute_query_embedding"})
    graph.add_edge("file_agent", "db_agent")
    graph.add_conditional_edges("db_agent", _route_after_db, {
        "compute_query_embedding": "compute_query_embedding",
        "sql_agent": "sql_agent",
        "mind_buddy_agent": "mind_buddy_agent",
    })
    graph.add_edge("compute_query_embedding", "sql_agent")
    graph.add_edge("sql_agent", "db_agent")
    graph.add_edge("mind_buddy_agent", END)

    return graph
