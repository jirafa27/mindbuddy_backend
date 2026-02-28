"""Сборка LangGraph StateGraph для RAG-пайплайна POST /ask."""
from typing import Optional

from langgraph.graph import StateGraph, START, END

from app.graph.state import AskState
from app.graph.nodes.file_agent import FileAgent
from app.graph.nodes.save_file_node import SaveFileNode
from app.graph.nodes.execute_search_node import ExecuteSearchNode
from app.graph.nodes.sql_agent import SQLAgent
from app.graph.nodes.mind_buddy_agent import MindBuddyAgent
from app.graph.nodes.query_embedding_node import QueryEmbeddingNode
from app.graph.nodes.router_node import RouterNode
from app.graph.nodes.summary_node import SummaryNode
from app.graph.nodes.index_url_node import IndexUrlNode
from app.domain.protocols import BlobStorage, EmbeddingProvider, TaskPublisher
from app.services.file_service import FileService
from app.services.text_chunker import TextChunkerService
from app.services.content_extractor import ContentExtractorService
from app.utils.file_readers import FileReaderFactory
from app.domain.protocols import LLMProvider
from app.core.enums import IntentType


MAX_SQL_RETRIES = 2


def _route_after_router(state: AskState) -> str:
    """
    После RouterNode: маршрутизация по намерению.
    
    Интенты:
    - save_file: сохранение файла -> FileAgent
    - summarize: суммаризация -> SummaryNode
    - index_url: сохранение URL -> IndexUrlNode
    - rag_query: вопрос по базе -> compute_query_embedding
    
    Если RouterNode уже установил answer (ошибка валидации) — идём в END через mind_buddy.
    """
    intent = state.get("intent")
    
    if state.get("answer"):
        return "mind_buddy_agent"
    
    if intent == IntentType.SAVE_FILE:
        return "file_agent"
    elif intent == IntentType.SUMMARIZE:
        return "summary_node"
    elif intent == IntentType.INDEX_URL:
        return "index_url_node"
    else:  # IntentType.RAG_QUERY
        return "compute_query_embedding"


def _route_after_save_file(state: AskState) -> str:
    """
    После SaveFileNode: если файл сохранён — идём в compute_query_embedding;
    иначе — в mind_buddy_agent (ошибка или нечего искать).
    """
    if state.get("file_id") is not None and not state.get("db_error"):
        return "compute_query_embedding"
    return "mind_buddy_agent"


def _route_after_search(state: AskState) -> str:
    """
    После ExecuteSearchNode: при ошибке SQL и retry < N — повтор в sql_agent;
    иначе — mind_buddy_agent.
    """
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
    blob_storage: BlobStorage,
    intent_classifier,
    content_extractor: Optional[ContentExtractorService] = None,
    task_publisher: Optional[TaskPublisher] = None,
) -> StateGraph:
    """
    Собирает граф для /ask. Перед использованием вызвать .compile().
    
    Структура графа:
    START -> RouterNode -> [по намерению]:
        - file_upload -> FileAgent -> SaveFileNode -> compute_query_embedding | mind_buddy
        - summarize_url -> SummaryNode -> END
        - index_url -> IndexUrlNode -> END
        - rag_query -> compute_query_embedding -> SQLAgent -> ExecuteSearchNode -> sql_agent | mind_buddy -> END
    """
    router_node = RouterNode(intent_classifier=intent_classifier)
    file_agent = FileAgent(
        file_reader_factory=file_reader_factory,
        text_chunker=text_chunker,
        embedding_service=embedding_service,
        blob_storage=blob_storage,
    )
    save_file_node = SaveFileNode(file_service=file_service, blob_storage=blob_storage)
    execute_search_node = ExecuteSearchNode()
    sql_agent = SQLAgent(llm_service=llm_service)
    mind_buddy_agent = MindBuddyAgent(llm_service=llm_service)
    query_embedding_node = QueryEmbeddingNode(embedding_service=embedding_service)

    graph = StateGraph(AskState)

    graph.add_node("router", router_node.run)
    graph.add_node("file_agent", file_agent.run)
    graph.add_node("save_file_node", save_file_node.run)
    graph.add_node("execute_search_node", execute_search_node.run)
    graph.add_node("compute_query_embedding", query_embedding_node.run)
    graph.add_node("sql_agent", sql_agent.run)
    graph.add_node("mind_buddy_agent", mind_buddy_agent.run)

    # Опциональные ноды для суммаризации и индексации URL
    summary_node = SummaryNode()
    graph.add_node("summary_node", summary_node.run)

    if content_extractor and task_publisher:
        index_url_node = IndexUrlNode(
            content_extractor=content_extractor,
            file_service=file_service,
            task_publisher=task_publisher,
        )
        graph.add_node("index_url_node", index_url_node.run)

    # Рёбра графа
    graph.add_edge(START, "router")
    
    # Условная маршрутизация после роутера
    routing_map = {
        "file_agent": "file_agent",
        "summary_node": "summary_node",
        "compute_query_embedding": "compute_query_embedding",
        "mind_buddy_agent": "mind_buddy_agent",  # Для случаев когда RouterNode уже сформировал answer
    }
    if content_extractor and task_publisher:
        routing_map["index_url_node"] = "index_url_node"

    graph.add_conditional_edges("router", _route_after_router, routing_map)

    graph.add_edge("file_agent", "save_file_node")
    graph.add_conditional_edges("save_file_node", _route_after_save_file, {
        "compute_query_embedding": "compute_query_embedding",
        "mind_buddy_agent": "mind_buddy_agent",
    })

    graph.add_edge("compute_query_embedding", "sql_agent")
    graph.add_edge("sql_agent", "execute_search_node")
    graph.add_conditional_edges("execute_search_node", _route_after_search, {
        "sql_agent": "sql_agent",
        "mind_buddy_agent": "mind_buddy_agent",
    })
    graph.add_edge("mind_buddy_agent", END)
    
    # Суммаризация и индексация -> END
    graph.add_edge("summary_node", END)
    if content_extractor and task_publisher:
        graph.add_edge("index_url_node", END)

    return graph
