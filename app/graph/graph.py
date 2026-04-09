"""Сборка LangGraph StateGraph для RAG-пайплайна POST /ask."""
from typing import Optional

from langgraph.graph import StateGraph, START, END

from app.graph.state import AskState
from app.graph.nodes.crud_node import CrudNode
from app.graph.nodes.file_agent import FileAgent
from app.graph.nodes.save_file_node import SaveFileNode
from app.graph.nodes.execute_search_node import ExecuteSearchNode
from app.graph.nodes.sql_agent import SQLAgent
from app.graph.nodes.mind_buddy_agent import MindBuddyAgent
from app.graph.nodes.query_embedding_node import QueryEmbeddingNode
from app.graph.nodes.router_node import RouterNode
from app.graph.nodes.intent_node import IntentNode
from app.graph.nodes.action_resolver_node import ActionResolverNode
from app.graph.nodes.summary_node import SummaryNode
from app.graph.nodes.index_url_node import IndexUrlNode
from app.graph.nodes.send_file_node import SendFileNode
from app.graph.nodes.multi_action_node import MultiActionNode
from app.graph.nodes.generate_content_node import GenerateContentNode
from app.domain.protocols import BlobStorage, EmbeddingProvider, TaskPublisher
from app.services.file_service import FileService
from app.services.namespace_service import NamespaceService
from app.services.text_chunker import TextChunkerService
from app.services.content_extractor import ContentExtractorService
from app.services.intent_classifier import IntentClassifierService
from app.utils.file_readers import FileReaderFactory
from app.domain.protocols import LLMProvider
from app.core.enums import IntentType


MAX_SQL_RETRIES = 2


def _route_after_router(state: AskState) -> str:
    """
    После RouterNode: маршрутизация по намерению.

    Если intent задан (fast paths: URL-only, file-only, override) — прямо в узел.
    Если intent не задан — запрос требует LLM-планирования → intent_node.
    Если pending_actions заполнен — multi_action_node.
    Если answer уже есть (ошибка) — mind_buddy_agent.
    """
    if state.get("answer"):
        return "mind_buddy_agent"

    if state.get("pending_actions"):
        return "multi_action_node"

    intent = state.get("intent")

    if not intent:
        return "intent_node"

    return _intent_to_node(intent)


def _route_after_resolver(state: AskState) -> str:
    """
    После ActionResolverNode: маршрутизация по результату.

    Если pending_actions — multi_action_node.
    Иначе — по intent (одношаговый план).
    """
    if state.get("answer"):
        return "mind_buddy_agent"

    if state.get("pending_actions"):
        return "multi_action_node"

    intent = state.get("intent")
    return _intent_to_node(intent)


def _intent_to_node(intent) -> str:
    """Маппинг intent → имя узла."""
    if intent == IntentType.SAVE_FILE:
        return "file_agent"
    elif intent == IntentType.SUMMARIZE:
        return "summary_node"
    elif intent == IntentType.INDEX_URL:
        return "index_url_node"
    elif intent == IntentType.LIST_FILES:
        return "execute_search_node"
    elif intent == IntentType.SEND_FILE:
        return "send_file_node"
    elif intent == IntentType.GENERAL_CHAT:
        return "mind_buddy_agent"
    elif intent in (
        IntentType.CREATE_NAMESPACE,
        IntentType.DELETE_NAMESPACE,
        IntentType.EDIT_NAMESPACE_NAME,
        IntentType.EDIT_NAMESPACE_DESCRIPTION,
        IntentType.MOVE_FILE,
        IntentType.CREATE_FILE,
        IntentType.DELETE_FILE,
        IntentType.EDIT_FILE,
        IntentType.RENAME_FILE,
        IntentType.SAVE_SUMMARY,
    ):
        return "crud_node"
    else:  # IntentType.RAG_QUERY и всё остальное
        return "compute_query_embedding"


def _route_after_save_file(state: AskState) -> str:
    """
    После SaveFileNode:
    - ошибка → mind_buddy_agent
    - файл сохранён + есть pending rag_query (file_save_notice) → compute_query_embedding
    - файл сохранён (answer) → END
    - иначе → mind_buddy_agent (ошибка)
    """
    if state.get("db_error"):
        return "mind_buddy_agent"
    if state.get("file_save_notice") and not state.get("answer"):
        return "compute_query_embedding"
    if state.get("answer"):
        return END
    return "mind_buddy_agent"


def _route_after_search(state: AskState) -> str:
    """
    После ExecuteSearchNode: при ошибке SQL и retry < N — повтор в sql_agent;
    иначе — mind_buddy_agent.
    Для list_files ретрай не нужен — SQL задан статически, SQLAgent его испортит.
    """
    if (
        state.get("db_error")
        and (state.get("retry_count") or 0) < MAX_SQL_RETRIES
        and state.get("intent") != IntentType.LIST_FILES
    ):
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
    namespace_service: Optional[NamespaceService] = None,
    content_extractor: Optional[ContentExtractorService] = None,
    task_publisher: Optional[TaskPublisher] = None,
) -> StateGraph:
    """
    Собирает граф для /ask. Перед использованием вызвать .compile().

    Структура графа:
    START -> RouterNode -> [fast paths: index_url, file_agent] | intent_node
    intent_node -> action_resolver_node -> [routing by intent] | multi_action_node -> mind_buddy_agent -> END
    """
    router_node = RouterNode()

    intent_classifier = IntentClassifierService(llm_service=llm_service)
    intent_node = IntentNode(intent_classifier=intent_classifier)
    action_resolver_node = ActionResolverNode(llm_service=llm_service)

    crud_node = CrudNode(
        file_service=file_service,
        namespace_service=namespace_service,
        llm_service=llm_service,
        storage=file_service.storage if file_service else None,
        task_publisher=task_publisher,
        content_extractor=content_extractor,
    )

    summary_node = SummaryNode(blob_storage=blob_storage)

    index_url_node: Optional[IndexUrlNode] = None
    if content_extractor and task_publisher:
        index_url_node = IndexUrlNode(
            content_extractor=content_extractor,
            file_service=file_service,
            task_publisher=task_publisher,
        )

    file_agent = FileAgent(
        file_reader_factory=file_reader_factory,
        text_chunker=text_chunker,
        embedding_service=embedding_service,
        blob_storage=blob_storage,
    )
    save_file_node = SaveFileNode(file_service=file_service, blob_storage=blob_storage)
    multi_action_node = MultiActionNode(
        crud_node=crud_node,
        index_url_node=index_url_node,
        summary_node=summary_node,
        generate_content_node=GenerateContentNode(llm_service=llm_service),
        file_agent=file_agent,
        save_file_node=save_file_node,
    )
    execute_search_node = ExecuteSearchNode()
    sql_agent = SQLAgent(llm_service=llm_service)
    mind_buddy_agent = MindBuddyAgent(llm_service=llm_service)
    query_embedding_node = QueryEmbeddingNode(embedding_service=embedding_service)
    send_file_node = SendFileNode(embedding_service=embedding_service, llm_service=llm_service)

    graph = StateGraph(AskState)

    graph.add_node("router", router_node.run)
    graph.add_node("intent_node", intent_node.run)
    graph.add_node("action_resolver_node", action_resolver_node.run)
    graph.add_node("crud_node", crud_node.run)
    graph.add_node("multi_action_node", multi_action_node.run)
    graph.add_node("file_agent", file_agent.run)
    graph.add_node("save_file_node", save_file_node.run)
    graph.add_node("execute_search_node", execute_search_node.run)
    graph.add_node("compute_query_embedding", query_embedding_node.run)
    graph.add_node("sql_agent", sql_agent.run)
    graph.add_node("mind_buddy_agent", mind_buddy_agent.run)
    graph.add_node("send_file_node", send_file_node.run)
    graph.add_node("summary_node", summary_node.run)
    if content_extractor and task_publisher:
        graph.add_node("index_url_node", index_url_node.run)

    # Рёбра графа
    graph.add_edge(START, "router")

    # Маршрутизация после роутера
    routing_map = {
        "intent_node": "intent_node",
        "file_agent": "file_agent",
        "summary_node": "summary_node",
        "compute_query_embedding": "compute_query_embedding",
        "execute_search_node": "execute_search_node",
        "send_file_node": "send_file_node",
        "crud_node": "crud_node",
        "multi_action_node": "multi_action_node",
        "mind_buddy_agent": "mind_buddy_agent",
    }
    if content_extractor and task_publisher:
        routing_map["index_url_node"] = "index_url_node"
    graph.add_conditional_edges("router", _route_after_router, routing_map)

    # IntentNode → ActionResolverNode (безусловное)
    graph.add_edge("intent_node", "action_resolver_node")

    # Маршрутизация после ActionResolverNode
    resolver_routing_map = {k: v for k, v in routing_map.items() if k != "intent_node"}
    graph.add_conditional_edges("action_resolver_node", _route_after_resolver, resolver_routing_map)

    graph.add_edge("file_agent", "save_file_node")
    graph.add_conditional_edges("save_file_node", _route_after_save_file, {
        END: END,
        "mind_buddy_agent": "mind_buddy_agent",
        "compute_query_embedding": "compute_query_embedding",
    })

    graph.add_edge("compute_query_embedding", "execute_search_node")
    graph.add_edge("sql_agent", "execute_search_node")
    graph.add_conditional_edges("execute_search_node", _route_after_search, {
        "sql_agent": "sql_agent",
        "mind_buddy_agent": "mind_buddy_agent",
    })
    graph.add_edge("mind_buddy_agent", END)
    graph.add_edge("crud_node", END)
    graph.add_edge("multi_action_node", "mind_buddy_agent")

    graph.add_edge("summary_node", END)
    graph.add_edge("send_file_node", END)
    if content_extractor and task_publisher:
        graph.add_edge("index_url_node", END)

    return graph
