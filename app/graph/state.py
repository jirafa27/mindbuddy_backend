"""Состояние графа для RAG-пайплайна POST /ask."""
from typing import TypedDict, Optional, Any, Literal


# Типы намерений пользователя
IntentType = Literal[
    "summarize",     # Суммаризация (URL, файла или из истории)
    "index_url",     # Сохранить URL в базу (без суммаризации)
    "save_file",     # Сохранить файл в базу
    "rag_query",     # Обычный вопрос по базе знаний
]


class AskState(TypedDict, total=False):
    """
    Единое состояние графа LangGraph для /ask.
    
    Содержит только данные. Инфраструктурные зависимости (db, repositories)
    передаются через RunnableConfig["configurable"].
    """

    question: str
    namespace_id: int
    user_id: int
    
    # Принудительный интент (если задан — RouterNode не анализирует текст)
    override_intent: Optional[IntentType]
    
    # История сообщений (последние N сообщений для контекста)
    history: list[dict[str, str]]  # [{"role": "user"|"assistant", "text": "...", "file_id": ...}]

    # Файл (опционально)
    file_content: bytes
    filename: str

    blob_key: Optional[str]

    # Намерение пользователя (определяется RouterNode или из override_intent)
    intent: IntentType
    detected_url: Optional[str]  # URL из вопроса или истории
    history_file_id: Optional[int]  # file_id из истории

    # После сохранения в SaveFileNode
    file_id: Optional[int]

    # Поиск
    query_embedding: list[float]
    sql_query: str
    search_result: list[dict[str, Any]]  # chunk_text, filename, relevance
    db_error: Optional[str]
    retry_count: int

    # Суммаризация
    summary_result: Optional[dict[str, Any]]  # file_id, summary, title, source_url

    # Ответ
    agent_steps: list[str]
    answer: str
    sources: list[dict[str, Any]]  # filename, relevance
