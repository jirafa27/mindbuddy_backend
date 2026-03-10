"""Состояние графа для RAG-пайплайна POST /ask."""
from typing import TypedDict, Optional, Any, Dict, Literal, List


# Типы намерений пользователя
IntentType = Literal[
    "summarize",          # Суммаризация (URL, файла или из истории)
    "index_url",          # Сохранить URL в базу (без суммаризации)
    "save_file",          # Сохранить файл в базу
    "rag_query",          # Вопрос по базе знаний (поиск по файлам)
    "list_files",         # Перечислить файлы в пространстве/у пользователя
    "general_chat",       # Приветствие, болтовня — без поиска по файлам
    "send_file",          # Найти файл и отправить пользователю ссылку для скачивания
    "create_namespace",   # Создать пространство знаний
    "delete_namespace",   # Удалить пространство знаний
    "edit_namespace",     # Редактировать название/описание пространства
    "move_file",          # Переместить файл в пространство
    "create_file",        # Создать файл из текста
    "delete_file",        # Удалить файл
    "edit_file",          # Редактировать содержимое файла
    "save_summary",       # Сохранить последний ответ ассистента как файл
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
    history: list[dict[str, str]]  # [{"role": "user"|"assistant", "text": "...", "file_ids": [...]}]

    # Файл (опционально)
    file_content: bytes
    filename: str

    blob_key: Optional[str]

    # Намерение пользователя (определяется RouterNode или из override_intent)
    intent: IntentType
    detected_url: Optional[str]  # URL из вопроса или истории
    history_file_id: Optional[int]  # file_id из истории

    # Имя пространства, упомянутое в тексте вопроса (напр. "в пространстве Работа")
    namespace_name_hint: Optional[str]

    # После сохранения в SaveFileNode (один файл)
    file_id: Optional[int]
    # После send_file: список найденных файлов для скачивания
    file_ids: list[int]

    # Скоуп поиска по файлам (список user_files.id)
    search_file_ids: Optional[List[int]]

    # Поиск
    query_embedding: list[float]
    sql_query: str
    search_result: list[dict[str, Any]]  # chunk_text, filename, relevance
    db_error: Optional[str]
    retry_count: int

    # Поисковый запрос, очищенный LLM (без команд, на чистую тему)
    # Используется QueryEmbeddingNode и SendFileNode вместо полного question
    search_query: Optional[str]

    # Режим поиска для SEND_FILE: "by_topic" | "by_name" | "by_content"
    send_file_search_mode: Optional[str]

    # Суммаризация
    summary_result: Optional[dict[str, Any]]  # file_id, summary, title, source_url

    # Параметры для CRUD-операций (create_namespace, create_file, edit_file и т.д.)
    entity_name: Optional[str]         # имя пространства или заголовок файла
    entity_description: Optional[str]  # описание пространства
    entity_content: Optional[str]      # содержимое файла (для create_file / edit_file)

    # Отложенное действие (хранится в чате, передаётся ChatService → grafu при confirm)
    pending_action: Optional[Dict[str, Any]]

    # Уведомление о сохранении файла (используется когда после сохранения нужно выполнить RAG-поиск)
    file_save_notice: Optional[str]

    # True, если пространство было создано автоматически в процессе загрузки файла
    namespace_created: Optional[bool]

    # True, если контент файла уже проиндексирован (эмбеддинги уже есть в БД)
    content_already_indexed: Optional[bool]

    # Ответ
    agent_steps: list[str]
    answer: str
    sources: list[dict[str, Any]]  # filename, relevance
