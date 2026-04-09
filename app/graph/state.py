"""Состояние графа для RAG-пайплайна POST /ask."""
from typing import TypedDict, Optional, Any, Dict, Literal, List, Tuple


class AttachedFile(TypedDict, total=False):
    """
    Один файл из batch-загрузки: метаданные + ключ сырых байтов в BlobStorage.
    Байты не хранятся в state — скачиваются узлами по file_blob_key.
    """
    # Ключ в MinIO: payload {"raw": bytes, "filename": str}.
    file_blob_key: str
    # Имя файла из HTTP-запроса (до decode_filename).
    filename: str
    # MIME-тип из HTTP-запроса (например "application/pdf"). None, если не передан.
    content_type: Optional[str]
    # Размер файла в байтах.
    size: int


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
    "edit_namespace_name",        # Переименовать пространство
    "edit_namespace_description", # Изменить описание пространства
    "move_file",          # Переместить файл в пространство
    "create_file",        # Создать файл из текста
    "delete_file",        # Удалить файл
    "edit_file",          # Редактировать содержимое файла
    "save_summary",       # Сохранить последний ответ ассистента как файл
]


class FileSaveBlob(TypedDict, total=False):
    """
    Одна запись в state["blobs"]: промежуточный результат FileAgent для SaveFileNode.
    Сырые байты файла в state не кладутся — только ключи MinIO и метаданные.
    """

    # Имя файла после decode_filename (для upload_file и сообщений пользователю).
    filename: str
    # SHA-256 hex исходных байтов; совпадает с тем, что использовал FileAgent для дедупликации.
    content_hash: str
    # Ключ в MinIO: payload с chunks + embeddings для vector_repository.create_batch.
    # None, если эмбеддинги не генерировали (контент уже в БД).
    blob_key: Optional[str]
    # Ключ в MinIO: payload {"raw": bytes, "filename": str} — байты для file_service.upload_file.
    # None при early_duplicate (сохранять нечего).
    file_blob_key: Optional[str]
    # True: ContentFile с эмбеддингами уже есть; нужен только UserFile в namespace (blob_key обычно None).
    content_already_indexed: bool
    # True: полный дубликат в этом namespace — SaveFileNode только возвращает file_id/answer, без MinIO/БД.
    early_duplicate: bool
    # True: файл не распарсился или пустой чанкинг — SaveFileNode пропускает запись тихо.
    parse_error: bool
    # При early_duplicate: user_file.id для ответа и search_file_ids.
    file_id: Optional[int]
    # При early_duplicate: текст «файл уже есть в пространстве».
    answer: Optional[str]
    # При early_duplicate: [user_file.id] для последующего RAG.
    search_file_ids: Optional[List[int]]


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
    
    # Персистентный контекст диалога (ConversationContext.to_dict())
    conversation_context: Optional[Dict[str, Any]]

    # История сообщений (последние N сообщений для контекста)
    history: list[dict[str, str]]  # [{"role": "user"|"assistant", "text": "...", "file_ids": [...]}]

    # Файлы batch-загрузки: список AttachedFile — метаданные + ключ MinIO.
    # Байты файлов в state не хранятся — только ключи BlobStorage.
    attached_files: Optional[List[AttachedFile]]

    # См. FileSaveBlob — по одному элементу на каждый файл из attached_files.
    blobs: Optional[List[FileSaveBlob]]

    # Намерение пользователя (определяется RouterNode или из override_intent)
    intent: IntentType
    detected_url: Optional[str]  # URL из вопроса или истории
    history_file_id: Optional[int]  # file_id из истории

    # Флаги контекста URL (выставляются RouterNode, используются ActionResolverNode)
    url_in_current_message: Optional[bool]  # URL в текущем сообщении (не из истории)
    has_history_url: Optional[bool]         # URL найден в истории, но не в текущем сообщении

    # Имя пространства, упомянутое в тексте вопроса (напр. "в пространстве Работа")
    namespace_name_hint: Optional[str]

    # После сохранения в SaveFileNode (один файл)
    file_id: Optional[int]
    # После send_file: список найденных файлов для скачивания
    file_ids: list[int]

    # Скоуп поиска по файлам (список user_files.id)
    search_file_ids: Optional[List[int]]

    # True, если search_file_ids задан явно через API (file_ids query param),
    # а не извлечён автоматически из истории чата RouterNode-ом.
    explicit_file_ids: Optional[bool]

    # Поиск
    query_embedding: list[float]
    sql_query: str
    search_result: list[dict[str, Any]]  # chunk_text, filename, relevance
    db_error: Optional[str]
    retry_count: int

    # Поисковый запрос, очищенный LLM (без команд, на чистую тему)
    # Используется QueryEmbeddingNode и SendFileNode вместо полного question
    search_query: Optional[str]

    # Максимальное число чанков для векторного поиска (по умолчанию 10 в ExecuteSearchNode)
    search_limit: Optional[int]

    # Режим поиска для SEND_FILE: "by_topic" | "by_name" | "by_content"
    send_file_search_mode: Optional[str]

    # Суммаризация
    summary_result: Optional[dict[str, Any]]  # file_id, summary, title, source_url

    # Отчёт о выполнении pipeline/multi_action шагов для MindBuddyAgent
    # Каждый элемент: {"step": str, "ok": bool, "message": str}
    pipeline_report: Optional[List[dict[str, Any]]]

    # Параметры для CRUD-операций (create_namespace, create_file, edit_file и т.д.)
    entity_name: Optional[str]         # имя пространства или заголовок файла
    entity_description: Optional[str]  # описание пространства
    entity_content: Optional[str]      # содержимое файла (для create_file / edit_file)

    # Отложенное действие (хранится в чате, передаётся ChatService → grafu при confirm)
    pending_action: Optional[Dict[str, Any]]

    # Список намерений, определённых IntentNode (для ActionResolverNode)
    planned_intents: Optional[List[str]]

    # Список действий для MultiActionNode (multi_action intent)
    pending_actions: Optional[List[Dict[str, Any]]]

    # Уведомление о сохранении файла (используется когда после сохранения нужно выполнить RAG-поиск)
    file_save_notice: Optional[str]

    # True, если пространство было создано автоматически в процессе загрузки файла
    namespace_created: Optional[bool]

    # ID и имя пространства, созданного в рамках текущего запроса (MultiActionNode → ChatService)
    created_namespace_id: Optional[int]
    created_namespace_name: Optional[str]

    # True, если контент файла уже проиндексирован (эмбеддинги уже есть в БД)
    content_already_indexed: Optional[bool]

    # Ответ
    agent_steps: list[str]
    answer: str
    sources: list[dict[str, Any]]  # filename, relevance
