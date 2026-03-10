"""Перечисления для приложения"""
from enum import Enum



class SummaryMethod(str, Enum):
    """Метод суммаризации."""
    STUFFING = "stuffing"      # Короткие тексты — один запрос к LLM
    MAP_REDUCE = "map_reduce"  # Длинные тексты — чанки + объединение
    CACHED = "cached"          # Из кэша (уже была суммаризация)



class IntentType(str, Enum):
    """Тип намерения пользователя."""
    SUMMARIZE = "summarize"                   # Суммаризация (URL, файла или из истории)
    INDEX_URL = "index_url"                   # Сохранить URL в базу (без суммаризации)
    SAVE_FILE = "save_file"                   # Сохранить файл в базу
    RAG_QUERY = "rag_query"                   # Вопрос по базе знаний (поиск по файлам)
    LIST_FILES = "list_files"                 # Перечислить файлы в пространстве/у пользователя
    GENERAL_CHAT = "general_chat"             # Приветствие, болтовня — без поиска по файлам
    SEND_FILE = "send_file"                   # Найти файл и отправить пользователю ссылку для скачивания
    CREATE_NAMESPACE = "create_namespace"     # Создать пространство знаний
    DELETE_NAMESPACE = "delete_namespace"     # Удалить пространство знаний
    EDIT_NAMESPACE = "edit_namespace"         # Редактировать название/описание пространства
    MOVE_FILE = "move_file"                   # Переместить файл в пространство
    CREATE_FILE = "create_file"               # Создать файл из текста
    DELETE_FILE = "delete_file"               # Удалить файл
    EDIT_FILE = "edit_file"                   # Редактировать содержимое файла
    SAVE_SUMMARY = "save_summary"             # Сохранить последний ответ ассистента как файл


class ChatMessageRole(str, Enum):
    """Роль сообщения в чате."""
    USER = "user"
    ASSISTANT = "assistant"