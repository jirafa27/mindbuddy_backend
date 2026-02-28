"""Перечисления для приложения"""
from enum import Enum



class SummaryMethod(str, Enum):
    """Метод суммаризации."""
    STUFFING = "stuffing"      # Короткие тексты — один запрос к LLM
    MAP_REDUCE = "map_reduce"  # Длинные тексты — чанки + объединение
    CACHED = "cached"          # Из кэша (уже была суммаризация)



class IntentType(str, Enum):
    """Тип намерения пользователя."""
    SUMMARIZE = "summarize"     # Суммаризация (URL, файла или из истории)
    INDEX_URL = "index_url"     # Сохранить URL в базу (без суммаризации)
    SAVE_FILE = "save_file"     # Сохранить файл в базу
    RAG_QUERY = "rag_query"     # Обычный вопрос по базе знаний