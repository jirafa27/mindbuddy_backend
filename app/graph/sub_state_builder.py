"""
Централизованный builder для sub-state шагов графа.

Вместо {**state, ...} каждый узел/шаг получает только поля,
которые ему нужны согласно типу интента.
"""
from typing import Any

# --------------------------------------------------------------------------
# Группы полей
# --------------------------------------------------------------------------

COMMON: set[str] = {
    "user_id",
    "question",
    "intent",
    "namespace_id",
    "namespace_name_hint",
    "agent_steps",
}

ACTION: set[str] = {
    "entity_name",
    "entity_description",
    "entity_content",
    "search_query",
    "search_limit",
}

FILE_CONTEXT: set[str] = {
    "history_file_id",
    "search_file_ids",
    "attached_files",
    "detected_url",
    "blobs",
}

CHAT_CONTEXT: set[str] = {
    "history",
    "search_result",
    "answer",
    "pipeline_report",
}

# --------------------------------------------------------------------------
# Таблица разрешённых полей по интенту
# --------------------------------------------------------------------------

_INTENT_FIELDS: dict[str, set[str]] = {
    # CRUD — файловый контекст из истории НЕ передаётся
    # detected_url — для _create_file при пустом entity_content (URL из роутера)
    "create_file":                COMMON | ACTION | {"history", "detected_url"},
    "edit_file":                  COMMON | ACTION | {"history", "history_file_id", "search_file_ids"},
    "rename_file":                COMMON | ACTION | {"history", "history_file_id", "search_file_ids"},
    "delete_file":                COMMON | ACTION | {"history", "history_file_id", "search_file_ids"},
    "move_file":                  COMMON | ACTION | {"history", "history_file_id", "search_file_ids"},
    "create_namespace":           COMMON | ACTION,
    "delete_namespace":           COMMON | ACTION,
    "edit_namespace_name":        COMMON | ACTION,
    "edit_namespace_description": COMMON | ACTION,
    "save_summary":               COMMON | ACTION | {"history"},
    "list_files":                 COMMON | {"sql_query"},
    # Файловые операции — нужен FILE_CONTEXT
    "save_file":                  COMMON | ACTION | FILE_CONTEXT,
    "send_file":                  COMMON | ACTION | FILE_CONTEXT | {"send_file_search_mode"},
    # Смысловые операции — нужен FILE_CONTEXT
    "summarize":                  COMMON | ACTION | FILE_CONTEXT,
    "rag_query":                  COMMON | ACTION | FILE_CONTEXT,
    # Прочие
    "index_url":                  COMMON | {"detected_url"},
    "general_chat":               COMMON | CHAT_CONTEXT,
}


def build_sub_state(state: dict[str, Any], intent: str, **overrides: Any) -> dict[str, Any]:
    """
    Собирает sub-state для шага с заданным интентом.

    Из `state` берутся только поля, разрешённые для данного интента.
    `overrides` применяются поверх без фильтрации None
    (например, answer=None — корректный сброс значения).
    """
    allowed = _INTENT_FIELDS.get(intent, COMMON | ACTION | FILE_CONTEXT)
    result: dict[str, Any] = {
        k: state[k] for k in allowed if k in state and state[k] is not None
    }
    result.update(overrides)
    # Второй позиционный аргумент — интент шага; не дублировать как intent= в overrides (TypeError).
    result["intent"] = intent
    return result
