"""SQLAgent: генерация Raw SQL для векторного поиска по эмбеддингам."""
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.graph.state import AskState
from app.domain.protocols import LLMProvider
from app.infrastructure.llm.yandex_completion import LLMCompletionError

logger = logging.getLogger(__name__)

SCHEMA_HINT = """
Таблицы (реальная схема БД):
- vector_embeddings: id, file_id, chunk_index, chunk_text, embedding (vector(256)) — НЕТ колонки namespace_id
- files: id, content_hash, source_url, file_path, media_metadata (JSON), created_at — НЕТ filename, НЕТ user_id
- user_files: id, user_id, file_id, namespace_id, custom_title, created_at — связь пользователь–файл, user_id здесь
- namespaces: id, user_id, name, description, created_at

Имя файла для вывода: COALESCE(uf.custom_title, f.media_metadata->>'title', f.source_url, f.file_path, 'Document') AS filename.
Фильтр по пользователю: только через user_files — uf.user_id = :user_id (в таблице files НЕТ user_id).

ТИП 1 — СТРУКТУРНЫЙ: используй user_files uf JOIN files f ON f.id = uf.file_id JOIN namespaces n ON n.id = uf.namespace_id, WHERE uf.user_id = :user_id. Не используй vector_embeddings.

ТИП 2 — СЕМАНТИЧЕСКИЙ: ОБЯЗАТЕЛЬНО JOIN vector_embeddings ve JOIN files f ON f.id = ve.file_id JOIN user_files uf ON uf.file_id = ve.file_id.
Фильтр: WHERE uf.user_id = :user_id. Имя файла: COALESCE(uf.custom_title, f.media_metadata->>'title', f.source_url, f.file_path, 'Document') AS filename.
Если указан namespace_id: AND (:namespace_id::integer IS NULL OR uf.namespace_id = :namespace_id). НЕ используй ve.namespace_id — такой колонки нет.
Если нужно ограничить поиск одним файлом: AND ve.file_id = :file_id.

Пример SQL для семантического поиска (строго по этой схеме):
SELECT ve.chunk_text, COALESCE(uf.custom_title, f.media_metadata->>'title', f.source_url, f.file_path, 'Document') AS filename, 1 - (ve.embedding <=> CAST(:query_embedding AS vector)) AS relevance
FROM vector_embeddings ve
JOIN files f ON f.id = ve.file_id
JOIN user_files uf ON uf.file_id = ve.file_id
WHERE uf.user_id = :user_id
ORDER BY ve.embedding <=> CAST(:query_embedding AS vector) ASC
LIMIT :limit

Правила: user_id только через uf.user_id; filename только через COALESCE(...); LIMIT :limit; не используй f.filename, f.user_id, ve.namespace_id. Верни только SQL.
"""


def _extract_sql(text: str) -> str:
    """Извлекает один SQL-запрос из ответа (убирает markdown, лишние пробелы)."""
    text = text.strip()
    for marker in ("```sql", "```SQL", "```"):
        if marker in text:
            parts = text.split(marker)
            if len(parts) >= 2:
                text = parts[1].strip()
                if text.endswith("```"):
                    text = text[:-3].strip()
                break
    return text.strip() if text else ""


class SQLAgent:
    """Генерирует SQL для векторного поиска по question и схеме. Поддерживает self-correction по db_error."""

    def __init__(self, *, llm_service: LLMProvider):
        self.llm_service = llm_service

    async def run(self, state: AskState, config: RunnableConfig) -> dict[str, Any]:
        question = state.get("question") or ""
        namespace_id = state.get("namespace_id")
        db_error = state.get("db_error")
        agent_steps = list(state.get("agent_steps") or [])

        if not question:
            return {"agent_steps": agent_steps + ["SQLAgent"]}

        user_id = state.get("user_id")
        search_file_ids = state.get("search_file_ids") or []

        if namespace_id is not None:
            ns_hint = f"namespace_id = {namespace_id} (поиск внутри конкретного пространства)."
        else:
            ns_hint = "namespace_id = NULL (поиск по ВСЕМ файлам пользователя)."

        file_hint = ""
        if search_file_ids:
            file_hint = (
                f"\nПоиск только по файлам с user_files.id IN ({','.join(map(str, search_file_ids))}). "
                "Используй шаблон с IN (__FILE_IDS__) для фильтра по файлам."
            )

        user_text = (
            f"Сгенерируй один SQL-запрос для поиска по базе знаний.\n"
            f"Вопрос пользователя: {question}\n"
            f"user_id = {user_id}\n"
            f"{ns_hint}"
            f"{file_hint}\n"
            f"{SCHEMA_HINT}"
        )
        if db_error:
            user_text += f"\nПредыдущий запрос вернул ошибку, исправь SQL:\n{db_error}"

        messages = [
            {"role": "system", "text": "Ты генерируешь только валидный SQL для PostgreSQL с pgvector. Ответь одним запросом, без markdown."},
            {"role": "user", "text": user_text},
        ]

        retry_count = state.get("retry_count") or 0
        if db_error:
            retry_count = retry_count + 1

        try:
            raw = await self.llm_service.complete(messages, temperature=0.1, max_tokens=1024)
            sql = _extract_sql(raw)
            if not sql:
                return {
                    "sql_query": "",
                    "db_error": "Не удалось извлечь SQL из ответа модели",
                    "retry_count": retry_count,
                    "agent_steps": agent_steps + ["SQLAgent"],
                }
            logger.info (f"SQL: {sql}")
            return {
                "sql_query": sql,
                "db_error": None,
                "retry_count": retry_count,
                "agent_steps": agent_steps + ["SQLAgent"],
            }
        except LLMCompletionError as e:
            return {
                "sql_query": "",
                "db_error": str(e),
                "retry_count": retry_count,
                "agent_steps": agent_steps + ["SQLAgent"],
            }
