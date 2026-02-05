"""SQLAgent: генерация Raw SQL для векторного поиска по эмбеддингам."""
import logging
from typing import Any

from app.agents.state import AskState
from app.domain.protocols import LLMProvider
from app.infrastructure.llm.yandex_completion import LLMCompletionError

logger = logging.getLogger(__name__)

SCHEMA_HINT = """
Таблицы:
- vector_embeddings: id, file_id, namespace_id (может быть NULL), chunk_index, chunk_text, embedding (тип vector(256))
- files: id, namespace_id, user_id, filename, file_path, file_type, file_size, created_at, updated_at
- namespaces: id, user_id, name, description, created_at

ОПРЕДЕЛИ ТИП ЗАПРОСА:

ТИП 1 — СТРУКТУРНЫЙ ЗАПРОС (о файлах, папках, пространствах):
Ключевые слова: "что лежит", "какие файлы", "покажи файлы", "список файлов", "что в папке", "что в пространстве", "мои папки", "сколько файлов", "перечисли"
Примеры вопросов:
- "Что у меня лежит в пространстве X?" → СТРУКТУРНЫЙ
- "Какие файлы в папке Y?" → СТРУКТУРНЫЙ
- "Покажи мои пространства" → СТРУКТУРНЫЙ
- "Сколько у меня файлов?" → СТРУКТУРНЫЙ

Для СТРУКТУРНЫХ запросов:
- Используй ТОЛЬКО таблицы files и namespaces
- НЕ используй vector_embeddings
- НЕ используй :query_embedding и embedding
- Фильтруй по имени пространства: n.name ILIKE '%ИмяПространства%'

Пример SQL для "Что лежит в пространстве Пурум?":
SELECT f.filename, f.file_size, f.created_at, n.name as namespace_name
FROM files f
JOIN namespaces n ON f.namespace_id = n.id
WHERE n.user_id = :user_id AND n.name ILIKE '%Пурум%'
LIMIT :limit

ТИП 2 — СЕМАНТИЧЕСКИЙ ПОИСК (о содержимом документов):
Ключевые слова: "о чём", "что написано", "найди информацию", "расскажи про", "объясни", вопросы о смысле/содержании
Примеры вопросов:
- "О чём фильм Жизнь Чака?" → СЕМАНТИЧЕСКИЙ
- "Что написано про архитектуру?" → СЕМАНТИЧЕСКИЙ
- "Найди информацию о проекте" → СЕМАНТИЧЕСКИЙ

Для СЕМАНТИЧЕСКИХ запросов:
- Используй vector_embeddings с оператором <=>
- Включи ve.chunk_text в SELECT
- ORDER BY ve.embedding <=> CAST(:query_embedding AS vector) ASC
- ВАЖНО: используй CAST(:query_embedding AS vector), НЕ ::vector

Пример SQL для семантического поиска:
SELECT ve.chunk_text, f.filename, 1 - (ve.embedding <=> CAST(:query_embedding AS vector)) AS relevance
FROM vector_embeddings ve
JOIN files f ON f.id = ve.file_id
WHERE f.user_id = :user_id
ORDER BY ve.embedding <=> CAST(:query_embedding AS vector) ASC
LIMIT :limit

ОБЩИЕ ПРАВИЛА:
- ВСЕГДА добавляй WHERE ... user_id = :user_id
- ВСЕГДА добавляй LIMIT :limit
- Если указан namespace_id — добавь фильтр namespace_id = :namespace_id
- Верни ТОЛЬКО SQL-запрос, без пояснений и markdown
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

    async def run(self, state: AskState, config: dict | None = None) -> dict[str, Any]:
        question = state.get("question") or ""
        namespace_id = state.get("namespace_id")
        db_error = state.get("db_error")
        agent_steps = list(state.get("agent_steps") or [])

        if not question:
            return {"agent_steps": agent_steps + ["SQLAgent"]}

        user_id = state.get("user_id")
        if namespace_id is not None:
            ns_hint = f"namespace_id = {namespace_id} (поиск внутри конкретного пространства)."
        else:
            ns_hint = "namespace_id = NULL (поиск по ВСЕМ файлам пользователя)."
        user_text = (
            f"Сгенерируй один SQL-запрос для поиска по базе знаний.\n"
            f"Вопрос пользователя: {question}\n"
            f"user_id = {user_id}\n"
            f"{ns_hint}\n"
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
