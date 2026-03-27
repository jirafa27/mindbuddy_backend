"""SendFileNode: поиск файлов пользователя и возврат списка file_id для скачивания."""
import logging
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from sqlalchemy import text

from app.graph.state import AskState
from app.domain.protocols import EmbeddingProvider, LLMProvider
from app.infrastructure.repositories.vector_queries import (
    FIND_FILE_BY_TOPIC_SQL,
    FIND_FILE_BY_CONTENT_SQL,
)

logger = logging.getLogger(__name__)

# Минимальная длина слова для включения в поиск по имени
_MIN_WORD_LEN = 3
# Длина префикса-стемма (обход русских падежей)
_STEM_LEN = 7
# Минимальная релевантность вектор-поиска (ниже — считаем нерелевантным)
_MIN_VECTOR_RELEVANCE = 0.68
# Максимальное количество файлов, возвращаемых за один запрос
SEND_FILE_LIMIT = 10

# Базовый SQL для поиска файла по имени (ILIKE)
# uf.id — это user_files.id, именно его ожидает file_service.download_file()
_FIND_FILE_BY_NAME_SQL = """
SELECT uf.id AS file_id,
       COALESCE(uf.custom_title, f.media_metadata->>'title', f.source_url, f.file_path, 'Document') AS filename
FROM user_files uf
JOIN files f ON f.id = uf.file_id
WHERE uf.user_id = :user_id
  AND (CAST(:namespace_id AS integer) IS NULL OR uf.namespace_id = :namespace_id)
"""

_LIST_ALL_IN_NAMESPACE_SQL = """
SELECT uf.id AS file_id,
       COALESCE(uf.custom_title, f.media_metadata->>'title', f.source_url, f.file_path, 'Document') AS filename
FROM user_files uf
JOIN files f ON f.id = uf.file_id
WHERE uf.user_id = :user_id
  AND uf.namespace_id = :namespace_id
ORDER BY uf.created_at DESC
LIMIT :limit
"""

_FILENAME_EXPR = (
    "LOWER(COALESCE(uf.custom_title, f.media_metadata->>'title',"
    " f.source_url, f.file_path, ''))"
)


def _escape_ilike(term: str) -> str:
    """
    Экранирует спецсимволы ILIKE: % и _.
    В PostgreSQL LIKE/ILIKE символ '_' означает «любой один символ»,
    поэтому «Rynduk_konspekt.md» без экранирования совпадёт и с
    «RyndukXkonspekt.md», и с «Rynduk_konspekt_summary.md».
    """
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _build_exact_name_sql(name_term: str) -> tuple[str, dict]:
    """
    Строит SQL для поиска по точному (экранированному) имени файла.
    Использует ILIKE ESCAPE '\\' чтобы '_' и '%' трактовались буквально.
    """
    params: dict = {}
    if not name_term:
        return _FIND_FILE_BY_NAME_SQL + "ORDER BY uf.created_at DESC\nLIMIT :limit", params

    escaped = _escape_ilike(name_term)
    params["pattern"] = f"%{escaped}%"
    sql = (
        _FIND_FILE_BY_NAME_SQL
        + f"  AND {_FILENAME_EXPR} ILIKE LOWER(:pattern) ESCAPE '\\'\n"
        + "ORDER BY uf.created_at DESC\nLIMIT :limit"
    )
    return sql, params


def _build_stem_search_sql(search_term: str) -> tuple[str, dict]:
    """
    Строит SQL для поиска файла по пословным стеммам (обход русских падежей).
    Пример: «информационную безопасность» → ILIKE '%информа%' AND ILIKE '%безопас%'
    """
    params: dict = {}

    if not search_term:
        return _FIND_FILE_BY_NAME_SQL + "ORDER BY uf.created_at DESC\nLIMIT :limit", params

    words = [w for w in search_term.split() if len(w) >= _MIN_WORD_LEN]

    if not words:
        params["pattern"] = f"%{search_term}%"
        sql = _FIND_FILE_BY_NAME_SQL + f"  AND {_FILENAME_EXPR} ILIKE LOWER(:pattern)\n"
        return sql + "ORDER BY uf.created_at DESC\nLIMIT :limit", params

    conditions = []
    for i, word in enumerate(words):
        stem = word[:_STEM_LEN] if len(word) > _STEM_LEN else word
        key = f"w{i}"
        params[key] = f"%{stem}%"
        conditions.append(f"{_FILENAME_EXPR} ILIKE LOWER(:{key})")

    where = "  AND " + "\n  AND ".join(conditions) + "\n"
    return _FIND_FILE_BY_NAME_SQL + where + "ORDER BY uf.created_at DESC\nLIMIT :limit", params


def _build_answer(rows: List[dict], search_term: str) -> str:
    """
    Формирует текст ответа в зависимости от количества найденных файлов.

    - 0 файлов: сообщение «не найдено»
    - 1 файл: «Вот файл»
    - 2–N (< лимита): «Найдено N файлов»
    - N == лимит: «Показаны N наиболее подходящих» (намекаем на широкий запрос)
    """
    count = len(rows)
    if count == 0:
        hint = f" по запросу «{search_term}»" if search_term else ""
        return f"Не нашёл файлов{hint}. Уточните название или проверьте, что файлы загружены."

    if count == 1:
        return f"Вот ваш файл «{rows[0]['filename']}». Нажмите, чтобы скачать."

    names = "\n".join(f"• {r['filename']}" for r in rows)
    if count < SEND_FILE_LIMIT:
        return f"Найдено {count} файла(-ов):\n{names}"

    return (
        f"Показаны {count} наиболее подходящих файлов:\n{names}\n\n"
        "Если нужен конкретный файл — уточните запрос (по названию или теме)."
    )


class SendFileNode:
    """
    Ищет файлы пользователя и возвращает список file_id для скачивания.

    Режимы поиска (определяются RouterNode через send_file_search_mode):
    - by_name: ILIKE-поиск по названию файла.
    - by_content: поиск по буквальному вхождению текста в чанки.
    - by_topic: семантический поиск по содержанию (по умолчанию).

    Поисковый термин берётся из state["search_query"] (очищен LLM).
    Возвращает file_ids: list[int] и sources с file_id в каждом элементе.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingProvider] = None,
        llm_service: Optional[LLMProvider] = None,
    ):
        self.embedding_service = embedding_service
        self.llm_service = llm_service

    async def _llm_rerank(self, candidates: List[dict], search_term: str) -> List[dict]:
        """
        Фильтрует кандидатов по именам файлов через LLM.
        Возвращает только те файлы, которые LLM считает релевантными теме.
        При ошибке возвращает исходный список без изменений.
        """
        if not self.llm_service or not candidates:
            return candidates

        lines = []
        for i, r in enumerate(candidates):
            snippet = (r.get("snippet") or "").strip()
            if snippet:
                lines.append(f"{i}: {r['filename']}\n   Фрагмент: {snippet}")
            else:
                lines.append(f"{i}: {r['filename']}")
        file_list = "\n\n".join(lines)
        logger.info("[SendFileNode] LLM rerank input:\n%s", file_list)

        messages = [
            {
                "role": "system",
                "text": (
                    "Ты помощник по поиску файлов. Тебе дан запрос и список файлов с фрагментами их содержимого. "
                    "Верни через запятую номера ТОЛЬКО тех файлов, которые соответствуют запросу по теме или содержанию. "
                    "Если ни один не подходит — верни 'none'. "
                    "Отвечай только номерами или словом 'none', без лишних слов."
                ),
            },
            {
                "role": "user",
                "text": f"Запрос: «{search_term}»\n\nФайлы:\n\n{file_list}",
            },
        ]
        try:
            raw = await self.llm_service.complete(messages, temperature=0.0, max_tokens=64)
            raw = raw.strip().lower()
            logger.info("[SendFileNode] LLM rerank raw: %r", raw)
            if raw == "none" or not raw:
                return []
            indices = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
            result = [candidates[i] for i in indices if 0 <= i < len(candidates)]
            logger.info(
                "[SendFileNode] LLM rerank: %d -> %d file(s)",
                len(candidates), len(result),
            )
            return result
        except Exception as exc:
            logger.warning("[SendFileNode] LLM rerank failed: %s", exc)
            return candidates

    async def run(self, state: AskState, config: RunnableConfig) -> dict[str, Any]:
        configurable = config.get("configurable") or {}
        db = configurable.get("async_db")

        user_id = state.get("user_id")
        namespace_id = state.get("namespace_id")
        history_file_id = state.get("history_file_id")
        question = state.get("question", "")

        agent_steps = list(state.get("agent_steps") or []) + ["SendFileNode"]

        search_term = state.get("search_query") or question
        mode = state.get("send_file_search_mode") or "by_topic"

        # Файл из контекста — только если пользователь не называл конкретное имя
        if history_file_id and not search_term.strip():
            logger.info("[SendFileNode] Using history_file_id=%s", history_file_id)
            return {
                "file_ids": [history_file_id],
                "answer": "Вот ваш файл. Нажмите, чтобы скачать.",
                "sources": [{"filename": "Файл из контекста", "relevance": 1.0, "file_id": history_file_id}],
                "agent_steps": agent_steps,
            }

        if not db or user_id is None:
            return {
                "file_ids": [],
                "answer": "Не удалось найти файл: ошибка конфигурации.",
                "sources": [],
                "agent_steps": agent_steps,
            }

        logger.info(
            "[SendFileNode] mode=%s term=%r (user_id=%s, namespace_id=%s)",
            mode, search_term, user_id, namespace_id,
        )

        if mode == "by_name":
            rows = await self._search_by_name(db, user_id, namespace_id, search_term)
        elif mode == "by_content":
            rows = await self._search_by_content(db, user_id, namespace_id, search_term)
        elif mode == "all_in_namespace":
            rows = await self._list_all_in_namespace(db, user_id, namespace_id)
        else:
            rows = await self._search_by_topic(db, user_id, namespace_id, search_term)

        answer = _build_answer(rows, search_term)
        file_ids = [r["file_id"] for r in rows]
        sources = [
            {
                "filename": r["filename"],
                "relevance": float(r.get("relevance") or 1.0),
                "file_id": r["file_id"],
            }
            for r in rows
        ]

        logger.info(
            "[SendFileNode] Found %d file(s): %s",
            len(rows),
            [r["filename"] for r in rows],
        )

        return {
            "file_ids": file_ids,
            "answer": answer,
            "sources": sources,
            "agent_steps": agent_steps,
        }

    async def _list_all_in_namespace(
        self, db, user_id, namespace_id
    ) -> List[dict]:
        """Возвращает все файлы пользователя в указанном пространстве."""
        if not namespace_id:
            logger.warning("[SendFileNode] _list_all_in_namespace: namespace_id is None")
            return []
        try:
            result = await db.execute(
                text(_LIST_ALL_IN_NAMESPACE_SQL),
                {"user_id": user_id, "namespace_id": namespace_id, "limit": SEND_FILE_LIMIT},
            )
            rows = [dict(r) for r in result.mappings().all()]
            logger.info("[SendFileNode] all_in_namespace: found %d file(s)", len(rows))
            return rows
        except Exception as exc:
            logger.error("[SendFileNode] DB error (all_in_namespace): %s", exc)
            return []

    async def _search_by_name(
        self, db, user_id, namespace_id, search_term: str
    ) -> List[dict]:
        """
        Поиск файлов по точному имени, затем стемм-поиск как fallback.
        Возвращает список строк.
        """
        logger.info("[SendFileNode] Name search: term=%r", search_term)
        base_params = {"user_id": user_id, "namespace_id": namespace_id, "limit": SEND_FILE_LIMIT}

        sql, extra_params = _build_exact_name_sql(search_term)
        try:
            result = await db.execute(text(sql), {**base_params, **extra_params})
            rows = [dict(r) for r in result.mappings().all()]
        except Exception as exc:
            logger.error("[SendFileNode] DB error (exact name search): %s", exc)
            return []

        if rows:
            return rows

        logger.info("[SendFileNode] Exact match not found, trying stem search")
        sql, extra_params = _build_stem_search_sql(search_term)
        try:
            result = await db.execute(text(sql), {**base_params, **extra_params})
            return [dict(r) for r in result.mappings().all()]
        except Exception as exc:
            logger.error("[SendFileNode] DB error (stem name search): %s", exc)
            return []

    async def _search_by_topic(
        self, db, user_id, namespace_id, search_term: str
    ) -> List[dict]:
        """
        Семантический поиск файлов по теме/содержанию.
        При неудаче vector+rerank — fallback только если embedding недоступен.
        """
        logger.info("[SendFileNode] Topic search: term=%r", search_term)

        if self.embedding_service and search_term:
            rows = await self._vector_search(db, user_id, namespace_id, search_term)
            if rows:
                rows = await self._llm_rerank(rows, search_term)
            if rows:
                logger.info("[SendFileNode] Topic (vector+LLM) found %d file(s)", len(rows))
                return rows
            logger.info("[SendFileNode] Vector topic search: no match found")
            return []

        # Embedding недоступен — fallback на ILIKE-поиск по имени
        logger.info("[SendFileNode] Embedding unavailable, falling back to name search")
        sql, extra_params = _build_stem_search_sql(search_term)
        base_params = {"user_id": user_id, "namespace_id": namespace_id, "limit": SEND_FILE_LIMIT}
        try:
            result = await db.execute(text(sql), {**base_params, **extra_params})
            return [dict(r) for r in result.mappings().all()]
        except Exception as exc:
            logger.error("[SendFileNode] DB error (name fallback): %s", exc)
            return []

    async def _search_by_content(
        self, db, user_id, namespace_id, search_term: str
    ) -> List[dict]:
        """
        Поиск файлов по буквальному вхождению текста в чанки (ILIKE по chunk_text).
        Fallback — семантический поиск, затем поиск по имени.
        """
        logger.info("[SendFileNode] Content search: term=%r", search_term)
        base_params = {"user_id": user_id, "namespace_id": namespace_id, "limit": SEND_FILE_LIMIT}

        if search_term:
            try:
                result = await db.execute(
                    text(FIND_FILE_BY_CONTENT_SQL),
                    {**base_params, "content_pattern": f"%{search_term}%"},
                )
                rows = [dict(r) for r in result.mappings().all()]
            except Exception as exc:
                logger.error("[SendFileNode] DB error (content search): %s", exc)
                rows = []

            if rows:
                logger.info("[SendFileNode] Content found %d file(s)", len(rows))
                return rows
            logger.info("[SendFileNode] Content search found nothing, falling back to vector")

        if self.embedding_service and search_term:
            rows = await self._vector_search(db, user_id, namespace_id, search_term)
            if rows:
                return rows

        sql, extra_params = _build_stem_search_sql(search_term)
        try:
            result = await db.execute(text(sql), {**base_params, **extra_params})
            return [dict(r) for r in result.mappings().all()]
        except Exception as exc:
            logger.error("[SendFileNode] DB error (name fallback after content): %s", exc)
            return []

    async def _vector_search(
        self, db, user_id, namespace_id, search_term: str
    ) -> List[dict]:
        """
        Выполняет семантический поиск файлов.
        Возвращает только строки с релевантностью >= _MIN_VECTOR_RELEVANCE.
        """
        try:
            embedding = await self.embedding_service.generate_query_embedding(search_term)
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            # Первое значимое слово запроса для приоритизации ключевых чанков в сниппете
            first_kw = next((w for w in search_term.split() if len(w) >= 3), search_term)
            result = await db.execute(
                text(FIND_FILE_BY_TOPIC_SQL),
                {
                    "query_embedding": embedding_str,
                    "user_id": user_id,
                    "namespace_id": namespace_id,
                    "min_relevance": _MIN_VECTOR_RELEVANCE,
                    "limit": SEND_FILE_LIMIT,
                    "kw_pattern": f"%{first_kw}%",
                },
            )
            rows = [dict(r) for r in result.mappings().all()]
            logger.info(
                "[SendFileNode] Vector search returned %d file(s) above threshold %.2f",
                len(rows), _MIN_VECTOR_RELEVANCE,
            )
            if not rows:
                return rows

            logger.info("[SendFileNode] After dedup: %d unique file(s)", len(rows))

            return rows
        except Exception as exc:
            logger.error("[SendFileNode] Vector search error: %s", exc)
            try:
                await db.rollback()
            except Exception:
                pass
            return []
