"""RouterNode — определяет намерение пользователя через LLMIntentClassifier."""
import re
import logging
from typing import Optional, List, Dict

from langchain_core.runnables import RunnableConfig
from sqlalchemy import text

from app.graph.state import AskState
from app.core.enums import IntentType
from app.infrastructure.repositories.vector_queries import LIST_FILES_SQL
from app.services.llm_intent_classifier import LLMIntentClassifier

logger = logging.getLogger(__name__)

# Паттерн для распознавания URL (детерминированный — LLM здесь не нужен)
URL_PATTERN = re.compile(
    r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b'
    r'[-a-zA-Z0-9()@:%_\+.~#?&//=]*',
    re.IGNORECASE,
)

# Паттерн для явных файловых запросов — используется как guard для send_file
_FILE_REQUEST_TRIGGERS = re.compile(
    r'(?:скинь|отправь|дай|пришли|покажи файл|найди файл|найди документ|найди реферат'
    r'|найди конспект|найди презентацию|скачать|есть ли у меня файл'
    r'|есть ли.*(?:файл|конспект|реферат|документ))',
    re.IGNORECASE,
)

# Паттерн запросов на извлечение содержимого из файла — даже если есть «скинь»,
# это rag_query («скинь требования из этого файла», «что написано в файле»)
_CONTENT_EXTRACTION_TRIGGERS = re.compile(
    r'из\s+(?:этого\s+|данного\s+|прикреплённого\s+|приложенного\s+)?(?:файла|документа|pdf|доки?)'
    r'|что\s+(?:написано|есть|говорится|там)\s+(?:в|про|о)'
    r'|содержимое\s+файла'
    r'|вытащи|извлеки|достань',
    re.IGNORECASE,
)

# Количество сообщений в истории для поиска контекста
HISTORY_SCAN_LIMIT = 5


class RouterNode:
    """
    Определяет намерение пользователя.

    Два режима работы:
    1. Override mode: если override_intent задан — использует его напрямую.
    2. Auto mode: один LLM-вызов (LLMIntentClassifier) возвращает intent,
       search_query, namespace_hint, search_mode в JSON.

    Дополнительно:
    - Резолвит имя пространства из текста в namespace_id через БД.
    - URL-детектор остаётся детерминированным (regex надёжнее LLM для URL).
    """

    def __init__(self, llm_intent_classifier: LLMIntentClassifier) -> None:
        self.classifier = llm_intent_classifier

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _extract_url(self, text: str) -> Optional[str]:
        """Извлекает первый URL из текста."""
        match = URL_PATTERN.search(text)
        return match.group(0) if match else None

    def _extract_url_from_history(self, history: List[Dict]) -> Optional[str]:
        for msg in reversed(history[-HISTORY_SCAN_LIMIT:]):
            url = self._extract_url(msg.get("text", ""))
            if url:
                logger.info("[Router] Found URL in history: %s", url)
                return url
        return None

    def _extract_file_id_from_history(self, history: List[Dict]) -> Optional[int]:
        for msg in reversed(history[-HISTORY_SCAN_LIMIT:]):
            file_ids = msg.get("file_ids") or []
            if file_ids:
                logger.info("[Router] Found file_id in history: %d", file_ids[0])
                return file_ids[0]
        return None

    # Паттерн для извлечения имён файлов из текста сообщений (в кавычках-ёлочках или обычных)
    _FILENAME_PATTERN = re.compile(
        r'[«""]([^»""]{1,120}\.[a-zA-Z]{2,6})[»""]',
        re.IGNORECASE,
    )
    # Паттерн для поиска файла в ответах ассистента: «файл.md» был создан / отредактирован
    _ASSISTANT_FILE_ACTION_PATTERN = re.compile(
        r'[«""]([^»""]{1,120}\.[a-zA-Z]{2,6})[»""]\s+(?:был\s+)?(?:создан|отредактирован|перемещён|переименован)',
        re.IGNORECASE,
    )

    def _extract_active_file_context(
        self, history: List[Dict], history_file_id: Optional[int]
    ) -> Optional[str]:
        """
        Строит строку контекста вида «filename.pdf (пространство: Inbox)»
        для передачи в LLM-классификатор.

        Если history_file_id задан — ищем файл по этому id.
        Иначе — ищем последний файл из ответов ассистента (создан/отредактирован),
        чтобы «этого файла» корректно резолвилось без прикреплённого файла.
        """
        filename: Optional[str] = None
        namespace_name: Optional[str] = None

        recent = list(reversed(history[-HISTORY_SCAN_LIMIT:]))

        if history_file_id:
            # Ищем имя файла: сначала в сообщениях с file_ids, затем в соседних
            for msg in recent:
                file_ids = msg.get("file_ids") or []
                if history_file_id in file_ids:
                    msg_text = msg.get("text") or ""
                    m = self._FILENAME_PATTERN.search(msg_text)
                    if m:
                        filename = m.group(1)
                    break

            if not filename:
                for msg in recent:
                    msg_text = msg.get("text") or ""
                    m = self._FILENAME_PATTERN.search(msg_text)
                    if m:
                        filename = m.group(1)
                        break
        else:
            # Ищем последний файл упомянутый в ответах ассистента
            for msg in recent:
                if msg.get("role") != "assistant":
                    continue
                msg_text = msg.get("text") or ""
                m = self._ASSISTANT_FILE_ACTION_PATTERN.search(msg_text)
                if m:
                    filename = m.group(1)
                    break

        # Ищем пространство: «в пространстве X», «в пространство X»
        ns_pattern = re.compile(r'в\s+пространств[еоу]\s+[«""]?([^»""«,.\n]{1,50})[»""]?', re.IGNORECASE)
        for msg in recent:
            msg_text = msg.get("text") or ""
            m = ns_pattern.search(msg_text)
            if m:
                namespace_name = m.group(1).strip()
                break

        if not filename:
            return None

        if namespace_name:
            return f"{filename} (пространство: {namespace_name})"
        return filename

    def _is_url_only(self, text: str) -> bool:
        return bool(URL_PATTERN.fullmatch(text.strip()))

    async def _resolve_namespace_id(
        self, db, user_id: int, name: str
    ) -> Optional[int]:
        """Ищет namespace по имени (case-insensitive) для данного пользователя."""
        try:
            result = await db.execute(
                text(
                    "SELECT id FROM namespaces "
                    "WHERE user_id = :user_id AND LOWER(name) = LOWER(:name) "
                    "LIMIT 1"
                ),
                {"user_id": user_id, "name": name},
            )
            row = result.mappings().first()
            return row["id"] if row else None
        except Exception as exc:
            logger.warning("[Router] Failed to resolve namespace '%s': %s", name, exc)
            return None

    async def _fetch_namespace_filenames(
        self, db, user_id: int, namespace_id: int, limit: int = 30
    ) -> Optional[str]:
        """
        Возвращает строку с именами файлов в пространстве для передачи в классификатор.
        Например: 'Совет1.md, Совет2.md, Совет3.md'
        """
        try:
            result = await db.execute(
                text(
                    "SELECT COALESCE(uf.custom_title, f.media_metadata->>'title', "
                    "f.source_url, f.file_path, 'Document') AS filename "
                    "FROM user_files uf "
                    "JOIN files f ON f.id = uf.file_id "
                    "WHERE uf.user_id = :user_id AND uf.namespace_id = :namespace_id "
                    "ORDER BY uf.created_at DESC LIMIT :limit"
                ),
                {"user_id": user_id, "namespace_id": namespace_id, "limit": limit},
            )
            rows = result.mappings().all()
            if not rows:
                return None
            return ", ".join(r["filename"] for r in rows)
        except Exception as exc:
            logger.warning(
                "[Router] Failed to fetch filenames for namespace_id=%d: %s", namespace_id, exc
            )
            return None

    # ------------------------------------------------------------------
    # Основной метод
    # ------------------------------------------------------------------

    async def run(self, state: AskState, config: RunnableConfig) -> AskState:
        question = state.get("question", "").strip()
        file_content = state.get("file_content")
        filename = state.get("filename")
        history = state.get("history") or []
        override_intent = state.get("override_intent")

        detected_url = self._extract_url(question)
        history_url = self._extract_url_from_history(history) if not detected_url else None
        history_file_id = (
            state.get("history_file_id") or self._extract_file_id_from_history(history)
        )
        if not detected_url and history_url:
            detected_url = history_url

        if override_intent:
            return await self._handle_override(
                state, override_intent, detected_url, history_file_id,
                file_content, filename, config,
            )

        return await self._handle_auto(
            state, question, detected_url, history_file_id,
            file_content, filename, config,
        )

    # ------------------------------------------------------------------
    # Override mode
    # ------------------------------------------------------------------

    async def _handle_override(
        self,
        state: AskState,
        intent: str,
        detected_url: Optional[str],
        history_file_id: Optional[int],
        file_content: Optional[bytes],
        filename: Optional[str],
        config: RunnableConfig,
    ) -> AskState:
        logger.info("[Router] Override mode: intent=%s", intent)

        if intent == IntentType.SUMMARIZE and not detected_url and not file_content and not history_file_id:
            return {
                **state,
                "intent": IntentType.RAG_QUERY,
                "answer": "Не нашёл что суммаризировать. Отправьте ссылку или файл.",
                "agent_steps": state.get("agent_steps", []) + [
                    "[Router] Override: summarize, but no content found"
                ],
            }

        if intent == IntentType.INDEX_URL and not detected_url:
            return {
                **state,
                "intent": IntentType.RAG_QUERY,
                "answer": "Не нашёл ссылку для сохранения.",
                "agent_steps": state.get("agent_steps", []) + [
                    "[Router] Override: index_url, but no URL found"
                ],
            }

        if intent == IntentType.SAVE_FILE and not file_content:
            return {
                **state,
                "intent": IntentType.RAG_QUERY,
                "answer": "Не нашёл файл для сохранения.",
                "agent_steps": state.get("agent_steps", []) + [
                    "[Router] Override: save_file, but no file attached"
                ],
            }

        namespace_id = await self._maybe_resolve_namespace(state, config)
        return {
            **state,
            "intent": intent,
            "detected_url": detected_url,
            "history_file_id": history_file_id,
            "namespace_id": namespace_id,
            "agent_steps": state.get("agent_steps", []) + [
                f"[Router] Override: {intent}"
            ],
        }

    # ------------------------------------------------------------------
    # Auto mode
    # ------------------------------------------------------------------

    async def _handle_auto(
        self,
        state: AskState,
        question: str,
        detected_url: Optional[str],
        history_file_id: Optional[int],
        file_content: Optional[bytes],
        filename: Optional[str],
        config: RunnableConfig,
    ) -> AskState:
        history = state.get("history") or []
        # Детерминированные случаи обрабатываем без LLM
        if file_content and filename:
            return await self._handle_file_upload(
                state, question, detected_url, file_content, filename, config
            )

        if detected_url and self._is_url_only(question):
            logger.info("[Router] Auto: index_url (URL only)")
            namespace_id = await self._maybe_resolve_namespace(state, config)
            return {
                **state,
                "intent": IntentType.INDEX_URL,
                "detected_url": detected_url,
                "namespace_id": namespace_id,
                "agent_steps": state.get("agent_steps", []) + [
                    "[Router] Auto: index_url (URL only)"
                ],
            }

        # LLM-классификация
        active_file_ctx = self._extract_active_file_context(history, history_file_id)
        if active_file_ctx:
            logger.info("[Router] Active file context for classifier: %r", active_file_ctx)
        # Сокращаем историю если вопрос содержит несколько явных объектов (запятые + союзы)
        # — снижает риск того что LLM добавит лишние действия из истории
        has_explicit_list = question.count(",") >= 2 or bool(
            re.search(r'\bи\s+(?:создай|удали|перемести|переименуй)\b', question, re.IGNORECASE)
        )
        history_limit = 2 if has_explicit_list else None

        # Коррекция ("нет, ...", "не то, ...") — пользователь исправляет предыдущий ответ.
        # Ограничиваем историю до 2 сообщений чтобы LLM не тащил контекст из прошлых операций.
        _CORRECTION_PATTERN = re.compile(
            r'^(?:нет[,.]?\s|не\s+то[,.]?\s|не\s+так[,.]?\s|исправь[,.]?\s)',
            re.IGNORECASE,
        )
        if _CORRECTION_PATTERN.match(question.strip()):
            history_limit = min(history_limit or 999, 2)
            logger.info("[Router] Correction detected — limiting history to 2")

        # Когда пользователь ссылается на «этот файл» без явного имени —
        # убеждаемся что active_file_ctx заполнен (сканируем ответы ассистента)
        _THIS_FILE_PATTERN = re.compile(
            r'этот\s+файл|этого\s+файла|этому\s+файлу|этот\s+документ|этого\s+документа',
            re.IGNORECASE,
        )
        if _THIS_FILE_PATTERN.search(question) and not active_file_ctx:
            active_file_ctx = self._extract_active_file_context(history, None)
            if active_file_ctx:
                logger.info("[Router] Resolved 'этого файла' → %r", active_file_ctx)

        # Если выбрано пространство и вопрос ссылается на "все/каждый файл" —
        # передаём реальные имена файлов чтобы LLM не выдумывал их
        ns_files_ctx: Optional[str] = None
        _ALL_FILES_PATTERN = re.compile(
            r'каждый|каждую|все\s+файл|всех\s+файл|этого\s+пространства|в\s+этом\s+пространстве',
            re.IGNORECASE,
        )
        if _ALL_FILES_PATTERN.search(question):
            ns_id_for_files = state.get("namespace_id")
            if ns_id_for_files:
                configurable = config.get("configurable") or {}
                db = configurable.get("async_db")
                user_id = state.get("user_id")
                if db and user_id is not None:
                    ns_files_ctx = await self._fetch_namespace_filenames(db, user_id, ns_id_for_files)
                    if ns_files_ctx:
                        logger.info("[Router] Namespace files context: %r", ns_files_ctx[:100])
                        # Когда файлы известны явно — история только мешает, ограничиваем до 1
                        history_limit = 1

        parsed = await self.classifier.parse(
            question,
            has_file=bool(file_content),
            has_url=bool(detected_url),
            history=history,
            active_file_context=active_file_ctx,
            history_limit=history_limit,
            namespace_files_context=ns_files_ctx,
        )
        logger.info(
            "[Router] LLM parsed: intent=%s search_query=%r namespace_hint=%r search_mode=%s",
            parsed.intent, parsed.search_query, parsed.namespace_hint, parsed.search_mode,
        )

        # Резолвим namespace_id: приоритет — из state (пришёл с фронта),
        # затем — hint из LLM, резолвим через БД
        namespace_id = state.get("namespace_id")
        namespace_name_hint = parsed.namespace_hint
        if namespace_id is None and namespace_name_hint:
            configurable = config.get("configurable") or {}
            db = configurable.get("async_db")
            user_id = state.get("user_id")
            if db and user_id is not None:
                resolved = await self._resolve_namespace_id(db, user_id, namespace_name_hint)
                if resolved is not None:
                    namespace_id = resolved
                    logger.info(
                        "[Router] Resolved namespace '%s' → id=%d",
                        namespace_name_hint, resolved,
                    )
                else:
                    logger.info(
                        "[Router] Namespace '%s' not found for user_id=%s",
                        namespace_name_hint, user_id,
                    )

        intent = parsed.intent
        search_query = parsed.search_query
        
        # Fallback: если namespace_hint=None но search_query есть и intent=edit_file,
        # пытаемся резолвить namespace по search_query (например: "В начало каждого файла из пространства Пурум...")
        if (namespace_id is None and not namespace_name_hint and 
            intent == IntentType.EDIT_FILE and search_query):
            configurable = config.get("configurable") or {}
            db = configurable.get("async_db")
            user_id = state.get("user_id")
            if db and user_id is not None:
                resolved = await self._resolve_namespace_id(db, user_id, search_query)
                if resolved is not None:
                    namespace_id = resolved
                    namespace_name_hint = search_query
                    logger.info(
                        "[Router] Resolved namespace from search_query '%s' → id=%d",
                        search_query, resolved,
                    )
        search_file_ids = state.get("search_file_ids")
        base_steps = state.get("agent_steps", [])

        # Мульти-действие: разворачиваем список действий (после base_steps)
        if parsed.intent == "multi_action" and parsed.actions:
            return await self._handle_multi_action(state, parsed.actions, base_steps, config)

        # Если пользователь выбрал файлы или пространство, а классификатор вернул general_chat — ищем по контексту
        has_file_scope = bool(search_file_ids)
        has_namespace_scope = namespace_id is not None
        if (has_file_scope or has_namespace_scope) and intent == IntentType.GENERAL_CHAT:
            intent = IntentType.RAG_QUERY
            if not (search_query and search_query.strip()):
                search_query = (question or "").strip() or "содержимое документов"
            logger.info(
                "[Router] Override: general_chat → rag_query (files=%s, namespace=%s), search_query=%r",
                has_file_scope, has_namespace_scope, search_query[:80] if search_query else None,
            )

        if intent == IntentType.SUMMARIZE:
            return self._route_summarize(
                state, detected_url, namespace_id,
                namespace_name_hint, search_query, base_steps,
                history_file_id=history_file_id,
            )

        if intent == IntentType.INDEX_URL:
            if detected_url:
                return {
                    **state,
                    "intent": IntentType.INDEX_URL,
                    "detected_url": detected_url,
                    "namespace_id": namespace_id,
                    "namespace_name_hint": namespace_name_hint,
                    "search_query": search_query,
                    "agent_steps": base_steps + ["[Router] Auto: index_url (LLM)"],
                }
            return {
                **state,
                "intent": IntentType.RAG_QUERY,
                "answer": "Что сохранить? Отправьте ссылку или файл.",
                "agent_steps": base_steps + ["[Router] Auto: index_url requested, no URL"],
            }

        if intent == IntentType.LIST_FILES:
            return {
                **state,
                "intent": IntentType.LIST_FILES,
                "sql_query": LIST_FILES_SQL,
                "namespace_id": namespace_id,
                "namespace_name_hint": namespace_name_hint,
                "agent_steps": base_steps + ["[Router] Auto: list_files (LLM)"],
            }

        if intent == IntentType.SEND_FILE:
            # Guard 1: если в вопросе есть фраза «из этого файла» / «из документа» —
            # пользователь хочет контент, а не сам файл.
            if _CONTENT_EXTRACTION_TRIGGERS.search(question):
                logger.info(
                    "[Router] Guard: send_file → rag_query (content extraction phrase: %r)",
                    question[:80],
                )
                intent = IntentType.RAG_QUERY
                search_query = question
            # Guard 2: если вопрос вообще не содержит явных триггеров отправки файла
            elif not _FILE_REQUEST_TRIGGERS.search(question):
                logger.info(
                    "[Router] Guard: send_file → rag_query (no file triggers in question: %r)",
                    question[:80],
                )
                intent = IntentType.RAG_QUERY
                search_query = question
            else:
                return {
                    **state,
                    "intent": IntentType.SEND_FILE,
                    "history_file_id": history_file_id,
                    "namespace_id": namespace_id,
                    "namespace_name_hint": namespace_name_hint,
                    "search_query": search_query,
                    "send_file_search_mode": parsed.search_mode or "by_topic",
                    "agent_steps": base_steps + ["[Router] Auto: send_file (LLM)"],
            }

        if intent == IntentType.SAVE_FILE:
            # Guard: нет загруженного файла, но есть search_query + namespace → это move_file
            # (пользователь хочет переместить существующий файл, а не загрузить новый)
            if not state.get("file_content") and not parsed.entity_content and parsed.search_query and namespace_id:
                move_name = parsed.entity_name or parsed.search_query
                logger.info(
                    "[Router] save_file → move_file (no upload, existing file '%s' → namespace_id=%s)",
                    move_name, namespace_id,
                )
                return {
                    **state,
                    "intent": IntentType.MOVE_FILE,
                    "namespace_id": namespace_id,
                    "namespace_name_hint": namespace_name_hint,
                    "search_query": parsed.search_query,
                    "entity_name": move_name,
                    "agent_steps": base_steps + [
                        f"[Router] Auto: save_file → move_file (no upload, ns={namespace_name_hint})"
                    ],
                }
            # Если реального файла нет, но LLM вернул entity_content — создаём файл из текста
            if not state.get("file_content") and parsed.entity_content:
                logger.info(
                    "[Router] save_file → create_file (no upload, entity_content present)"
                )
                return {
                    **state,
                    "intent": IntentType.CREATE_FILE,
                    "namespace_id": namespace_id,
                    "namespace_name_hint": namespace_name_hint,
                    "entity_name": parsed.entity_name,
                    "entity_content": parsed.entity_content,
                    "agent_steps": base_steps + ["[Router] Auto: save_file → create_file (no upload)"],
                }
            return {
                **state,
                "intent": IntentType.SAVE_FILE,
                "namespace_id": namespace_id,
                "namespace_name_hint": namespace_name_hint,
                "agent_steps": base_steps + ["[Router] Auto: save_file (LLM)"],
            }

        if intent == IntentType.GENERAL_CHAT:
            return {
                **state,
                "intent": IntentType.GENERAL_CHAT,
                "search_result": [],
                "namespace_id": namespace_id,
                "namespace_name_hint": namespace_name_hint,
                "agent_steps": base_steps + ["[Router] Auto: general_chat (LLM)"],
            }

        if intent == IntentType.SAVE_SUMMARY:
            return self._route_save_summary(
                state, namespace_id, namespace_name_hint,
                parsed.entity_name, base_steps, history,
            )

        # CRUD-интенты
        if intent in (
            IntentType.CREATE_NAMESPACE,
            IntentType.DELETE_NAMESPACE,
            IntentType.EDIT_NAMESPACE,
            IntentType.MOVE_FILE,
            IntentType.CREATE_FILE,
            IntentType.DELETE_FILE,
            IntentType.EDIT_FILE,
        ):
            entity_name = parsed.entity_name
            # Для create_namespace LLM иногда кладёт название в namespace_hint вместо entity_name
            if intent == IntentType.CREATE_NAMESPACE and not entity_name and namespace_name_hint:
                entity_name = namespace_name_hint
                namespace_name_hint = None
                namespace_id = None
                logger.info(
                    "[Router] create_namespace: moved namespace_hint '%s' → entity_name",
                    entity_name,
                )
            return {
                **state,
                "intent": intent,
                "namespace_id": namespace_id,
                "namespace_name_hint": namespace_name_hint,
                "search_query": search_query,
                "entity_name": entity_name,
                "entity_description": parsed.entity_description,
                "entity_content": parsed.entity_content,
                "agent_steps": base_steps + [f"[Router] Auto: {intent} (LLM)"],
            }

        # RAG_QUERY (default)
        # Если пользователь спрашивает о конкретном файле из истории («этот файл», «его»),
        # скоупим поиск на него. search_file_ids из state имеет приоритет (явный выбор файлов).
        effective_search_file_ids = state.get("search_file_ids")
        if not effective_search_file_ids and history_file_id:
            effective_search_file_ids = [history_file_id]
            logger.info(
                "[Router] rag_query: scoping to history_file_id=%d", history_file_id
            )

        return {
            **state,
            "intent": IntentType.RAG_QUERY,
            "detected_url": detected_url,
            "namespace_id": namespace_id,
            "namespace_name_hint": namespace_name_hint,
            "search_query": search_query,
            "search_file_ids": effective_search_file_ids,
            "agent_steps": base_steps + ["[Router] Auto: rag_query (LLM)"],
        }

    # ------------------------------------------------------------------
    # Вспомогательные маршруты
    # ------------------------------------------------------------------

    async def _inbox_namespace_if_unset(
        self,
        namespace_id: Optional[int],
        namespace_name_hint: Optional[str],
        config: RunnableConfig,
        user_id: Optional[int],
    ) -> tuple[Optional[int], Optional[str]]:
        """
        Файл из чата без явного пространства в тексте — кладём в Inbox
        (не используем namespace_id из query/UI).
        """
        if namespace_id is not None:
            return namespace_id, namespace_name_hint
        configurable = config.get("configurable") or {}
        db = configurable.get("async_db")
        if not db or user_id is None:
            return None, namespace_name_hint
        inbox_id = await self._resolve_namespace_id(db, user_id, "Inbox")
        if inbox_id is not None:
            logger.info("[Router] Chat file upload: default → Inbox (id=%d)", inbox_id)
            return inbox_id, "Inbox"
        logger.warning("[Router] Chat file upload: Inbox not found for user_id=%s", user_id)
        return None, namespace_name_hint

    async def _handle_file_upload(
        self,
        state: AskState,
        question: str,
        detected_url: Optional[str],
        file_content: bytes,
        filename: str,
        config: RunnableConfig,
    ) -> AskState:
        """Файл приложен — определяем: суммаризация или сохранение."""
        # Не подставляем namespace из query: только явный текст пользователя или Inbox
        namespace_id: Optional[int] = None
        namespace_name_hint: Optional[str] = None
        user_id = state.get("user_id")
        if question:
            history = state.get("history") or []
            parsed = await self.classifier.parse(
                question,
                has_file=True,
                has_url=bool(detected_url),
                history=history,
            )
            # Явное пространство в сообщении (LLM вытащил hint)
            if parsed.namespace_hint:
                configurable = config.get("configurable") or {}
                db = configurable.get("async_db")
                if db and user_id is not None:
                    resolved = await self._resolve_namespace_id(db, user_id, parsed.namespace_hint)
                    if resolved is not None:
                        namespace_id = resolved
                        namespace_name_hint = parsed.namespace_hint
                        logger.info(
                            "[Router] File upload: resolved namespace '%s' → id=%d",
                            parsed.namespace_hint, resolved,
                        )
                    else:
                        logger.info(
                            "[Router] File upload: namespace '%s' not found, creating automatically",
                            parsed.namespace_hint,
                        )
                        # Пространство явно указано, но не найдено — создаём автоматически
                        namespace_service = configurable.get("namespace_service")
                        if namespace_service is not None:
                            try:
                                new_ns = await namespace_service.namespace_repository.create(
                                    name=parsed.namespace_hint,
                                    user_id=user_id,
                                    description=None,
                                )
                                await namespace_service.db.commit()
                                namespace_id = new_ns.id
                                namespace_name_hint = parsed.namespace_hint
                                state = {**state, "namespace_created": True}
                                logger.info(
                                    "[Router] File upload: auto-created namespace '%s' → id=%d",
                                    parsed.namespace_hint, new_ns.id,
                                )
                            except Exception as exc:
                                logger.warning(
                                    "[Router] File upload: failed to auto-create namespace '%s': %s",
                                    parsed.namespace_hint, exc,
                                )
            if parsed.intent == IntentType.SUMMARIZE:
                logger.info("[Router] Auto: summarize (file, LLM)")
                namespace_id, namespace_name_hint = await self._inbox_namespace_if_unset(
                    namespace_id, namespace_name_hint, config, user_id
                )
                return {
                    **state,
                    "intent": IntentType.SUMMARIZE,
                    "detected_url": detected_url,
                    "namespace_id": namespace_id,
                    "namespace_name_hint": namespace_name_hint,
                    "agent_steps": state.get("agent_steps", []) + [
                        "[Router] Auto: summarize (file, LLM)"
                    ],
                }
            if parsed.intent == IntentType.RAG_QUERY:
                if not parsed.search_query:
                    # Нет конкретного поискового запроса — вопрос общего характера о файле,
                    # обрабатываем как суммаризацию
                    logger.info("[Router] Auto: summarize (file, rag_query with no search_query)")
                    namespace_id, namespace_name_hint = await self._inbox_namespace_if_unset(
                        namespace_id, namespace_name_hint, config, user_id
                    )
                    return {
                        **state,
                        "intent": IntentType.SUMMARIZE,
                        "detected_url": detected_url,
                        "namespace_id": namespace_id,
                        "namespace_name_hint": namespace_name_hint,
                        "agent_steps": state.get("agent_steps", []) + [
                            "[Router] Auto: summarize (file, rag_query with no search_query)"
                        ],
                    }
                logger.info(
                    "[Router] Auto: save_file (filename=%s) + pending rag_query=%r",
                    filename, parsed.search_query,
                )
                namespace_id, namespace_name_hint = await self._inbox_namespace_if_unset(
                    namespace_id, namespace_name_hint, config, user_id
                )
                return {
                    **state,
                    "intent": IntentType.SAVE_FILE,
                    "detected_url": detected_url,
                    "namespace_id": namespace_id,
                    "namespace_name_hint": namespace_name_hint,
                    "search_query": parsed.search_query,
                    "agent_steps": state.get("agent_steps", []) + [
                        f"[Router] Auto: save_file (filename={filename}, pending rag_query)"
                    ],
                }
        namespace_id, namespace_name_hint = await self._inbox_namespace_if_unset(
            namespace_id, namespace_name_hint, config, user_id
        )
        logger.info("[Router] Auto: save_file (filename=%s)", filename)
        return {
            **state,
            "intent": IntentType.SAVE_FILE,
            "detected_url": detected_url,
            "namespace_id": namespace_id,
            "namespace_name_hint": namespace_name_hint,
            "agent_steps": state.get("agent_steps", []) + ["[Router] Auto: save_file"],
        }

    async def _handle_multi_action(
        self,
        state: AskState,
        actions: List,
        base_steps: list,
        config: RunnableConfig,
    ) -> AskState:
        """Резолвит namespace для каждого действия и складывает список в pending_actions."""
        configurable = config.get("configurable") or {}
        db = configurable.get("async_db")
        user_id = state.get("user_id")

        resolved_actions = []
        pending_ns_names: list[str] = []  # имена пространств из create_namespace в этом же батче
        for action in actions:
            ns_hint = action.namespace_hint
            ns_id: Optional[int] = None
            entity_name = action.entity_name

            if ns_hint and db and user_id is not None:
                ns_id = await self._resolve_namespace_id(db, user_id, ns_hint)

            # Для create_namespace — имя может попасть в namespace_hint вместо entity_name
            if action.intent == IntentType.CREATE_NAMESPACE and not entity_name and ns_hint:
                entity_name = ns_hint
                ns_id = None
                ns_hint = None

            # Запоминаем имена создаваемых пространств для последующих действий в батче
            if action.intent == IntentType.CREATE_NAMESPACE and entity_name:
                pending_ns_names.append(entity_name)

            # Если create_file без namespace_hint, но в этом батче создаётся пространство — используем его
            if (
                action.intent == IntentType.CREATE_FILE
                and ns_id is None
                and not ns_hint
                and pending_ns_names
            ):
                ns_hint = pending_ns_names[-1]
                logger.info(
                    "[Router] Inferred namespace '%s' for create_file from batch create_namespace",
                    ns_hint,
                )

            resolved_actions.append({
                "intent": action.intent,
                "namespace_id": ns_id,
                "namespace_name_hint": ns_hint,
                "search_query": action.search_query,
                "entity_name": entity_name,
                "entity_description": action.entity_description,
                "entity_content": action.entity_content,
            })

        logger.info("[Router] multi_action: %d actions resolved", len(resolved_actions))
        return {
            **state,
            "pending_actions": resolved_actions,
            "agent_steps": base_steps + [
                f"[Router] Auto: multi_action ({len(resolved_actions)} actions)"
            ],
        }

    def _route_summarize(
        self,
        state: AskState,
        detected_url: Optional[str],
        namespace_id: Optional[int],
        namespace_name_hint: Optional[str],
        search_query: Optional[str],
        base_steps: list,
        history_file_id: Optional[int] = None,
    ) -> AskState:
        if detected_url:
            return {
                **state,
                "intent": IntentType.SUMMARIZE,
                "detected_url": detected_url,
                "namespace_id": namespace_id,
                "namespace_name_hint": namespace_name_hint,
                "agent_steps": base_steps + ["[Router] Auto: summarize (URL, LLM)"],
            }
        ids = state.get("search_file_ids") or []
        file_id = state.get("history_file_id") or history_file_id or (ids[0] if ids else None)
        if file_id is not None:
            return {
                **state,
                "intent": IntentType.SUMMARIZE,
                "detected_url": None,
                "history_file_id": file_id,
                "namespace_id": namespace_id,
                "namespace_name_hint": namespace_name_hint,
                "agent_steps": base_steps + [
                    f"[Router] Auto: summarize (file_id={file_id}, LLM)"
                ],
            }
        # Нечего суммаризировать — болтовня
        return {
            **state,
            "intent": IntentType.GENERAL_CHAT,
            "search_result": [],
            "namespace_id": namespace_id,
            "namespace_name_hint": namespace_name_hint,
            "search_query": search_query,
            "agent_steps": base_steps + ["[Router] Auto: general_chat (summarize, no content)"],
        }

    def _route_save_summary(
        self,
        state: AskState,
        namespace_id: Optional[int],
        namespace_name_hint: Optional[str],
        entity_name: Optional[str],
        base_steps: list,
        history: List[Dict],
    ) -> AskState:
        """Находит последнее сообщение ассистента в истории и роутит как create_file."""
        last_assistant_text: Optional[str] = None
        for msg in reversed(history):
            role = (msg.get("role") or "").strip().lower()
            text = (msg.get("text") or "").strip()
            if role == "assistant" and text:
                last_assistant_text = text
                break

        if not last_assistant_text:
            logger.info("[Router] save_summary: no assistant message in history")
            return {
                **state,
                "intent": IntentType.GENERAL_CHAT,
                "search_result": [],
                "namespace_id": namespace_id,
                "namespace_name_hint": namespace_name_hint,
                "answer": "Нечего сохранять — в истории нет предыдущего ответа.",
                "agent_steps": base_steps + ["[Router] save_summary: no assistant message"],
            }

        logger.info(
            "[Router] save_summary → create_file (namespace_id=%s, content_len=%d)",
            namespace_id, len(last_assistant_text),
        )
        return {
            **state,
            "intent": IntentType.CREATE_FILE,
            "namespace_id": namespace_id,
            "namespace_name_hint": namespace_name_hint,
            "entity_name": entity_name,
            "entity_content": last_assistant_text,
            "agent_steps": base_steps + [
                f"[Router] save_summary → create_file (namespace={namespace_name_hint})"
            ],
        }

    async def _maybe_resolve_namespace(
        self, state: AskState, config: RunnableConfig
    ) -> Optional[int]:
        """Возвращает namespace_id из state (если уже есть) без дополнительного резолвинга."""
        return state.get("namespace_id")
