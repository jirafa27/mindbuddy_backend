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
        parsed = await self.classifier.parse(
            question,
            has_file=bool(file_content),
            has_url=bool(detected_url),
            history=history,
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
        search_file_ids = state.get("search_file_ids")
        base_steps = state.get("agent_steps", [])

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
            # Guard: если вопрос не содержит явных файловых триггеров —
            # это скорее всего фактический вопрос, а не запрос на отправку файла.
            # LLM может ошибочно вернуть send_file из-за файлового контекста в истории.
            if not _FILE_REQUEST_TRIGGERS.search(question):
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
            return {
                **state,
                "intent": intent,
                "namespace_id": namespace_id,
                "namespace_name_hint": namespace_name_hint,
                "search_query": search_query,
                "entity_name": parsed.entity_name,
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
        namespace_id = await self._maybe_resolve_namespace(state, config)
        namespace_name_hint: Optional[str] = None
        if question:
            history = state.get("history") or []
            parsed = await self.classifier.parse(
                question,
                has_file=True,
                has_url=bool(detected_url),
                history=history,
            )
            # Если LLM нашёл упоминание пространства — резолвим его
            if parsed.namespace_hint and namespace_id is None:
                configurable = config.get("configurable") or {}
                db = configurable.get("async_db")
                user_id = state.get("user_id")
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
                return {
                    **state,
                    "intent": IntentType.SUMMARIZE,
                    "detected_url": detected_url,
                    "namespace_id": namespace_id,
                    "agent_steps": state.get("agent_steps", []) + [
                        "[Router] Auto: summarize (file, LLM)"
                    ],
                }
            if parsed.intent == IntentType.RAG_QUERY:
                if not parsed.search_query:
                    # Нет конкретного поискового запроса — вопрос общего характера о файле,
                    # обрабатываем как суммаризацию
                    logger.info("[Router] Auto: summarize (file, rag_query with no search_query)")
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
        logger.info("[Router] Auto: save_file (filename=%s)", filename)
        return {
            **state,
            "intent": IntentType.SAVE_FILE,
            "detected_url": detected_url,
            "namespace_id": namespace_id,
            "namespace_name_hint": namespace_name_hint,
            "agent_steps": state.get("agent_steps", []) + ["[Router] Auto: save_file"],
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
