"""RouterNode — определяет намерение пользователя (fast paths) и передаёт остальное IntentNode."""
import logging
from typing import Optional, List, Dict

from langchain_core.runnables import RunnableConfig

from app.core.namespace_constants import INBOX_NAMESPACE_NAME
from app.graph.state import AskState
from app.core.enums import IntentType
from app.utils.url import extract_first_http_url, is_http_url_only
from app.graph.utils.namespace import resolve_namespace_id

logger = logging.getLogger(__name__)

# Количество сообщений в истории для поиска контекста
HISTORY_SCAN_LIMIT = 5


class RouterNode:
    """
    Тонкий маршрутизатор: обрабатывает детерминированные fast-paths
    и передаёт всё остальное в IntentNode → ActionResolverNode.

    Fast paths:
    - override_intent задан → прямой узел
    - URL без текста → index_url
    - файл без вопроса → save_file

    Всё остальное → IntentNode (intent не задан, url_in_current_message / has_history_url выставлены).
    """

    def _extract_url_from_history(self, history: List[Dict]) -> Optional[str]:
        for msg in reversed(history[-HISTORY_SCAN_LIMIT:]):
            url = extract_first_http_url(msg.get("text", "") or "")
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

    def _extract_file_ids_from_history(self, history: List[Dict]) -> List[int]:
        """Возвращает все file_ids из последнего сообщения с файлами в истории."""
        for msg in reversed(history[-HISTORY_SCAN_LIMIT:]):
            file_ids = msg.get("file_ids") or []
            if file_ids:
                logger.info("[Router] Found file_ids in history: %s", file_ids)
                return list(file_ids)
        return []

    async def _resolve_namespace_id(
        self, db, user_id: int, name: str
    ) -> Optional[int]:
        return await resolve_namespace_id(db, user_id, name)

    async def _inbox_namespace_if_unset(
        self,
        namespace_id: Optional[int],
        namespace_name_hint: Optional[str],
        config: RunnableConfig,
        user_id: Optional[int],
    ) -> tuple[Optional[int], Optional[str]]:
        """
        Файл из чата без явного пространства → кладём в Inbox
        (не используем namespace_id из query/UI).
        """
        if namespace_id is not None:
            return namespace_id, namespace_name_hint
        configurable = config.get("configurable") or {}
        db = configurable.get("async_db")
        if not db or user_id is None:
            return None, namespace_name_hint
        inbox_id = await self._resolve_namespace_id(db, user_id, INBOX_NAMESPACE_NAME)
        if inbox_id is not None:
            logger.info("[Router] Chat file upload: default → Inbox (id=%d)", inbox_id)
            return inbox_id, INBOX_NAMESPACE_NAME
        logger.warning("[Router] Chat file upload: Inbox not found for user_id=%s", user_id)
        return None, namespace_name_hint

    async def _maybe_resolve_namespace(
        self, state: AskState, config: RunnableConfig
    ) -> Optional[int]:
        """Возвращает namespace_id из state (если уже есть)."""
        return state.get("namespace_id")

    # ------------------------------------------------------------------
    # Основной метод
    # ------------------------------------------------------------------

    async def run(self, state: AskState, config: RunnableConfig) -> AskState:
        question = state.get("question", "").strip()
        has_file = bool(state.get("attached_files"))
        history = state.get("history") or []
        override_intent = state.get("override_intent")
        user_id = state.get("user_id")

        detected_url = extract_first_http_url(question)
        url_in_current_message = bool(detected_url)
        history_url = self._extract_url_from_history(history) if not detected_url else None
        history_file_id = (
            state.get("history_file_id") or self._extract_file_id_from_history(history)
        )

        # Все file_ids из последнего сообщения с файлами — для multi-file сценариев
        history_file_ids: List[int] = state.get("search_file_ids") or []
        if not history_file_ids and not state.get("history_file_id"):
            history_file_ids = self._extract_file_ids_from_history(history)

        # Объединяем URL из текущего сообщения и из истории для последующих узлов
        effective_detected_url = detected_url or history_url

        if override_intent:
            return await self._handle_override(
                state, override_intent, effective_detected_url, history_file_id, has_file,
            )

        # Fast path: только URL без текста → index_url
        if detected_url and is_http_url_only(question):
            logger.info("[Router] Fast path: index_url (URL only)")
            namespace_id, ns_hint = await self._inbox_namespace_if_unset(
                state.get("namespace_id"), None, config, user_id
            )
            return {
                **state,
                "intent": IntentType.INDEX_URL,
                "detected_url": detected_url,
                "namespace_id": namespace_id,
                "namespace_name_hint": ns_hint,
                "url_in_current_message": True,
                "has_history_url": False,
                "agent_steps": state.get("agent_steps", []) + [
                    "[Router] Fast path: index_url (URL only)"
                ],
            }

        # Fast path: файл без вопроса → save_file
        if has_file and not question:
            logger.info("[Router] Fast path: save_file (file, no question)")
            namespace_id, ns_hint = await self._inbox_namespace_if_unset(
                None, None, config, user_id
            )
            return {
                **state,
                "intent": IntentType.SAVE_FILE,
                "detected_url": effective_detected_url,
                "namespace_id": namespace_id,
                "namespace_name_hint": ns_hint,
                "url_in_current_message": url_in_current_message,
                "has_history_url": bool(history_url),
                "history_file_id": history_file_id,
                "agent_steps": state.get("agent_steps", []) + [
                    "[Router] Fast path: save_file (file, no question)"
                ],
            }

        # Файл + текст требует анализа намерения:
        # это может быть вопрос к файлу, сохранение в указанное пространство,
        # создание пространства + сохранение, суммаризация и т.д.
        if has_file and question:
            logger.info("[Router] File with text → IntentNode")
            return {
                **state,
                "detected_url": effective_detected_url,
                "url_in_current_message": url_in_current_message,
                "has_history_url": bool(history_url),
                "history_file_id": history_file_id,
                "agent_steps": state.get("agent_steps", []) + [
                    "[Router] File with text → IntentNode"
                ],
            }

        # Всё остальное → IntentNode (intent не задан)
        return {
            **state,
            "detected_url": effective_detected_url,
            "history_file_id": history_file_id,
            "search_file_ids": history_file_ids or state.get("search_file_ids") or [],
            "url_in_current_message": url_in_current_message,
            "has_history_url": bool(history_url),
            "agent_steps": state.get("agent_steps", []) + [
                "[Router] → IntentNode"
            ],
        }

    # ------------------------------------------------------------------
    # Override mode
    # ------------------------------------------------------------------

    async def _handle_override(
        self,
        state: AskState,
        intent: str,
        detected_url: Optional[str],
        history_file_id: Optional[int],
        has_file: bool,
    ) -> AskState:
        logger.info("[Router] Override mode: intent=%s", intent)

        if intent == IntentType.SUMMARIZE and not detected_url and not has_file and not history_file_id:
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

        if intent == IntentType.SAVE_FILE and not has_file:
            return {
                **state,
                "intent": IntentType.RAG_QUERY,
                "answer": "Не нашёл файл для сохранения.",
                "agent_steps": state.get("agent_steps", []) + [
                    "[Router] Override: save_file, but no file attached"
                ],
            }

        namespace_id = state.get("namespace_id")
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
