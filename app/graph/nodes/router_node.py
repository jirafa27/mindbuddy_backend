"""RouterNode — определяет намерение пользователя."""
import re
import logging
from typing import Optional, List, Dict

from app.graph.state import AskState
from app.core.enums import IntentType

logger = logging.getLogger(__name__)

# Паттерны для распознавания URL
URL_PATTERN = re.compile(
    r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*',
    re.IGNORECASE
)

# Количество сообщений в истории для поиска контекста
HISTORY_SCAN_LIMIT = 5


class RouterNode:
    """
    Определяет намерение пользователя.
    
    Два режима работы:
    1. Override mode: если override_intent задан — использует его напрямую
    2. Auto mode: анализирует текст через IntentClassifier
    
    Интенты:
    - summarize: суммаризация URL/файла
    - index_url: сохранить URL в базу
    - save_file: сохранить файл в базу
    - rag_query: вопрос по базе знаний
    """

    def __init__(self, intent_classifier):
        """
        Args:
            intent_classifier: Классификатор намерений
        """
        self.classifier = intent_classifier

    def _extract_url(self, text: str) -> Optional[str]:
        """Извлекает первый URL из текста."""
        match = URL_PATTERN.search(text)
        return match.group(0) if match else None

    def _extract_url_from_history(self, history: List[Dict]) -> Optional[str]:
        """Извлекает последний URL из истории сообщений."""
        if not history:
            return None
        
        for msg in reversed(history[-HISTORY_SCAN_LIMIT:]):
            url = self._extract_url(msg.get("text", ""))
            if url:
                logger.info("[Router] Found URL in history: %s", url)
                return url
        return None

    def _extract_file_id_from_history(self, history: List[Dict]) -> Optional[int]:
        """Извлекает последний file_id из истории сообщений."""
        if not history:
            return None
        
        for msg in reversed(history[-HISTORY_SCAN_LIMIT:]):
            file_id = msg.get("file_id")
            if file_id:
                logger.info("[Router] Found file_id in history: %d", file_id)
                return file_id
        return None

    def _is_url_only(self, text: str) -> bool:
        """Проверяет, состоит ли текст только из URL."""
        return bool(URL_PATTERN.fullmatch(text.strip()))

    def run(self, state: AskState) -> AskState:
        """
        Определяет намерение пользователя.
        
        Returns:
            Обновлённый state с intent, detected_url, history_file_id
        """
        question = state.get("question", "").strip()
        file_content = state.get("file_content")
        filename = state.get("filename")
        history = state.get("history") or []
        override_intent = state.get("override_intent")
        
        # Извлекаем URL и file_id из сообщения и истории (или из запроса)
        detected_url = self._extract_url(question)
        history_url = self._extract_url_from_history(history) if not detected_url else None
        history_file_id = state.get("history_file_id") or self._extract_file_id_from_history(history)
        
        # Используем URL из истории если в текущем сообщении нет
        if not detected_url and history_url:
            detected_url = history_url
        
        # === OVERRIDE MODE ===
        if override_intent:
            return self._handle_override(
                state, override_intent, detected_url, history_file_id, file_content, filename
            )
        
        # === AUTO MODE ===
        return self._handle_auto(
            state, question, detected_url, history_file_id, file_content, filename
        )

    def _handle_override(
        self,
        state: AskState,
        intent: IntentType,
        detected_url: Optional[str],
        history_file_id: Optional[int],
        file_content: Optional[bytes],
        filename: Optional[str],
    ) -> AskState:
        """Обработка override режима — интент задан явно."""
        logger.info("[Router] Override mode: intent=%s", intent)
        
        # Валидация: для summarize нужен URL, файл или file_id из истории
        if intent == IntentType.SUMMARIZE:
            if not detected_url and not file_content and not history_file_id:
                logger.warning("[Router] summarize requires URL, file or history_file_id")
                return {
                    **state,
                    "intent": "rag_query",
                    "answer": "Не нашёл что суммаризировать. Отправьте ссылку или файл.",
                    "agent_steps": state.get("agent_steps", []) + [
                        "[Router] Override: summarize, but no content found"
                    ],
                }
        
        # Валидация: для index_url нужен URL
        if intent == IntentType.INDEX_URL and not detected_url:
            logger.warning("[Router] index_url requires URL")
            return {
                **state,
                "intent": "rag_query",
                "answer": "Не нашёл ссылку для сохранения.",
                "agent_steps": state.get("agent_steps", []) + [
                    "[Router] Override: index_url, but no URL found"
                ],
            }
        
        # Валидация: для save_file нужен файл
        if intent == IntentType.SAVE_FILE and not file_content:
            logger.warning("[Router] save_file requires file")
            return {
                **state,
                "intent": "rag_query",
                "answer": "Не нашёл файл для сохранения.",
                "agent_steps": state.get("agent_steps", []) + [
                    "[Router] Override: save_file, but no file attached"
                ],
            }
        
        return {
            **state,
            "intent": intent,
            "detected_url": detected_url,
            "history_file_id": history_file_id,
            "agent_steps": state.get("agent_steps", []) + [
                f"[Router] Override: {intent}"
            ],
        }

    def _handle_auto(
        self,
        state: AskState,
        question: str,
        detected_url: Optional[str],
        history_file_id: Optional[int],
        file_content: Optional[bytes],
        filename: Optional[str],
    ) -> AskState:
        """Обработка auto режима — определяем интент через classifier."""
        
        # Если есть файл — определяем что с ним делать
        if file_content and filename:
            semantic_intent, score = self.classifier.classify(question) if question else (None, 0.0)
            
            if semantic_intent == IntentType.SUMMARIZE:
                logger.info("[Router] Auto: summarize (file=%s, score=%.3f)", filename, score)
                return {
                    **state,
                    "intent": IntentType.SUMMARIZE,
                    "detected_url": detected_url,
                    "agent_steps": state.get("agent_steps", []) + [
                        f"[Router] Auto: summarize (file, score={score:.3f})"
                    ],
                }
            else:
                logger.info("[Router] Auto: save_file (filename=%s)", filename)
                return {
                    **state,
                    "intent": IntentType.SAVE_FILE,
                    "detected_url": detected_url,
                    "agent_steps": state.get("agent_steps", []) + [
                        "[Router] Auto: save_file"
                    ],
                }
        
        # Если только URL без текста — сохранение
        if detected_url and self._is_url_only(question):
            logger.info("[Router] Auto: index_url (URL only)")
            return {
                **state,
                "intent": IntentType.INDEX_URL,
                "detected_url": detected_url,
                "agent_steps": state.get("agent_steps", []) + [
                    "[Router] Auto: index_url (URL only)"
                ],
            }
        
        # Классификация текста
        semantic_intent, score = self.classifier.classify(question)
        logger.info("[Router] Auto classification: %s (score=%.3f)", semantic_intent, score)
        
        if semantic_intent == IntentType.SUMMARIZE:
            # Суммаризация нужна, но нет контента
            if detected_url:
                logger.info("[Router] Auto: summarize (URL in text)")
                return {
                    **state,
                    "intent": IntentType.SUMMARIZE,
                    "detected_url": detected_url,
                    "history_file_id": history_file_id,
                    "agent_steps": state.get("agent_steps", []) + [
                        f"[Router] Auto: summarize (URL, score={score:.3f})"
                    ],
                }
            elif history_file_id:
                logger.info("[Router] Auto: summarize (file_id from history)")
                return {
                    **state,
                    "intent": IntentType.SUMMARIZE,
                    "detected_url": None,
                    "history_file_id": history_file_id,
                    "agent_steps": state.get("agent_steps", []) + [
                        f"[Router] Auto: summarize (history file_id={history_file_id}, score={score:.3f})"
                    ],
                }
            else:
                logger.info("[Router] Auto: no content for summarize")
                return {
                    **state,
                    "intent": IntentType.RAG_QUERY,
                    "answer": "Что суммаризировать? Отправьте ссылку или файл.",
                    "agent_steps": state.get("agent_steps", []) + [
                        f"[Router] Auto: summarize requested but no content (score={score:.3f})"
                    ],
                }
        
        elif semantic_intent == IntentType.INDEX_URL:
            if detected_url:
                logger.info("[Router] Auto: index_url (save intent with URL)")
                return {
                    **state,
                    "intent": IntentType.INDEX_URL,
                    "detected_url": detected_url,
                    "agent_steps": state.get("agent_steps", []) + [
                        f"[Router] Auto: index_url (score={score:.3f})"
                    ],
                }
            else:
                return {
                    **state,
                    "intent": IntentType.RAG_QUERY,
                    "answer": "Что сохранить? Отправьте ссылку или файл.",
                    "agent_steps": state.get("agent_steps", []) + [
                        f"[Router] Auto: save requested but nothing to save (score={score:.3f})"
                    ],
                }
        
        # По умолчанию — RAG запрос
        logger.info("[Router] Auto: rag_query (default)")
        return {
            **state,
            "intent": IntentType.RAG_QUERY,
            "detected_url": detected_url,
            "agent_steps": state.get("agent_steps", []) + [
                f"[Router] Auto: rag_query (semantic={semantic_intent}, score={score:.3f})"
            ],
        }
