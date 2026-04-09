"""IntentNode — определяет список намерений через LLM и записывает в state.planned_intents."""
import logging

from langchain_core.runnables import RunnableConfig

from app.graph.state import AskState
from app.services.intent_classifier import IntentClassifierService

logger = logging.getLogger(__name__)

_HISTORY_SCAN_LIMIT = 10


class IntentNode:
    """
    Первый узел двухузловой системы планирования.
    Определяет ТОЛЬКО список намерений (без параметров).
    Записывает planned_intents в state для ActionResolverNode.
    """

    def __init__(self, intent_classifier: IntentClassifierService) -> None:
        self.intent_classifier = intent_classifier

    async def run(self, state: AskState, config: RunnableConfig) -> AskState:
        question = state.get("question", "").strip()
        history = state.get("history") or []
        has_file = bool(state.get("attached_files"))
        agent_steps = list(state.get("agent_steps") or [])

        url_in_current_message: bool = state.get("url_in_current_message") or False
        has_history_url: bool = state.get("has_history_url") or False

        has_history_summary = any(
            msg.get("role") == "assistant" and (msg.get("text") or "").strip()
            for msg in reversed(history[-_HISTORY_SCAN_LIMIT:])
        )

        intents = await self.intent_classifier.classify(
            question,
            has_file=has_file,
            has_url=url_in_current_message,
            has_history_url=has_history_url,
            has_history_summary=has_history_summary,
            history=history,
        )

        logger.info("[IntentNode] Classified intents: %s", " → ".join(intents))

        return {
            **state,
            "planned_intents": intents,
            "agent_steps": agent_steps + [
                f"[IntentNode] Intents: {' → '.join(intents)}"
            ],
        }
