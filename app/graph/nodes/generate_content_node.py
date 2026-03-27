"""GenerateContentNode — генерирует текстовое содержимое файла через LLM.

Используется в MultiActionNode перед create_file, когда entity_content не задан
и пользователь просит создать файл с текстом (шутка, заметка, статья и т.д.).

Входные поля state:
  entity_name    — название файла (контекст для генерации)
  question       — исходный запрос пользователя
  entity_content — если уже заполнен — пропускает генерацию

Выходные поля state:
  entity_content — сгенерированный текст
"""
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.graph.state import AskState
from app.domain.protocols import LLMProvider

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Ты — ассистент, который создаёт текстовый контент для файлов. "
    "Сгенерируй содержимое файла согласно запросу пользователя. "
    "Отвечай ТОЛЬКО текстом файла, без предисловий, заголовков и пояснений."
)


class GenerateContentNode:
    """Генерирует entity_content через LLM на основе entity_name и question."""

    def __init__(self, *, llm_service: LLMProvider) -> None:
        self.llm_service = llm_service

    async def run(self, state: AskState, config: RunnableConfig) -> dict[str, Any]:
        agent_steps = list(state.get("agent_steps") or []) + ["GenerateContentNode"]

        # Если контент уже есть — пропускаем генерацию
        if state.get("entity_content"):
            logger.info("[GenerateContentNode] entity_content already set, skipping")
            return {"agent_steps": agent_steps}

        title = state.get("entity_name") or ""
        question = state.get("question") or ""

        if not title and not question:
            logger.warning("[GenerateContentNode] Neither entity_name nor question — cannot generate")
            return {"agent_steps": agent_steps}

        context_parts = []
        if question:
            context_parts.append(f"Запрос пользователя: {question}")
        if title:
            context_parts.append(f"Название файла: {title}")
        context_parts.append("Напиши содержимое этого файла.")

        messages = [
            {"role": "system", "text": _SYSTEM_PROMPT},
            {"role": "user", "text": "\n\n".join(context_parts)},
        ]

        try:
            content = await self.llm_service.complete(messages, temperature=0.7, max_tokens=512)
            content = content.strip()
            logger.info("[GenerateContentNode] Generated %d chars for title=%r", len(content), title)
            return {
                "entity_content": content,
                "agent_steps": agent_steps,
            }
        except Exception as exc:
            logger.error("[GenerateContentNode] LLM generation failed: %s", exc)
            return {"agent_steps": agent_steps}
