"""MindBuddyAgent: синтез ответа по найденным чанкам (RAG)."""
import logging
from typing import Any, List

from langchain_core.runnables import RunnableConfig

from app.graph.state import AskState
from app.domain.protocols import LLMProvider
from app.infrastructure.llm.yandex_completion import LLMCompletionError

logger = logging.getLogger(__name__)

def _format_chunks(search_result: List[dict]) -> str:
    """Форматирует результаты поиска для LLM.
    
    Поддерживает два типа результатов:
    - Семантический поиск: chunk_text, filename, relevance
    - Структурные запросы: filename, file_size, created_at, namespace_name
    """
    chunks_text = []
    for i, row in enumerate(search_result, 1):
        chunk_text = row.get("chunk_text", "")
        filename = row.get("filename", "?")
        relevance = row.get("relevance")
        
        if chunk_text:
            # Семантический поиск — есть текст чанка
            rel_str = f" (релевантность: {relevance:.2f})" if relevance is not None else ""
            chunks_text.append(f"[{i}] Файл: {filename}{rel_str}\n{chunk_text}")
        else:
            # Структурный запрос — метаданные файлов/пространств
            parts = [f"[{i}] Файл: {filename}"]
            if row.get("namespace_name"):
                parts.append(f"Пространство: {row['namespace_name']}")
            if row.get("file_size"):
                size_kb = row["file_size"] / 1024
                parts.append(f"Размер: {size_kb:.1f} КБ")
            if row.get("created_at"):
                parts.append(f"Создан: {row['created_at']}")
            chunks_text.append(" | ".join(parts))
    
    return "\n\n".join(chunks_text)


class MindBuddyAgent:
    """Формирует ответ на вопрос только на основе переданных фрагментов (search_result)."""

    def __init__(self, *, llm_service: LLMProvider):
        self.llm_service = llm_service

    async def run(self, state: AskState, config: RunnableConfig) -> dict[str, Any]:
        question = state.get("question") or ""
        search_result = state.get("search_result") or []
        agent_steps = list(state.get("agent_steps") or [])
        
        # Если ответ уже сформирован ранее (например, RouterNode) — просто возвращаем его
        existing_answer = state.get("answer")
        if existing_answer and not search_result:
            logger.info("[MindBuddyAgent] Returning pre-set answer")
            return {
                "answer": existing_answer,
                "sources": [],
                "agent_steps": agent_steps + ["MindBuddyAgent (pass-through)"],
            }

        sources = []
        seen_files = set()
        for row in search_result:
            fn = row.get("filename", "?")
            rel = row.get("relevance")
            if fn not in seen_files:
                seen_files.add(fn)
                sources.append({"filename": fn, "relevance": float(rel) if rel is not None else 0.0})

        if not search_result:
            return {
                "answer": "По выбранному пространству знаний релевантных фрагментов не найдено. Загрузите документы или уточните вопрос.",
                "sources": [],
                "agent_steps": agent_steps + ["MindBuddyAgent"],
            }

        chunks_block = _format_chunks(search_result)

        messages = [
            {
                "role": "system",
                "text": (
                    "Ты — ассистент по базе знаний. Отвечай только на основе приведённой информации. "
                    "Информация может быть двух типов:\n"
                    "1. Фрагменты текста из документов — отвечай по их содержимому.\n"
                    "2. Список файлов/папок с метаданными — перечисли их пользователю.\n"
                    "Не придумывай факты. Если информации недостаточно, скажи об этом."
                ),
            },
            {
                "role": "user",
                "text": f"Вопрос пользователя: {question}\n\nИнформация из базы знаний:\n\n{chunks_block}\n\nДай краткий ответ.",
            },
        ]

        try:
            answer = await self.llm_service.complete(messages, temperature=0.3, max_tokens=1024)
            logger.info(f"ANSWER: {answer.strip()}")
            return {
                "answer": answer.strip(),
                "sources": sources,
                "agent_steps": agent_steps + ["MindBuddyAgent"],
            }
        except LLMCompletionError:
            return {
                "answer": "Не удалось сформировать ответ. Попробуйте переформулировать вопрос.",
                "sources": sources,
                "agent_steps": agent_steps + ["MindBuddyAgent"],
            }
