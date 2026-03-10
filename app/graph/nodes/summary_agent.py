"""SummaryAgent: суммаризация текста с поддержкой Map-Reduce для больших документов."""
import asyncio
import logging
from typing import Optional

from app.graph.schemas import SummaryResult
from app.core.config import settings
from app.core.enums import SummaryMethod
from app.domain.protocols import LLMProvider
from app.services.text_chunker import TextChunkerService

logger = logging.getLogger(__name__)


class SummaryAgent:
    """Агент суммаризации с поддержкой Stuffing и Map-Reduce.
    
    Порог и размер чанков привязаны к окну контекста Yandex (YANDEX_COMPLETION_CONTEXT_TOKENS):
    - Stuffing: один запрос, если текст укладывается в (окно − запас под промпт и ответ)
    - Map-Reduce: текст режется на чанки чуть меньше окна, каждый чанк суммаризируется, затем финальное объединение
    """
    
    SYSTEM_PROMPT_SUMMARY = (
        "Ты — инструмент суммаризации документов. "
        "Твоя единственная задача: пересказать содержимое текста, который тебе передан. "
        "СТРОГИЕ ПРАВИЛА: "
        "1. Используй ИСКЛЮЧИТЕЛЬНО информацию из текста между тегами <text> и </text>. "
        "2. ЗАПРЕЩЕНО добавлять любые факты, примеры, имена, термины из своих знаний. "
        "3. Если факт не написан в тексте явно — не упоминай его. "
        "4. Не интерпретируй тему документа на основе своих знаний — только пересказывай написанное. "
        "Отвечай на русском языке."
    )

    SYSTEM_PROMPT_CHUNK = (
        "Ты — инструмент суммаризации фрагментов текста. "
        "Перескажи содержимое фрагмента между тегами <text> и </text>. "
        "СТРОГО: используй только то, что написано в тексте. Ничего от себя. "
        "Отвечай на русском языке."
    )

    SYSTEM_PROMPT_FINAL = (
        "Ты — инструмент суммаризации. "
        "Объедини резюме фрагментов в единое связное резюме документа. "
        "СТРОГО: используй только то, что есть в резюме фрагментов. Ничего от себя. "
        "Убери повторы, сохрани все важные факты. "
        "Отвечай на русском языке."
    )
    
    def __init__(
        self,
        llm_service: LLMProvider,
        text_chunker: Optional[TextChunkerService] = None,
    ):
        """
        Args:
            llm_service: LLM провайдер для генерации
            text_chunker: Сервис разбиения на чанки (для Map-Reduce)
        """
        self.llm_service = llm_service
        self.text_chunker = text_chunker
    
    def _max_input_tokens(self) -> int:
        """Максимум токенов на вход в одном запросе (окно минус запас под промпт и ответ)."""
        return max(
            512,
            settings.YANDEX_COMPLETION_CONTEXT_TOKENS - settings.YANDEX_SUMMARY_CONTEXT_RESERVE,
        )

    async def summarize(
        self,
        text: str,
        title: Optional[str] = None,
    ) -> SummaryResult:
        """
        Создаёт резюме текста.
        
        Автоматически выбирает метод по размеру в токенах (окно Yandex):
        - Stuffing: текст помещается в (окно − запас)
        - Map-Reduce: текст режется на чанки чуть меньше окна, затем финальное объединение
        
        Args:
            text: Текст для суммаризации
            title: Заголовок документа (опционально)
            
        Returns:
            SummaryResult с резюме и метаданными
        """
        if not self.text_chunker:
            # Без чанкера — всегда stuffing (риск при длинном тексте)
            return await self._stuffing(text, title)
        # Считаем токены того же user-сообщения, что уйдёт в LLM при stuffing
        user_text = text if not title else f"Документ: {title}\n\n{text}"
        user_message = f"Создай резюме следующего текста:\n\n{user_text}"
        system_tokens = self.text_chunker.count_tokens(self.SYSTEM_PROMPT_SUMMARY)
        user_tokens = self.text_chunker.count_tokens(user_message)
        max_input = self._max_input_tokens()
        if system_tokens + user_tokens <= max_input:
            return await self._stuffing(text, title)
        return await self._map_reduce(text, title)
    
    async def _stuffing(self, text: str, title: Optional[str] = None) -> SummaryResult:
        """Суммаризация коротких текстов одним запросом к Pro модели."""
        logger.info("[Summary] Starting Stuffing method (text length: %d)", len(text))
        logger.info("[Summary] Text preview (first 500 chars): %s", repr(text[:500]))

        title_line = f"Название документа: {title}\n\n" if title else ""
        user_message = (
            f"{title_line}"
            f"<text>\n{text}\n</text>\n\n"
            "Составь подробное резюме документа выше. "
            "Пиши ТОЛЬКО то, что явно написано в тексте внутри тегов <text>. "
            "Не добавляй ничего из своих знаний."
        )

        messages = [
            {"role": "system", "text": self.SYSTEM_PROMPT_SUMMARY},
            {"role": "user", "text": user_message},
        ]

        summary = await self.llm_service.complete(
            messages,
            temperature=0.1,
            max_tokens=4096,
        )
        
        logger.info("[Summary] Stuffing completed, summary length: %d", len(summary))
        
        return SummaryResult(
            content=summary.strip(),
            model_name="yandexgpt-pro",
            method=SummaryMethod.STUFFING,
            chunks_processed=1,
        )
    
    async def _map_reduce(self, text: str, title: Optional[str] = None) -> SummaryResult:
        """Суммаризация длинных текстов через Map-Reduce.
        
        1. Map: Разбиваем текст на чанки по размеру чуть меньше окна Yandex, суммаризируем каждый
        2. Reduce: Объединяем резюме чанков в финальное резюме
        """
        max_chunk_tokens = self._max_input_tokens()
        logger.info(
            "[Summary] Map-Reduce started for text length: %d, chunk max tokens: %d",
            len(text),
            max_chunk_tokens,
        )
        
        # Map: разбиваем на чанки по токенам (чуть меньше окна контекста)
        chunks = self.text_chunker.chunk_text(
            text,
            chunk_size=max_chunk_tokens,
            chunk_overlap=min(100, max_chunk_tokens // 10),
        )
        logger.info("[Summary] Split into %d chunks", len(chunks))
        
        # Map: параллельная суммаризация чанков через asyncio.gather
        async def summarize_chunk(chunk: str, index: int) -> str:
            messages = [
                {"role": "system", "text": self.SYSTEM_PROMPT_CHUNK},
                {
                    "role": "user",
                    "text": (
                        f"<text>\n{chunk}\n</text>\n\n"
                        "Перескажи содержимое фрагмента выше. "
                        "Только то, что в нём написано — ничего от себя."
                    ),
                },
            ]
            result = await self.llm_service.complete(
                messages,
                temperature=0.1,
                max_tokens=1024,
            )
            logger.info("[Summary] Chunk %d/%d processed", index + 1, len(chunks))
            return result.strip()

        chunk_tasks = [
            summarize_chunk(chunk, i) for i, chunk in enumerate(chunks)
        ]
        chunk_summaries = await asyncio.gather(*chunk_tasks)

        logger.info("[Summary] All %d chunks processed, starting Reduce", len(chunks))

        combined_summaries = "\n\n---\n\n".join(
            f"Фрагмент {i+1}:\n{s}" for i, s in enumerate(chunk_summaries)
        )

        title_line = f"Название документа: {title}\n\n" if title else ""
        reduce_message = (
            f"{title_line}"
            f"<fragments>\n{combined_summaries}\n</fragments>\n\n"
            "Объедини резюме фрагментов в единое резюме документа. "
            "Используй ТОЛЬКО то, что написано в фрагментах выше."
        )

        messages = [
            {"role": "system", "text": self.SYSTEM_PROMPT_FINAL},
            {"role": "user", "text": reduce_message},
        ]

        final_summary = await self.llm_service.complete(
            messages,
            temperature=0.1,
            max_tokens=4096,
        )
        
        logger.info("[Summary] Map-Reduce completed, final summary length: %d", len(final_summary))
        
        return SummaryResult(
            content=final_summary.strip(),
            model_name="yandexgpt-pro",
            method=SummaryMethod.MAP_REDUCE,
            chunks_processed=len(chunks),
        )
