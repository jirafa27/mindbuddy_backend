"""SummaryNode — дирижёр суммаризации: FileService (контент) → SummaryService (кэш) → Agent → SummaryService (сохранение)."""
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.graph.state import AskState
from app.services.summary_service import SummaryService
from app.services.file_service import FileService
from app.schemas.summary import SummaryResponse

logger = logging.getLogger(__name__)


class SummaryNode:
    """
    Дирижёр суммаризации:
    1. FileService — получить контент (URL / файл / существующий file_id)
    2. SummaryService.get_cached_summary — если есть кэш, вернуть
    3. SummaryAgent.summarize — иначе суммаризация через LLM
    4. SummaryService.save_summary + build_summary_response
    """

    async def run(self, state: AskState, config: RunnableConfig) -> AskState:
        user_id = state.get("user_id")
        configurable = (config or {}).get("configurable") or {}
        file_service: FileService = configurable.get("file_service")
        content_extractor = configurable.get("content_extractor")
        summary_service: SummaryService = configurable.get("summary_service")
        summary_agent = configurable.get("summary_agent")

        if not user_id or not summary_service or not summary_agent:
            logger.error("[SummaryNode] user_id, summary_service and summary_agent are required")
            return {
                **state,
                "answer": "Ошибка: недостаточно данных для обработки",
                "agent_steps": state.get("agent_steps", []) + ["[SummaryNode] Error: missing deps"],
            }
        if (state.get("detected_url") and (not file_service or not content_extractor)) or (
            (state.get("file_content") or state.get("history_file_id")) and not file_service
        ):
            logger.error("[SummaryNode] file_service (and content_extractor for URL) are required")
            return {
                **state,
                "answer": "Ошибка: сервис файлов недоступен",
                "agent_steps": state.get("agent_steps", []) + ["[SummaryNode] Error: missing file_service"],
            }
        file_content = state.get("file_content")
        detected_url = state.get("detected_url")
        history_file_id = state.get("history_file_id")
        if file_content and state.get("filename"):
            return await self._summarize_file(state, file_service, summary_service, summary_agent, user_id)
        elif detected_url:
            return await self._summarize_url(
                state, file_service, content_extractor, summary_service, summary_agent, user_id
            )
        elif history_file_id:
            return await self._summarize_existing_file(state, file_service, summary_service, summary_agent, user_id)
        return {
            **state,
            "answer": "Не нашёл что суммаризировать. Отправьте файл или ссылку.",
            "agent_steps": state.get("agent_steps", []) + ["[SummaryNode] Error: no content"],
        }

    def _state_from_response(self, state: AskState, result: SummaryResponse, label: str) -> AskState:
        answer = self._format_summary_response(result)
        return {
            **state,
            "summary_result": result.model_dump(),
            "file_id": result.user_file_id,
            "answer": answer,
            "agent_steps": state.get("agent_steps", []) + [
                f"[SummaryNode] {label}",
                f"[SummaryNode] Method: {result.method}, cached: {result.is_cached}",
            ],
        }

    async def _summarize_url(
        self,
        state: AskState,
        file_service: FileService,
        content_extractor,
        summary_service: SummaryService,
        summary_agent,
        user_id: int,
    ) -> AskState:
        detected_url = state.get("detected_url") or ""
        if not detected_url:
            return {
                **state,
                "answer": "Не удалось найти ссылку для суммаризации.",
                "agent_steps": state.get("agent_steps", []) + ["[SummaryNode] Error: no URL"],
            }
        try:
            parsed = await content_extractor.extract(detected_url)
            content = await file_service.get_or_create_content_from_extracted_url(parsed, detected_url, user_id)
            cached = await summary_service.get_cached_summary(content.user_file_id)
            if cached:
                return self._state_from_response(state, cached, f"Summarized URL: {detected_url}")
            summary_result = await summary_agent.summarize(content.text, title=content.title)
            await summary_service.save_summary(content.content_file_id, summary_result)
            result = SummaryService.build_summary_response(content, summary_result)
            return self._state_from_response(state, result, f"Summarized URL: {detected_url}")
        except ValueError as e:
            return {**state, "answer": f"Ошибка: {e}", "agent_steps": state.get("agent_steps", []) + [f"[SummaryNode] Error: {e}"]}
        except Exception as e:
            logger.exception("[SummaryNode] Unexpected error")
            return {**state, "answer": f"Произошла ошибка при суммаризации: {e}", "agent_steps": state.get("agent_steps", []) + [f"[SummaryNode] Exception: {e}"]}

    async def _summarize_file(
        self,
        state: AskState,
        file_service: FileService,
        summary_service: SummaryService,
        summary_agent,
        user_id: int,
    ) -> AskState:
        file_content = state.get("file_content")
        filename = state.get("filename")
        if not file_content or not filename:
            return {
                **state,
                "answer": "Не удалось найти файл для суммаризации.",
                "agent_steps": state.get("agent_steps", []) + ["[SummaryNode] Error: no file"],
            }
        try:
            content = await file_service.get_content_from_uploaded_file(
                file_content=file_content, filename=filename, user_id=user_id
            )
            cached = await summary_service.get_cached_summary(content.user_file_id)
            if cached:
                return self._state_from_response(state, cached, f"Summarized file: {filename}")
            summary_result = await summary_agent.summarize(content.text, title=content.title)
            await summary_service.save_summary(content.content_file_id, summary_result)
            result = SummaryService.build_summary_response(content, summary_result)
            return self._state_from_response(state, result, f"Summarized file: {filename}")
        except ValueError as e:
            return {**state, "answer": f"Ошибка: {e}", "agent_steps": state.get("agent_steps", []) + [f"[SummaryNode] Error: {e}"]}
        except Exception as e:
            logger.exception("[SummaryNode] Unexpected error")
            return {**state, "answer": f"Произошла ошибка при суммаризации файла: {e}", "agent_steps": state.get("agent_steps", []) + ["[SummaryNode] Exception"]}

    async def _summarize_existing_file(
        self,
        state: AskState,
        file_service: FileService,
        summary_service: SummaryService,
        summary_agent,
        user_id: int,
    ) -> AskState:
        history_file_id = state.get("history_file_id")
        if not history_file_id:
            return {
                **state,
                "answer": "Не удалось найти файл в истории.",
                "agent_steps": state.get("agent_steps", []) + ["[SummaryNode] Error: no history_file_id"],
            }
        try:
            file = await file_service.get_file_info(history_file_id, user_id)
            if not file:
                return {
                    **state,
                    "answer": "Не удалось найти файл в истории.",
                    "agent_steps": state.get("agent_steps", []) + ["[SummaryNode] Error: no file"],
                }
            text = await file_service.get_file_text(file.user_file_id, user_id)
            if not text:
                return {
                    **state,
                    "answer": "Не удалось получить текст файла.",
                    "agent_steps": state.get("agent_steps", []) + ["[SummaryNode] Error: no text"],
                }
            summary_result = await summary_agent.summarize(text, title=file.filename)
            await summary_service.save_summary(file.content_file_id, summary_result)
            result = SummaryResponse(
                user_file_id=file.user_file_id,
                content_file_id=file.content_file_id,
                summary=summary_result.content,
                title=file.filename,
                source_url=None,
                is_cached=False,
                method=summary_result.method,
            )
            return self._state_from_response(state, result, f"Summarized existing file: {file.filename}")
        except ValueError as e:
            return {**state, "answer": f"Ошибка: {e}", "agent_steps": state.get("agent_steps", []) + [f"[SummaryNode] Error: {e}"]}
        except Exception as e:
            logger.exception("[SummaryNode] Unexpected error")
            return {**state, "answer": f"Произошла ошибка при суммаризации: {e}", "agent_steps": state.get("agent_steps", []) + ["[SummaryNode] Exception"]}

    def _format_summary_response(self, result: Any) -> str:
        parts = []
        if result.title:
            parts.append(f"{result.title}")
        parts.append(result.summary)
        if result.source_url:
            parts.append(f"\n\n🔗 Источник: {result.source_url}")
        if result.is_cached:
            parts.append("\n\n_💾 Из кэша_")
        return "".join(parts)
