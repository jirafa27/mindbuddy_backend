"""SummaryNode — дирижёр суммаризации: FileService (контент) → SummaryService (кэш) → Agent → SummaryService (сохранение)."""
import asyncio
import logging
from typing import Any, List, Optional

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
        search_file_ids: List[int] = state.get("search_file_ids") or []

        if file_content and state.get("filename"):
            return await self._summarize_file(state, file_service, summary_service, summary_agent, user_id)
        elif detected_url:
            return await self._summarize_url(
                state, summary_service, summary_agent, user_id
            )
        elif len(search_file_ids) > 1:
            return await self._summarize_multiple_files(
                state, summary_service, summary_agent, user_id, search_file_ids
            )
        elif history_file_id:
            return await self._summarize_existing_file(state, file_service, summary_service, summary_agent, user_id)
        return {
            **state,
            "answer": "Не нашёл что суммаризировать. Отправьте файл или ссылку.",
            "agent_steps": state.get("agent_steps", []) + ["[SummaryNode] Error: no content"],
        }

    def _state_from_response(
        self,
        state: AskState,
        result: SummaryResponse,
        label: str,
        namespace_name_hint: Optional[str] = None,
    ) -> AskState:
        answer = self._format_summary_response(result, namespace_name_hint=namespace_name_hint)
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
        namespace_id = state.get("namespace_id")
        namespace_name_hint = state.get("namespace_name_hint")
        try:
            content_or_cached = await summary_service.get_content_for_summarization_url(
                url=detected_url, user_id=user_id, namespace_id=namespace_id
            )
            if isinstance(content_or_cached, SummaryResponse):
                return self._state_from_response(
                    state, content_or_cached, f"Summarized URL: {detected_url}",
                    namespace_name_hint=namespace_name_hint,
                )
            content = content_or_cached
            summary_result = await summary_agent.summarize(content.text, title=content.title)
            await summary_service.save_summary(content.content_file_id, summary_result)
            result = SummaryService.build_summary_response(content, summary_result)
            return self._state_from_response(
                state, result, f"Summarized URL: {detected_url}",
                namespace_name_hint=namespace_name_hint,
            )
        except ValueError as e:
            return {**state, "answer": f"Ошибка: {e}", "agent_steps": state.get("agent_steps", []) + [f"[SummaryNode] Error: {e}"]}
        except Exception as e:
            logger.exception("[SummaryNode] Unexpected error")
            return {**state, "answer": f"Произошла ошибка при суммаризации: {e}", "agent_steps": state.get("agent_steps", []) + [f"[SummaryNode] Exception: {e}"]}

    async def _summarize_multiple_files(
        self,
        state: AskState,
        summary_service: SummaryService,
        summary_agent,
        user_id: int,
        file_ids: List[int],
    ) -> AskState:
        """Суммаризует каждый файл из списка параллельно и объединяет ответы."""
        logger.info("[SummaryNode] Summarizing %d files: %s", len(file_ids), file_ids)

        async def _do_one(file_id: int) -> Optional[SummaryResponse]:
            try:
                content_or_cached = await summary_service.get_content_for_summarization_existing_file(
                    file_id=file_id, user_id=user_id
                )
                if isinstance(content_or_cached, SummaryResponse):
                    logger.info("[SummaryNode] Cached summary for file_id=%d", file_id)
                    return content_or_cached
                content = content_or_cached
                summary_result = await summary_agent.summarize(content.text, title=content.title)
                await summary_service.save_summary(content.content_file_id, summary_result)
                return SummaryService.build_summary_response(content, summary_result)
            except Exception:
                logger.exception("[SummaryNode] Failed to summarize file_id=%d", file_id)
                return None

        results = await asyncio.gather(*[_do_one(fid) for fid in file_ids])

        parts: List[str] = []
        steps: List[str] = []
        for fid, res in zip(file_ids, results):
            if res is None:
                parts.append(f"**[file_id={fid}]** — не удалось суммаризировать.")
                steps.append(f"[SummaryNode] Failed: file_id={fid}")
            else:
                title = res.title or f"file_id={fid}"
                cached_mark = " _(💾 кэш)_" if res.is_cached else ""
                parts.append(f"**{title}**{cached_mark}\n{res.summary}")
                steps.append(f"[SummaryNode] Done: file_id={fid}, cached={res.is_cached}")

        answer = "\n\n---\n\n".join(parts)
        return {
            **state,
            "answer": answer,
            "agent_steps": state.get("agent_steps", []) + steps,
        }

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
        namespace_id = state.get("namespace_id")
        namespace_name_hint = state.get("namespace_name_hint")
        try:
            content_or_cached = await summary_service.get_content_for_summarization_file(
                file_content=file_content,
                filename=filename,
                user_id=user_id,
                namespace_id=namespace_id,
            )
            if isinstance(content_or_cached, SummaryResponse):
                return self._state_from_response(
                    state, content_or_cached,
                    f"Summarized file: {filename}",
                    namespace_name_hint=namespace_name_hint,
                )
            content = content_or_cached
            summary_result = await summary_agent.summarize(content.text, title=content.title)
            await summary_service.save_summary(content.content_file_id, summary_result)
            result = SummaryService.build_summary_response(content, summary_result)
            return self._state_from_response(
                state, result,
                f"Summarized file: {filename}",
                namespace_name_hint=namespace_name_hint,
            )
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
            content_or_cached = await summary_service.get_content_for_summarization_existing_file(
                file_id=history_file_id,
                user_id=user_id,
            )
            if isinstance(content_or_cached, SummaryResponse):
                return self._state_from_response(
                    state, content_or_cached, f"Summarized existing file (cached): file_id={history_file_id}"
                )
            content = content_or_cached
            summary_result = await summary_agent.summarize(content.text, title=content.title)
            await summary_service.save_summary(content.content_file_id, summary_result)
            result = SummaryService.build_summary_response(content, summary_result)
            return self._state_from_response(state, result, f"Summarized existing file: {content.title}")
        except ValueError as e:
            return {**state, "answer": f"Ошибка: {e}", "agent_steps": state.get("agent_steps", []) + [f"[SummaryNode] Error: {e}"]}
        except Exception as e:
            logger.exception("[SummaryNode] Unexpected error")
            return {**state, "answer": f"Произошла ошибка при суммаризации: {e}", "agent_steps": state.get("agent_steps", []) + ["[SummaryNode] Exception"]}

    def _format_summary_response(self, result: Any, namespace_name_hint: Optional[str] = None) -> str:
        parts = []
        if namespace_name_hint:
            parts.append(f"_Файл сохранён в пространство «{namespace_name_hint}»._\n\n")
        if result.title:
            parts.append(f"{result.title}")
        parts.append(result.summary)
        if result.source_url:
            parts.append(f"\n\n🔗 Источник: {result.source_url}")
        if result.is_cached:
            parts.append("\n\n_💾 Из кэша_")
        return "".join(parts)
