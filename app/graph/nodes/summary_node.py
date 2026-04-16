"""SummaryNode — дирижёр суммаризации: FileService (контент) → SummaryService (кэш) → Agent → SummaryService (сохранение)."""
import logging
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig

from app.core.namespace_constants import INBOX_NAMESPACE_NAME
from app.graph.state import AskState
from app.domain.protocols import BlobStorage
from app.services.summary_service import SummaryService
from app.services.file_service import FileService
from app.schemas.summary import SummaryResponse
from app.graph.utils.namespace import resolve_namespace_id

logger = logging.getLogger(__name__)


class SummaryNode:
    """
    Дирижёр суммаризации:
    1. FileService — получить контент (URL / файл / существующий file_id)
    2. SummaryService.get_cached_summary — если есть кэш, вернуть
    3. SummaryAgent.summarize — иначе суммаризация через LLM
    4. SummaryService.save_summary + build_summary_response

    Для attached_files байты скачиваются из BlobStorage по ключам из state,
    после суммаризации ключи удаляются.
    """

    def __init__(self, *, blob_storage: Optional[BlobStorage] = None) -> None:
        self.blob_storage = blob_storage

    async def run(self, state: AskState, config: RunnableConfig) -> AskState:
        user_id = state.get("user_id")
        configurable = (config or {}).get("configurable") or {}
        file_service: Optional[FileService] = configurable.get("file_service")
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
            (state.get("attached_files") or state.get("history_file_id")) and not file_service
        ):
            logger.error("[SummaryNode] file_service (and content_extractor for URL) are required")
            return {
                **state,
                "answer": "Ошибка: сервис файлов недоступен",
                "agent_steps": state.get("agent_steps", []) + ["[SummaryNode] Error: missing file_service"],
            }
        attached_files = state.get("attached_files") or []
        detected_url = state.get("detected_url")
        history_file_id = state.get("history_file_id")
        search_file_ids: List[int] = state.get("search_file_ids") or []

        if attached_files:
            return await self._summarize_attached_files(
                state, file_service, summary_service, summary_agent, user_id, attached_files,
            )
        elif detected_url:
            return await self._summarize_url(
                state, summary_service, summary_agent, user_id, config
            )
        elif len(search_file_ids) > 1:
            return await self._summarize_multiple_files(
                state, summary_service, summary_agent, user_id, search_file_ids
            )
        elif history_file_id:
            return await self._summarize_existing_file(state, file_service, summary_service, summary_agent, user_id)

        # Нет конкретного файла/URL — если есть namespace, суммаризируем все файлы пространства
        namespace_id = state.get("namespace_id")
        if namespace_id:
            if not file_service:
                logger.error("[SummaryNode] file_service required for namespace summarization")
                return {
                    **state,
                    "answer": "Ошибка: сервис файлов недоступен",
                    "agent_steps": state.get("agent_steps", []) + ["[SummaryNode] Error: missing file_service"],
                }
            ns_file_ids = await file_service.list_user_file_ids_in_namespace(
                user_id, namespace_id
            )
            if ns_file_ids:
                logger.info(
                    "[SummaryNode] Summarizing %d files from namespace_id=%d",
                    len(ns_file_ids), namespace_id,
                )
                return await self._summarize_multiple_files(
                    state, summary_service, summary_agent, user_id, ns_file_ids
                )

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
        config: Optional[RunnableConfig] = None,
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

        # Для хранения файла-контента используем отдельный namespace_id:
        # если явно не задан — кладём в Inbox (не меняем state, чтобы не затронуть контекст диалога)
        storage_namespace_id = namespace_id
        storage_namespace_name = namespace_name_hint
        if storage_namespace_id is None and config:
            db = (config.get("configurable") or {}).get("async_db")
            if db and user_id:
                inbox_id = await resolve_namespace_id(db, user_id, INBOX_NAMESPACE_NAME)
                if inbox_id:
                    storage_namespace_id = inbox_id
                    storage_namespace_name = INBOX_NAMESPACE_NAME
                    logger.info("[SummaryNode] URL content: defaulting storage to Inbox (id=%d)", inbox_id)
        try:
            content_or_cached = await summary_service.get_content_for_summarization_url(
                url=detected_url, user_id=user_id, namespace_id=storage_namespace_id
            )
            if isinstance(content_or_cached, SummaryResponse):
                return self._state_from_response(
                    state, content_or_cached, f"Summarized URL: {detected_url}",
                    namespace_name_hint=storage_namespace_name,
                )
            content = content_or_cached
            summary_result = await summary_agent.summarize(content.text, title=content.title)
            await summary_service.save_summary(content.content_file_id, summary_result)
            result = SummaryService.build_summary_response(content, summary_result)
            return self._state_from_response(
                state, result, f"Summarized URL: {detected_url}",
                namespace_name_hint=storage_namespace_name,
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
        """Суммаризует файлы последовательно и объединяет ответы."""
        logger.info("[SummaryNode] Summarizing %d files: %s", len(file_ids), file_ids)

        parts: List[str] = []
        steps: List[str] = []
        for fid in file_ids:
            try:
                content_or_cached = await summary_service.get_content_for_summarization_existing_file(
                    file_id=fid, user_id=user_id
                )
                if isinstance(content_or_cached, SummaryResponse):
                    logger.info("[SummaryNode] Cached summary for file_id=%d", fid)
                    res = content_or_cached
                else:
                    content = content_or_cached
                    summary_result = await summary_agent.summarize(content.text, title=content.title)
                    await summary_service.save_summary(content.content_file_id, summary_result)
                    res = SummaryService.build_summary_response(content, summary_result)
                title = res.title or f"file_id={fid}"
                cached_mark = " _(кэш)_" if res.is_cached else ""
                parts.append(f"**{title}**{cached_mark}\n{res.summary}")
                steps.append(f"[SummaryNode] Done: file_id={fid}, cached={res.is_cached}")
            except Exception:
                logger.exception("[SummaryNode] Failed to summarize file_id=%d", fid)
                parts.append(f"**[file_id={fid}]** — не удалось суммаризировать.")
                steps.append(f"[SummaryNode] Failed: file_id={fid}")

        answer = "\n\n---\n\n".join(parts)
        return {
            **state,
            "answer": answer,
            "agent_steps": state.get("agent_steps", []) + steps,
        }

    async def _summarize_attached_files(
        self,
        state: AskState,
        file_service: FileService,
        summary_service: SummaryService,
        summary_agent,
        user_id: int,
        attached_files: list,
    ) -> AskState:
        """Суммаризует все приложенные файлы последовательно и объединяет ответы."""
        namespace_id = state.get("namespace_id")
        namespace_name_hint = state.get("namespace_name_hint")

        if len(attached_files) == 1:
            file_entry = attached_files[0]
            return await self._summarize_single_file(
                state, summary_service, summary_agent, user_id,
                file_blob_key=file_entry["file_blob_key"],
                filename=file_entry["filename"],
                namespace_id=namespace_id,
                namespace_name_hint=namespace_name_hint,
            )

        parts: List[str] = []
        steps: List[str] = []
        for file_entry in attached_files:
            file_blob_key = file_entry["file_blob_key"]
            filename = file_entry["filename"]
            try:
                file_content = await self._download_blob(file_blob_key, filename)
                if file_content is None:
                    parts.append(f"**{filename}** — не удалось загрузить файл.")
                    steps.append(f"[SummaryNode] Failed (blob not found): {filename}")
                    continue
                content_or_cached = await summary_service.get_content_for_summarization_file(
                    file_content=file_content,
                    filename=filename,
                    user_id=user_id,
                    namespace_id=namespace_id,
                )
                if isinstance(content_or_cached, SummaryResponse):
                    res = content_or_cached
                else:
                    content = content_or_cached
                    summary_result = await summary_agent.summarize(content.text, title=content.title)
                    await summary_service.save_summary(content.content_file_id, summary_result)
                    res = SummaryService.build_summary_response(content, summary_result)
                title = res.title or filename
                cached_mark = " _(кэш)_" if res.is_cached else ""
                parts.append(f"**{title}**{cached_mark}\n{res.summary}")
                steps.append(f"[SummaryNode] Done: {filename}, cached={res.is_cached}")
            except Exception:
                logger.exception("[SummaryNode] Failed to summarize file: %s", filename)
                parts.append(f"**{filename}** — не удалось суммаризировать.")
                steps.append(f"[SummaryNode] Failed: {filename}")
            finally:
                if self.blob_storage:
                    await self.blob_storage.delete_blob(file_blob_key)

        answer = "\n\n---\n\n".join(parts)
        return {
            **state,
            "answer": answer,
            "agent_steps": state.get("agent_steps", []) + steps,
        }

    async def _download_blob(self, file_blob_key: str, filename: str) -> Optional[bytes]:
        """Скачивает сырые байты файла из BlobStorage. Возвращает None при ошибке."""
        if not self.blob_storage:
            logger.error("[SummaryNode] blob_storage not configured, cannot download file=%s", filename)
            return None
        payload = await self.blob_storage.get_blob(file_blob_key)
        if payload is None:
            logger.error("[SummaryNode] blob not found: key=%s file=%s", file_blob_key, filename)
            return None
        return payload.get("raw")

    async def _summarize_single_file(
        self,
        state: AskState,
        summary_service: SummaryService,
        summary_agent,
        user_id: int,
        *,
        file_blob_key: str,
        filename: str,
        namespace_id: Optional[int],
        namespace_name_hint: Optional[str],
    ) -> AskState:
        """Суммаризует один приложенный файл (оптимизированный путь с summary_result)."""
        try:
            file_content = await self._download_blob(file_blob_key, filename)
            if file_content is None:
                return {
                    **state,
                    "answer": f"Не удалось загрузить файл «{filename}».",
                    "agent_steps": state.get("agent_steps", []) + ["[SummaryNode] Error: blob not found"],
                }
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
        finally:
            if self.blob_storage:
                await self.blob_storage.delete_blob(file_blob_key)

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
            if result.source_url:
                parts.append(f"_Содержимое страницы сохранено в пространство «{namespace_name_hint}»._\n\n")
            else:
                parts.append(f"_Содержимое файла сохранено в пространство «{namespace_name_hint}»._\n\n")
        if result.title:
            parts.append(f"{result.title}\n\n")
        parts.append(result.summary)
        if result.source_url:
            parts.append(f"\n\n🔗 Источник: {result.source_url}")
        if result.is_cached:
            parts.append("\n\n_💾 Из кэша_")
        return "".join(parts)
