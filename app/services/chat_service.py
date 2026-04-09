import logging
from typing import Optional, List, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.graph import build_ask_graph
from app.graph.state import AskState, AttachedFile
from app.graph.schemas import AskResponse, SourceItem
from app.schemas.file import RawFileUpload
from app.domain.protocols import (
    BlobStorage,
    EmbeddingProvider,
    TaskPublisher,
    FileRepository,
    LLMProvider,
    VectorRepository,
    ChatRepository,
    UserFileRepository,
)
from app.domain.entities import ChatEntity, ChatMessageEntity, ConversationContext
from app.core.exceptions import NotFoundError, ForbiddenError
from app.core.enums import ChatMessageRole
from app.services.file_service import FileService
from app.services.namespace_service import NamespaceService
from app.services.search_service import SearchService
from app.services.text_chunker import TextChunkerService
from app.services.content_extractor import ContentExtractorService
from app.services.summary_service import SummaryService
from app.utils.file_readers import FileReaderFactory

logger = logging.getLogger(__name__)


# Скользящее окно: сколько последних сообщений чата передавать в LLM
CHAT_HISTORY_LIMIT = 10

_CONFIRM_WORDS = {"да", "yes"}


def _is_confirmation(text: str) -> bool:
    """True если текст — явное подтверждение («да» / «yes»)."""
    return text.strip().lower().rstrip("!.,") in _CONFIRM_WORDS


def _decline_message(pending: dict) -> str:
    """Возвращает сообщение об отмене в зависимости от типа отложенного действия."""
    action_type = pending.get("type", "")
    target = pending.get("target", "")
    if action_type == "delete_namespace":
        return f"Хорошо, {target} не удалено."
    if action_type == "delete_file":
        return f"Хорошо, {target} не перемещён в корзину."
    return f"Хорошо, действие отменено."


class ChatService:
    def __init__(
        self,
        *,
        db: AsyncSession,
        file_repository: FileRepository,
        user_file_repository: UserFileRepository,
        vector_repository: VectorRepository,
        search_service: SearchService,
        summary_service: SummaryService,
        summary_agent,
        file_reader_factory: FileReaderFactory,
        text_chunker: TextChunkerService,
        embedding_service: EmbeddingProvider,
        file_service: FileService,
        llm_service: LLMProvider,
        blob_storage: BlobStorage,
        namespace_service: Optional[NamespaceService] = None,
        content_extractor: Optional[ContentExtractorService] = None,
        task_publisher: Optional[TaskPublisher] = None,
        chat_repository: Optional[ChatRepository] = None,
    ):
        self.db = db
        self.file_repository = file_repository
        self.user_file_repository = user_file_repository
        self.chat_repository = chat_repository
        self.vector_repository = vector_repository
        self.search_service = search_service
        self.summary_service = summary_service
        self.summary_agent = summary_agent
        self.file_service = file_service
        self.namespace_service = namespace_service
        self.content_extractor = content_extractor
        self.blob_storage = blob_storage

        graph = build_ask_graph(
            file_reader_factory=file_reader_factory,
            text_chunker=text_chunker,
            embedding_service=embedding_service,
            file_service=file_service,
            llm_service=llm_service,
            blob_storage=blob_storage,
            namespace_service=namespace_service,
            content_extractor=content_extractor,
            task_publisher=task_publisher,
        )
        self.ask_graph = graph.compile()

    async def ask(
        self,
        question: str,
        user_id: int,
        namespace_id: Optional[int] = None,
        files: Optional[List[RawFileUpload]] = None,
        file_ids: Optional[List[int]] = None,
        history: Optional[List[dict]] = None,
        override_intent: Optional[str] = None,
        chat_id: Optional[int] = None,
        chat_name: Optional[str] = None,
    ) -> AskResponse:
        """
        Обработка запроса: при отсутствии chat_id создаётся новый чат,
        после ответа графа сообщения сохраняются в чат.

        Перед запуском графа проверяет pending_action: если чат ожидает подтверждения
        удаления и пользователь подтверждает — выполняет действие без вызова графа.
        """
        # Получаем контекст диалога из чата или создаём новый
        resolved_chat_id, history_for_llm, conv_context = await self._resolve_chat(
            chat_id, user_id, chat_name, fallback_history=history
        )

        # Обрабатываем ожидающее подтверждения действие
        if chat_id is not None:
            pending_response = await self._try_resolve_pending_action(
                chat_id, resolved_chat_id, question, namespace_id, user_id
            )
            if pending_response:
                return pending_response

        # Формируем состояние для графа
        state = await self._build_graph_state(
            question, user_id, namespace_id, conv_context,
            files, file_ids, history_for_llm, override_intent,
        )
        # Формируем конфигурацию для графа
        config = self._build_graph_config()
        # Запускаем граф
        result = await self.ask_graph.ainvoke(state, config=config)

        answer_text = result.get("answer", "")
        result_file_ids: List[int] = result.get("file_ids") or []
        if not result_file_ids and result.get("file_id") is not None:
            result_file_ids = [result["file_id"]]
        is_save_file = result.get("intent") == "save_file" or override_intent == "save_file" or bool(files)

        if self.chat_repository and resolved_chat_id is not None:
            await self._persist_turn(
                resolved_chat_id, question, namespace_id, files, file_ids,
                result_file_ids, is_save_file, answer_text, result, conv_context,
            )

        return AskResponse(
            answer=answer_text,
            sources=self._build_sources(result),
            agent_steps=result.get("agent_steps") or [],
            file_ids=result_file_ids,
            chat_id=resolved_chat_id,
        )

    # ------------------------------------------------------------------
    # Вспомогательные методы ask()
    # ------------------------------------------------------------------

    async def _resolve_chat(
        self,
        chat_id: Optional[int],
        user_id: int,
        chat_name: Optional[str],
        fallback_history: Optional[List[dict]],
    ) -> Tuple["Optional[int]", List[dict], ConversationContext]:
        """Загружает или создаёт чат. Возвращает (resolved_chat_id, history_for_llm, conv_context)."""
        if not self.chat_repository:
            return None, list(fallback_history or []), ConversationContext()

        if chat_id is not None:
            existing = await self.chat_repository.get_chat_by_id(chat_id, user_id)
            if existing is None:
                raise NotFoundError("Чат не найден или доступ запрещён")
            history_for_llm = await self._load_chat_history(chat_id, user_id)
            conv_context = ConversationContext.from_dict(existing.context)
            return chat_id, history_for_llm, conv_context

        new_chat = await self.chat_repository.create_chat(user_id=user_id, name=chat_name)
        await self.db.commit()
        return new_chat.id, list(fallback_history or []), ConversationContext()

    async def _load_chat_history(self, chat_id: int, user_id: int) -> List[dict]:
        """Загружает последние N сообщений чата в формате для LLM."""
        total = await self.chat_repository.get_messages_count(chat_id, user_id)
        if total == 0:
            return []
        offset = max(0, total - CHAT_HISTORY_LIMIT)
        messages = await self.chat_repository.get_messages(
            chat_id=chat_id, user_id=user_id, limit=CHAT_HISTORY_LIMIT, offset=offset,
        )
        return [
            {
                "role": m.role.value,
                "text": m.text,
                "file_ids": m.file_ids,
                "namespace_id": m.namespace_id,
            }
            for m in messages
        ]

    async def _try_resolve_pending_action(
        self,
        chat_id: int,
        resolved_chat_id: Optional[int],
        question: str,
        namespace_id: Optional[int],
        user_id: int,
    ) -> Optional[AskResponse]:
        """Обрабатывает ожидающее подтверждения действие (удаление и т.п.).
        Возвращает готовый AskResponse если pending был обработан, иначе None."""
        if not self.chat_repository:
            return None
        pending = await self.chat_repository.get_pending_action(chat_id)
        if not pending:
            return None

        if _is_confirmation(question):
            answer_text = await self._execute_pending_action(pending=pending, user_id=user_id)
            step = "ChatService(confirmed)"
        else:
            answer_text = _decline_message(pending)
            step = "ChatService(declined)"
            logger.info("[ChatService] pending_action cancelled for chat_id=%d", chat_id)

        await self.chat_repository.clear_pending_action(chat_id)
        await self.chat_repository.add_message(
            chat_id, ChatMessageRole.USER.value, question,
            file_ids=[], namespace_id=namespace_id,
        )
        await self.chat_repository.add_message(
            chat_id, ChatMessageRole.ASSISTANT.value, answer_text,
            file_ids=[], namespace_id=namespace_id,
        )
        await self.db.commit()
        return AskResponse(
            answer=answer_text, sources=[], agent_steps=[step],
            file_ids=[], chat_id=resolved_chat_id,
        )

    async def _build_graph_state(
        self,
        question: str,
        user_id: int,
        namespace_id: Optional[int],
        conv_context: ConversationContext,
        files: Optional[List[RawFileUpload]],
        file_ids: Optional[List[int]],
        history: List[dict],
        override_intent: Optional[str],
    ) -> AskState:
        attached_files: Optional[List[AttachedFile]] = None
        if files:
            attached_files = []
            for f in files:
                key = await self.blob_storage.put_blob({"raw": f["content"], "filename": f["filename"]})
                attached_files.append(AttachedFile(
                    file_blob_key=key,
                    filename=f["filename"],
                    content_type=f.get("content_type"),
                    size=f.get("size", len(f["content"])),
                ))

        state: AskState = {
            "question": question,
            "user_id": user_id,
            "namespace_id": namespace_id,
            "conversation_context": conv_context.to_dict(),
            "attached_files": attached_files,
            "history": history,
            "override_intent": override_intent,
        }
        if file_ids:
            state["history_file_id"] = file_ids[0]
            state["search_file_ids"] = file_ids
            state["explicit_file_ids"] = True
        return state

    def _build_graph_config(self) -> dict:
        return {
            "configurable": {
                "async_db": self.db,
                "file_repository": self.file_repository,
                "user_file_repository": self.user_file_repository,
                "vector_repository": self.vector_repository,
                "search_service": self.search_service,
                "summary_service": self.summary_service,
                "summary_agent": self.summary_agent,
                "file_service": self.file_service,
                "content_extractor": self.content_extractor,
                "namespace_service": self.namespace_service,
            }
        }

    async def _persist_turn(
        self,
        resolved_chat_id: int,
        question: str,
        namespace_id: Optional[int],
        files: Optional[List[RawFileUpload]],
        file_ids: Optional[List[int]],
        result_file_ids: List[int],
        is_save_file: bool,
        answer_text: str,
        result: dict,
        conv_context: ConversationContext,
    ) -> None:
        """Сохраняет оба сообщения хода, обновляет ConversationContext и pending_action в чате."""
        # Сообщение пользователя: только его файлы + новые при batch-загрузке
        user_file_ids = list(dict.fromkeys(file_ids or []))
        if bool(files) and result_file_ids:
            seen = set(user_file_ids)
            for fid in result_file_ids:
                if fid not in seen:
                    user_file_ids.append(fid)
                    seen.add(fid)
        await self.chat_repository.add_message(
            resolved_chat_id, ChatMessageRole.USER.value, question,
            file_ids=user_file_ids, namespace_id=namespace_id,
        )

        # Сообщение ассистента: при save_file файлы уже в сообщении пользователя
        assistant_file_ids = [] if is_save_file else result_file_ids
        resolved_namespace_id = (
            result.get("created_namespace_id")
            or result.get("namespace_id")
            or namespace_id
            or conv_context.active_namespace_id
        )
        await self.chat_repository.add_message(
            resolved_chat_id, ChatMessageRole.ASSISTANT.value, answer_text,
            file_ids=assistant_file_ids, namespace_id=resolved_namespace_id,
        )

        conv_context.active_namespace_id = resolved_namespace_id
        conv_context.active_file_ids = result_file_ids or []
        await self.chat_repository.update_context(resolved_chat_id, conv_context.to_dict())

        if pending_from_graph := result.get("pending_action"):
            await self.chat_repository.set_pending_action(resolved_chat_id, pending_from_graph)
            logger.info(
                "[ChatService] Set pending_action for chat_id=%d: type=%s",
                resolved_chat_id, pending_from_graph.get("type"),
            )

        await self.db.flush()
        await self.db.commit()

    def _build_sources(self, result: dict) -> List[SourceItem]:
        return [
            SourceItem(
                filename=s.get("filename", "?"),
                relevance=s.get("relevance", 0.0),
                file_id=s.get("file_id"),
            )
            for s in (result.get("sources") or [])
        ]

    async def _execute_pending_action(
        self, pending: dict, user_id: int
    ) -> str:
        """
        Выполняет подтверждённое отложенное действие (delete_namespace или delete_file).
        Возвращает текст ответа.
        """
        action_type = pending.get("type")
        params = pending.get("params") or {}
        target = pending.get("target", "объект")

        try:
            if action_type == "delete_namespace":
                namespace_id = params.get("namespace_id")
                if not namespace_id:
                    return "Ошибка: не указан ID пространства."
                if not self.namespace_service:
                    return "Сервис пространств недоступен."
                await self.namespace_service.delete_namespace(
                    namespace_id=namespace_id, user_id=user_id
                )
                return f"{target.capitalize()} удалено, а файлы перемещены в корзину."

            elif action_type == "delete_file":
                file_id = params.get("file_id")
                if not file_id:
                    return "Ошибка: не указан ID файла."
                await self.file_service.delete_file(file_id=file_id, user_id=user_id)
                return f"{target.capitalize()} перемещён в корзину."

            elif action_type == "delete_all_in_namespace":
                file_ids = params.get("file_ids") or []
                if not file_ids:
                    return "Ошибка: список файлов пуст."
                deleted, failed = 0, 0
                for fid in file_ids:
                    try:
                        await self.file_service.delete_file(file_id=fid, user_id=user_id)
                        deleted += 1
                    except Exception:
                        logger.warning("[ChatService] Failed to delete file_id=%d", fid)
                        failed += 1
                result_msg = f"Перемещено в корзину {deleted} файл(ов)."
                if failed:
                    result_msg += f" Не удалось переместить: {failed}."
                return result_msg

            elif action_type == "batch_delete":
                items = pending.get("items") or []
                answers = []
                for item in items:
                    answer = await self._execute_pending_action(item, user_id)
                    answers.append(answer)
                return "\n".join(answers) if answers else "Все объекты удалены."

            else:
                logger.warning("[ChatService] Unknown pending action type: %s", action_type)
                return "Неизвестное действие."

        except (NotFoundError, ForbiddenError) as exc:
            return str(exc)
        except Exception:
            logger.exception(
                "[ChatService] Failed to execute pending_action type=%s", action_type
            )
            return "Произошла ошибка при выполнении операции."

    async def get_user_chats(
        self,
        user_id: int,
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[List[Tuple[ChatEntity, int]], int]:
        """Список чатов пользователя с количеством сообщений и общее число чатов."""
        if not self.chat_repository:
            return [], 0
        return await self.chat_repository.get_user_chats(
            user_id=user_id, limit=limit, offset=offset
        )

    async def get_chat_messages(
        self,
        chat_id: int,
        user_id: int,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[ChatMessageEntity], int]:
        """История сообщений чата с пагинацией. NotFoundError если чат не найден или нет доступа."""
        if not self.chat_repository:
            raise NotFoundError("Чат не найден или доступ запрещён")
        chat = await self.chat_repository.get_chat_by_id(chat_id, user_id)
        if not chat:
            raise NotFoundError("Чат не найден или доступ запрещён")
        messages = await self.chat_repository.get_messages(
            chat_id=chat_id, user_id=user_id, limit=limit, offset=offset
        )
        total = await self.chat_repository.get_messages_count(chat_id, user_id)
        return messages, total

    async def update_chat_name(
        self, chat_id: int, user_id: int, name: Optional[str]
    ) -> Optional[ChatEntity]:
        """Переименовать чат. None если чат не найден или нет доступа."""
        if not self.chat_repository:
            return None
        chat_entity = await self.chat_repository.update_chat_name(
            chat_id=chat_id, user_id=user_id, name=name
        )
        if chat_entity:
            await self.db.commit()
        return chat_entity

    async def delete_chat(self, chat_id: int, user_id: int) -> bool:
        """Удалить чат и все сообщения. Возвращает True если удалён, False если не найден."""
        if not self.chat_repository:
            return False
        deleted = await self.chat_repository.delete_chat(chat_id=chat_id, user_id=user_id)
        if deleted:
            await self.db.commit()
        return deleted