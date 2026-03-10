import logging
from typing import Optional, List, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.graph import build_ask_graph
from app.graph.state import AskState
from app.graph.schemas import AskResponse, SourceItem
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
from app.domain.entities import ChatEntity, ChatMessageEntity
from app.core.exceptions import NotFoundError, ForbiddenError
from app.core.enums import ChatMessageRole
from app.services.file_service import FileService
from app.services.namespace_service import NamespaceService
from app.services.search_service import SearchService
from app.services.text_chunker import TextChunkerService
from app.services.content_extractor import ContentExtractorService
from app.services.summary_service import SummaryService
from app.services.llm_intent_classifier import LLMIntentClassifier
from app.utils.file_readers import FileReaderFactory

logger = logging.getLogger(__name__)

# Скользящее окно: сколько последних сообщений чата передавать в LLM
CHAT_HISTORY_LIMIT = 10


def _decline_message(pending: dict) -> str:
    """Возвращает сообщение об отмене в зависимости от типа отложенного действия."""
    action_type = pending.get("type", "")
    target = pending.get("target", "")
    if action_type == "delete_namespace":
        return f"Хорошо, {target} не удалено."
    if action_type == "delete_file":
        return f"Хорошо, {target} не удалён."
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
        intent_classifier,
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
        self.intent_classifier = intent_classifier

        graph = build_ask_graph(
            file_reader_factory=file_reader_factory,
            text_chunker=text_chunker,
            embedding_service=embedding_service,
            file_service=file_service,
            llm_service=llm_service,
            blob_storage=blob_storage,
            intent_classifier=intent_classifier,
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
        files: Optional[List[Tuple[bytes, str]]] = None,
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
        resolved_chat_id: Optional[int] = None
        history_for_llm: List[dict] = list(history or [])

        if self.chat_repository:
            if chat_id is not None:
                existing = await self.chat_repository.get_chat_by_id(chat_id, user_id)
                if existing is None:
                    raise NotFoundError("Чат не найден или доступ запрещён")
                resolved_chat_id = chat_id
                total = await self.chat_repository.get_messages_count(chat_id, user_id)
                if total > 0:
                    offset = max(0, total - CHAT_HISTORY_LIMIT)
                    history_entities = await self.chat_repository.get_messages(
                        chat_id=chat_id,
                        user_id=user_id,
                        limit=CHAT_HISTORY_LIMIT,
                        offset=offset,
                    )
                    history_for_llm = [
                        {"role": m.role.value, "text": m.text, "file_ids": m.file_ids}
                        for m in history_entities
                    ]

                # Проверяем pending_action — отложенное удаление ожидает подтверждения
                pending = await self.chat_repository.get_pending_action(chat_id)
                if pending:
                    if LLMIntentClassifier.is_confirmation(question):
                        answer_text = await self._execute_pending_action(
                            pending=pending, user_id=user_id
                        )
                        step = "ChatService(confirmed)"
                    else:
                        answer_text = _decline_message(pending)
                        step = "ChatService(declined)"
                        logger.info(
                            "[ChatService] pending_action cancelled for chat_id=%d", chat_id
                        )

                    await self.chat_repository.clear_pending_action(chat_id)
                    await self.chat_repository.add_message(
                        chat_id, ChatMessageRole.USER.value, question, file_ids=[]
                    )
                    await self.chat_repository.add_message(
                        chat_id, ChatMessageRole.ASSISTANT.value, answer_text, file_ids=[]
                    )
                    await self.db.commit()
                    return AskResponse(
                        answer=answer_text,
                        sources=[],
                        agent_steps=[step],
                        file_ids=[],
                        chat_id=resolved_chat_id,
                    )
            else:
                new_chat = await self.chat_repository.create_chat(
                    user_id=user_id, name=chat_name
                )
                resolved_chat_id = new_chat.id
                await self.db.commit()

        first_file_content = files[0][0] if files else None
        first_filename = files[0][1] if files else None

        state: AskState = {
            "question": question,
            "user_id": user_id,
            "namespace_id": namespace_id,
            "file_content": first_file_content,
            "filename": first_filename,
            "history": history_for_llm,
            "override_intent": override_intent,
        }
        if file_ids:
            state["history_file_id"] = file_ids[0]
            state["search_file_ids"] = file_ids
        config = {
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
        result = await self.ask_graph.ainvoke(state, config=config)

        answer_text = result.get("answer", "")
        result_intent = result.get("intent")
        is_save_file = (result_intent == "save_file" or override_intent == "save_file" or bool(files))

        result_file_ids: List[int] = result.get("file_ids") or []
        if not result_file_ids and result.get("file_id") is not None:
            result_file_ids = [result["file_id"]]

        # Обрабатываем файлы 2+ через тот же граф с override save_file
        if files and len(files) > 1:
            resolved_ns = result.get("namespace_id") or namespace_id
            ns_hint = result.get("namespace_name_hint")
            all_answers: List[str] = [answer_text]
            for extra_content, extra_filename in files[1:]:
                extra_state: AskState = {
                    "question": "",
                    "user_id": user_id,
                    "namespace_id": resolved_ns,
                    "namespace_name_hint": ns_hint,
                    "file_content": extra_content,
                    "filename": extra_filename,
                    "history": [],
                    "override_intent": "save_file",
                }
                extra_result = await self.ask_graph.ainvoke(extra_state, config=config)
                extra_ans = extra_result.get("answer", "")
                if extra_ans:
                    all_answers.append(extra_ans)
                extra_fid = extra_result.get("file_id")
                if extra_fid is not None:
                    result_file_ids.append(extra_fid)
            answer_text = "\n".join(all_answers)

        if self.chat_repository and resolved_chat_id is not None:
            user_message_file_ids = list(dict.fromkeys((file_ids or []) + result_file_ids))
            await self.chat_repository.add_message(
                resolved_chat_id, ChatMessageRole.USER.value, question,
                file_ids=user_message_file_ids,
            )
            # При сохранении файлов файлы относятся к сообщению пользователя, не ассистента
            assistant_file_ids = [] if is_save_file else result_file_ids
            await self.chat_repository.add_message(
                resolved_chat_id, ChatMessageRole.ASSISTANT.value, answer_text,
                file_ids=assistant_file_ids,
            )

            # Если CrudNode вернул pending_action — сохраняем его в чате
            pending_from_graph = result.get("pending_action")
            if pending_from_graph and resolved_chat_id:
                await self.chat_repository.set_pending_action(
                    resolved_chat_id, pending_from_graph
                )
                logger.info(
                    "[ChatService] Set pending_action for chat_id=%d: type=%s",
                    resolved_chat_id, pending_from_graph.get("type"),
                )

            await self.db.flush()
            await self.db.commit()

        sources_raw: List[dict] = result.get("sources") or []
        sources = [
            SourceItem(
                filename=s.get("filename", "?"),
                relevance=s.get("relevance", 0.0),
                file_id=s.get("file_id"),
            )
            for s in sources_raw
        ]

        return AskResponse(
            answer=answer_text,
            sources=sources,
            agent_steps=result.get("agent_steps") or [],
            file_ids=result_file_ids,
            chat_id=resolved_chat_id,
        )

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
                return f"{target.capitalize()} удалено вместе со всеми файлами."

            elif action_type == "delete_file":
                file_id = params.get("file_id")
                if not file_id:
                    return "Ошибка: не указан ID файла."
                await self.file_service.delete_file(file_id=file_id, user_id=user_id)
                return f"{target.capitalize()} удалён."

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