"""IndexUrlNode — нода для индексации URL без суммаризации."""
import logging

from langchain_core.runnables import RunnableConfig
from sqlalchemy.ext.asyncio import AsyncSession

from app.graph.state import AskState
from app.domain.protocols import TaskPublisher
from app.services.content_extractor import ContentExtractorService
from app.services.file_service import FileService

logger = logging.getLogger(__name__)


class IndexUrlNode:
    """
    Нода для индексации URL (YouTube, веб-страница) в RAG без суммаризации.

    Поток:
    1. Экстракция контента
    2. Проверка дедупликации
    3. Сохранение в MinIO и БД
    4. Постановка задачи на индексацию эмбеддингов
    5. Возврат ответа с предложением суммаризировать
    """

    def __init__(
        self,
        content_extractor: ContentExtractorService,
        file_service: FileService,
        task_publisher: TaskPublisher,
    ):
        self.content_extractor = content_extractor
        self.file_service = file_service
        self.task_publisher = task_publisher

    async def run(self, state: AskState, config: RunnableConfig) -> AskState:
        """
        Индексирует URL в RAG и предлагает суммаризацию.
        
        Args:
            state: Состояние графа
            config: RunnableConfig с async_db в configurable
        """
        detected_url = state.get("detected_url")
        user_id = state.get("user_id")
        namespace_id = state.get("namespace_id")
        
        configurable = (config or {}).get("configurable") or {}
        db: AsyncSession | None = configurable.get("async_db")
        
        if not detected_url:
            return {
                **state,
                "answer": "Не удалось найти URL в вашем сообщении",
                "agent_steps": state.get("agent_steps", []) + ["[IndexUrl] Error: no URL"],
            }
        
        if not user_id or not db:
            return {
                **state,
                "answer": "Ошибка: недостаточно данных для обработки",
                "agent_steps": state.get("agent_steps", []) + ["[IndexUrl] Error: missing user_id or db"],
            }
        
        try:
            logger.info("[IndexUrl] Extracting content from: %s", detected_url)
            parsed = await self.content_extractor.extract(detected_url)

            dedup_result = await self.file_service.check_deduplication(
                user_id=user_id,
                source_url=detected_url,
            )
            
            if dedup_result.is_duplicate and dedup_result.existing_file_id:
                logger.info("[IndexUrl] Duplicate found: file_id=%d", dedup_result.existing_file_id)
                return {
                    **state,
                    "file_id": dedup_result.existing_file_id,  # user_file_id
                    "answer": f"Контент уже сохранён в вашей базе знаний.\n\n"
                              f"{parsed.title}\n\n"
                              f"Хотите сделать краткое резюме?",
                    "agent_steps": state.get("agent_steps", []) + [
                        f"[IndexUrl] Duplicate: file_id={dedup_result.existing_file_id}",
                    ],
                }
            
            user_file = await self.file_service.save_extracted_content(
                user_id=user_id,
                text=parsed.text,
                title=parsed.title,
                source_url=detected_url,
                content_hash=parsed.content_hash,
                content_type=parsed.content_type,
                namespace_id=namespace_id,
            )
            logger.info("[IndexUrl] Saved file: user_file_id=%d, title=%s", user_file.id, parsed.title)

            self.task_publisher.send_embeddings_task(
                content_file_id=user_file.file_id,
                text=parsed.text,
                namespace_id=namespace_id,
                filename=parsed.title,
                user_file_id=user_file.id,
            )
            
            return {
                **state,
                "file_id": user_file.id,
                "answer": f"Изучил контент и добавил в вашу базу знаний.\n\n"
                          f"**{parsed.title}**\n\n"
                          f"Что хотите сделать?",
                "agent_steps": state.get("agent_steps", []) + [
                    f"[IndexUrl] Indexed: file_id={user_file.id}, title={parsed.title}",
                ],
            }
        
        except ValueError as e:
            logger.warning("[IndexUrl] Validation error: %s", e)
            return {
                **state,
                "answer": f"Не удалось обработать ссылку: {e}",
                "agent_steps": state.get("agent_steps", []) + [f"[IndexUrl] Error: {e}"],
            }
        except Exception as e:
            logger.exception("[IndexUrl] Unexpected error")
            return {
                **state,
                "answer": f"Произошла ошибка при обработке ссылки: {e}",
                "agent_steps": state.get("agent_steps", []) + [f"[IndexUrl] Exception: {e}"],
            }
