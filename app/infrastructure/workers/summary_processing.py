"""Celery задачи для суммаризации контента."""
import asyncio
import logging

from celery import Task
from celery.exceptions import MaxRetriesExceededError

from app.infrastructure.workers.celery_app import celery_app
from app.core.dependencies import create_summary_service_for_celery, get_text_chunker_service, get_llm_provider
from app.graph.nodes.summary_agent import SummaryAgent
from app.infrastructure.db import base as db_base

logger = logging.getLogger(__name__)


def _ensure_async_db() -> None:
    if db_base.AsyncSessionLocal is None:
        db_base.setup_async_engine_for_celery()


def _get_summary_agent_for_celery() -> SummaryAgent:
    """SummaryAgent для Celery (без FastAPI Depends)."""
    return SummaryAgent(
        llm_service=get_llm_provider(),
        text_chunker=get_text_chunker_service(),
    )


async def _run_summarize_url(url: str, user_id: int) -> dict:
    _ensure_async_db()
    async with db_base.AsyncSessionLocal() as db:
        summary_service = create_summary_service_for_celery(db)
        summary_agent = _get_summary_agent_for_celery()
        from app.schemas.summary import SummaryResponse
        content_or_cached = await summary_service.get_content_for_summarization_url(url=url, user_id=user_id)
        if isinstance(content_or_cached, SummaryResponse):
            await db.commit()
            return content_or_cached.model_dump()
        content = content_or_cached
        summary_result = await summary_agent.summarize(content.text, title=content.title)
        await summary_service.save_summary(content.content_file_id, summary_result)
        await db.commit()
        result = summary_service.build_summary_response(content, summary_result)
        return result.model_dump()


@celery_app.task(bind=True, name="process_summary_url")
def process_summary_url(
    self: Task,
    url: str,
    user_id: int,
) -> dict:
    """
    Celery задача для асинхронной суммаризации URL.
    
    Args:
        url: URL для суммаризации (YouTube, веб-страница)
        user_id: ID пользователя
        
    Returns:
        Словарь с результатом суммаризации
    """
    logger.info("[Summary Task] Starting summarization for URL: %s (user=%d)", url, user_id)
    
    try:
        result = asyncio.run(_run_summarize_url(url, user_id))
        logger.info("[Summary Task] Completed for URL: %s, user_file_id=%d", url, result.get("user_file_id"))
        return result
    
    except Exception as e:
        logger.error("[Summary Task] Error for URL %s: %s", url, e, exc_info=True)
        
        try:
            logger.warning(
                "[Summary Task] Retrying URL %s (attempt %d/3)",
                url,
                self.request.retries + 1,
            )
            raise self.retry(exc=e, countdown=60, max_retries=3)
        except MaxRetriesExceededError:
            logger.error("[Summary Task] All retries exhausted for URL: %s", url)
            raise
