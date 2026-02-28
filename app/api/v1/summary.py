"""API эндпоинты для суммаризации (дирижёр: Service → Agent → Service)."""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query

from app.core.dependencies import (
    get_current_user,
    get_summary_service,
    get_summary_agent,
    get_file_service,
    get_content_extractor,
)
from app.schemas.user import UserResponse
from app.services.summary_service import SummaryService
from app.schemas.summary import SummaryResponse
from app.schemas.base import ResponseMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/summary", tags=["Summary"])


async def _run_summary_url(
    file_service,
    content_extractor,
    summary_service: SummaryService,
    summary_agent,
    url: str,
    user_id: int,
) -> SummaryResponse:
    """
    Создаёт суммаризацию контента по URL.
    """
    parsed = await content_extractor.extract(url)
    content = await file_service.get_or_create_content_from_extracted_url(parsed, url, user_id)
    cached = await summary_service.get_cached_summary(content.user_file_id)
    if cached:
        return cached
    summary_result = await summary_agent.summarize(content.text, title=content.title)
    await summary_service.save_summary(content.content_file_id, summary_result)
    return SummaryService.build_summary_response(content, summary_result)


async def _run_summary_file(
    file_service,
    summary_service: SummaryService,
    summary_agent,
    file_content: bytes,
    filename: str,
    user_id: int,
) -> SummaryResponse:
    """
    Создаёт суммаризацию контента по файлу.
    """
    content = await file_service.get_content_from_uploaded_file(
        file_content=file_content, filename=filename, user_id=user_id
    )
    cached = await summary_service.get_cached_summary(content.user_file_id)
    if cached:
        return cached
    summary_result = await summary_agent.summarize(content.text, title=content.title)
    await summary_service.save_summary(content.content_file_id, summary_result)
    return SummaryService.build_summary_response(content, summary_result)


@router.post("", response_model=ResponseMessage[SummaryResponse])
async def create_summary(
    url: Optional[str] = Query(None, description="URL для суммаризации (YouTube, веб-страница)"),
    file: Optional[UploadFile] = File(None, description="Файл для суммаризации"),
    user: UserResponse = Depends(get_current_user),
    file_service=Depends(get_file_service),
    content_extractor=Depends(get_content_extractor),
    summary_service: SummaryService = Depends(get_summary_service),
    summary_agent=Depends(get_summary_agent),
) -> ResponseMessage[SummaryResponse]:
    """
    Создаёт суммаризацию контента по URL или файлу.
    Args:
        user: из JWT (Authorization: Bearer)
        url: URL для суммаризации (YouTube, веб-страница)
        file: Файл для суммаризации
        user: User
        file_service: FileService
        content_extractor: ContentExtractor
        summary_service: SummaryService
        summary_agent: SummaryAgent
    Returns:
        SummaryResponse
    Raises:
        HTTPException: 400 если не указано url или file
        HTTPException: 500 если произошла ошибка суммаризации
    """
    if not url and not file:
        raise HTTPException(status_code=400, detail="Необходимо указать url или загрузить файл")

    result = None
    try:
        if url:
            logger.info("[API] Summary request for URL: %s (user=%d)", url, user.id)
            result = await _run_summary_url(
                file_service, content_extractor, summary_service, summary_agent, url, user.id
            )
        if file:
            logger.info("[API] Summary request for file: %s (user=%d)", file.filename, user.id)
            file_content = await file.read()
            result = await _run_summary_file(file_service, summary_service, summary_agent, file_content, file.filename, user.id)
    except ValueError as e:
        logger.warning("[API] Summary error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("[API] Summary failed")
        raise HTTPException(status_code=500, detail="Ошибка суммаризации")

    return ResponseMessage[SummaryResponse](data=result)

@router.get("/{file_id}", response_model=ResponseMessage[SummaryResponse])
async def get_summary(
    file_id: int,
    user: UserResponse = Depends(get_current_user),
    summary_service: SummaryService = Depends(get_summary_service),
) -> ResponseMessage[SummaryResponse]:
    """Получить суммаризацию по ID файла."""
    result = await summary_service.get_summary_by_file_id(file_id, user.id)
    if not result:
        raise HTTPException(status_code=404, detail="Суммаризация не найдена")
    
    return ResponseMessage[SummaryResponse](data=result)
