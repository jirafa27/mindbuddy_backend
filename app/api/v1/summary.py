"""API эндпоинты для суммаризации (дирижёр: Service → Agent → Service)."""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query

from app.core.dependencies import (
    get_current_user,
    get_summary_service,
    get_summary_agent,
    get_file_service,
)
from app.schemas.user import UserResponse
from app.services.summary_service import SummaryService
from app.schemas.summary import SummaryResponse
from app.schemas.base import ResponseMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/summary")


async def _run_summary_url(
    summary_service: SummaryService,
    summary_agent,
    url: str,
    user_id: int,
) -> SummaryResponse:
    """
    Создаёт суммаризацию контента по URL.
    """
    content_or_cached = await summary_service.get_content_for_summarization_url(url=url, user_id=user_id)
    if isinstance(content_or_cached, SummaryResponse):
        return content_or_cached
    content = content_or_cached
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
    content_type: Optional[str] = None,
) -> SummaryResponse:
    """
    Создаёт суммаризацию контента по файлу.
    """
    content_or_cached = await summary_service.get_content_for_summarization_file(
        file_content=file_content, filename=filename, user_id=user_id, content_type=content_type
    )
    if isinstance(content_or_cached, SummaryResponse):
        return content_or_cached
    content = content_or_cached
    summary_result = await summary_agent.summarize(content.text, title=content.title)
    await summary_service.save_summary(content.content_file_id, summary_result)
    return SummaryService.build_summary_response(content, summary_result)


@router.post("", response_model=ResponseMessage[SummaryResponse])
async def create_summary(
    url: Optional[str] = Query(None, description="URL для суммаризации (YouTube, веб-страница)"),
    file_id: Optional[int] = Query(None, description="ID записи user_files для суммаризации"),
    content_file_id: Optional[int] = Query(
        None,
        description="ID контент-файла (files.id) из /content/extract — работает и без привязки к пространству",
    ),
    file: Optional[UploadFile] = File(None, description="Файл для суммаризации"),
    user: UserResponse = Depends(get_current_user),
    file_service=Depends(get_file_service),
    summary_service: SummaryService = Depends(get_summary_service),
    summary_agent=Depends(get_summary_agent),
) -> ResponseMessage[SummaryResponse]:
    """
    Создаёт суммаризацию контента по URL, content_file_id, user_file_id или загруженному файлу.

    - `url` — парсит URL и суммаризирует
    - `content_file_id` — суммаризирует по ID контент-файла (files.id) из `POST /content/extract`
    - `file_id` — суммаризирует по ID записи user_files либо по files.id (если из /content/extract — тот же id подойдёт)
    - `file` — загружает и суммаризирует файл
    """
    if not url and not file_id and not content_file_id and not file:
        raise HTTPException(
            status_code=400,
            detail="Необходимо указать url, content_file_id, file_id или загрузить файл",
        )

    result = None
    try:
        if url:
            logger.info("[API] Summary request for URL: %s (user=%d)", url, user.id)
            result = await _run_summary_url(summary_service, summary_agent, url, user.id)
        elif content_file_id:
            logger.info("[API] Summary request for content_file_id=%d (user=%d)", content_file_id, user.id)
            content_or_cached = await summary_service.get_content_for_summarization_by_content_file_id(
                content_file_id=content_file_id,
            )
            if isinstance(content_or_cached, SummaryResponse):
                result = content_or_cached
            else:
                summary_result = await summary_agent.summarize(content_or_cached.text, title=content_or_cached.title)
                await summary_service.save_summary(content_or_cached.content_file_id, summary_result)
                result = SummaryService.build_summary_response(content_or_cached, summary_result)
        elif file_id:
            logger.info("[API] Summary request for file_id=%d (user=%d)", file_id, user.id)
            try:
                content_or_cached = await summary_service.get_content_for_summarization_existing_file(
                    file_id=file_id, user_id=user.id
                )
            except ValueError:
                content_or_cached = await summary_service.get_content_for_summarization_by_content_file_id(
                    content_file_id=file_id,
                )
            if isinstance(content_or_cached, SummaryResponse):
                result = content_or_cached
            else:
                summary_result = await summary_agent.summarize(content_or_cached.text, title=content_or_cached.title)
                await summary_service.save_summary(content_or_cached.content_file_id, summary_result)
                result = SummaryService.build_summary_response(content_or_cached, summary_result)
        elif file:
            logger.info("[API] Summary request for file: %s (user=%d)", file.filename, user.id)
            file_content = await file.read()
            result = await _run_summary_file(
                file_service, summary_service, summary_agent,
                file_content, file.filename, user.id,
                content_type=file.content_type,
            )
    except ValueError as e:
        logger.warning("[API] Summary error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("[API] Summary failed")
        raise HTTPException(status_code=500, detail="Ошибка суммаризации")

    return ResponseMessage[SummaryResponse](data=result)


@router.get("/by-content/{content_file_id}", response_model=ResponseMessage[SummaryResponse])
async def get_summary_by_content_file(
    content_file_id: int,
    user: UserResponse = Depends(get_current_user),
    summary_service: SummaryService = Depends(get_summary_service),
) -> ResponseMessage[SummaryResponse]:
    """
    Получить суммаризацию по ID контент-файла (files.id).
    Работает для непривязанных файлов (без записи в user_files).
    """
    result = await summary_service.get_summary_by_content_file_id(content_file_id)
    if not result:
        raise HTTPException(status_code=404, detail="Суммаризация не найдена")
    return ResponseMessage[SummaryResponse](data=result)


@router.get("/{file_id}", response_model=ResponseMessage[SummaryResponse])
async def get_summary(
    file_id: int,
    user: UserResponse = Depends(get_current_user),
    summary_service: SummaryService = Depends(get_summary_service),
) -> ResponseMessage[SummaryResponse]:
    """Получить суммаризацию по ID записи user_files."""
    result = await summary_service.get_summary_by_file_id(file_id, user.id)
    if not result:
        raise HTTPException(status_code=404, detail="Суммаризация не найдена")
    
    return ResponseMessage[SummaryResponse](data=result)
