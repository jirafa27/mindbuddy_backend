"""API эндпоинты для извлечения контента по URL (без суммаризации)."""
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.dependencies import get_current_user, get_summary_service
from app.schemas.user import UserResponse
from app.schemas.base import ResponseMessage
from app.schemas.content import ContentExtractResponse
from app.services.summary_service import SummaryService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/extract", response_model=ResponseMessage[ContentExtractResponse])
async def extract_content(
    url: str = Query(..., description="URL веб-страницы или YouTube-видео для парсинга"),
    user: UserResponse = Depends(get_current_user),
    summary_service: SummaryService = Depends(get_summary_service),
) -> ResponseMessage[ContentExtractResponse]:
    """
    Извлекает контент по URL (веб-страница или YouTube) без суммаризации.

    Сохраняет распарсенный текст в хранилище и таблицу files, но НЕ создаёт запись в user_files.
    Возвращённый file_id можно использовать для:
    - `POST /summary?content_file_id={file_id}` — суммаризация без привязки к пространству
    - `POST /files/{file_id}/attach` — привязка файла к пространству пользователя

    При повторном запросе с тем же URL возвращает ранее сохранённый file_id.
    """
    try:
        result = await summary_service.extract_url_content(url=url, user_id=user.id)
    except ValueError as e:
        logger.warning("[API/Extract] Validation error for URL %s: %s", url, e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("[API/Extract] Failed to extract content for URL: %s", url)
        raise HTTPException(status_code=500, detail="Ошибка извлечения контента")

    return ResponseMessage[ContentExtractResponse](data=result)
