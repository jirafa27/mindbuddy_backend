from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import httpx
import logging

from app.core.exceptions import (
    AppException,
    ValidationError,
    NotFoundError,
    ForbiddenError,
    FileTooLargeError,
    EmbeddingGenerationError,
)

logger = logging.getLogger(__name__)

# Маппинг исключений на HTTP статус коды
EXCEPTION_STATUS_MAP = {
    ValidationError: status.HTTP_400_BAD_REQUEST,
    NotFoundError: status.HTTP_404_NOT_FOUND,
    ForbiddenError: status.HTTP_403_FORBIDDEN,
    FileTooLargeError: status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    EmbeddingGenerationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware для централизованной обработки исключений"""

    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except HTTPException as e:
            # FastAPI HTTPException пропускаем как есть
            raise
        except AppException as e:
            # Все исключения приложения - маппим на HTTP статус код
            exception_type = type(e)
            status_code = EXCEPTION_STATUS_MAP.get(
                exception_type, status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            
            log_level = logger.warning if 400 <= status_code < 500 else logger.error
            log_level(f"{exception_type.__name__}: {e.message}")
            
            return JSONResponse(
                status_code=status_code,
                content={"detail": e.message}
            )
        except UnicodeDecodeError as e:
            # Ошибки декодирования файлов
            logger.error(f"Unicode decode error: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "File must be valid UTF-8 text"}
            )
        except httpx.HTTPStatusError as e:
            # HTTP ошибки от внешних API
            logger.error(f"HTTP error from external API: {e.response.status_code} - {e.response.text}")
            return JSONResponse(
                status_code=status.HTTP_502_BAD_GATEWAY,
                content={"detail": f"External API error: {e.response.status_code}"}
            )
        except httpx.RequestError as e:
            # Ошибки сетевых запросов
            logger.error(f"Request error: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": f"Service unavailable: {str(e)}"}
            )
        except Exception as e:
            # Все остальные необработанные исключения
            logger.exception(f"Unexpected error: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"}
            )
