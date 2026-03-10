import httpx
import logging

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    AppException,
    ValidationError,
    NotFoundError,
    ForbiddenError,
    FileTooLargeError,
    EmbeddingGenerationError,
    FileProcessingError,
    ContentExtractionError,
)

logger = logging.getLogger(__name__)

EXCEPTION_STATUS_MAP = {
    ValidationError: status.HTTP_400_BAD_REQUEST,
    NotFoundError: status.HTTP_404_NOT_FOUND,
    ForbiddenError: status.HTTP_403_FORBIDDEN,
    FileTooLargeError: status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    EmbeddingGenerationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    FileProcessingError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    ContentExtractionError: status.HTTP_422_UNPROCESSABLE_ENTITY,
}


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Обработчик исключений приложения (ValidationError, NotFoundError и т.д.)."""
    exception_type = type(exc)
    status_code = EXCEPTION_STATUS_MAP.get(
        exception_type, status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    log_level = logger.warning if 400 <= status_code < 500 else logger.error
    log_level("%s: %s", exception_type.__name__, exc.message)
    return JSONResponse(status_code=status_code, content={"detail": exc.message})


def _sanitize_validation_errors(errors: list) -> list:
    """Приводит список ошибок валидации к JSON-сериализуемому виду (без объектов в ctx и т.п.)."""
    out = []
    for e in errors:
        if not isinstance(e, dict):
            out.append({"msg": str(e)})
            continue
        clean = {}
        for k, v in e.items():
            if v is None or isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, (list, tuple)):
                clean[k] = list(v)
            elif isinstance(v, dict):
                clean[k] = {str(a): str(b) for a, b in v.items()}
            else:
                clean[k] = str(v)
        out.append(clean)
    return out


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Логирует детали ошибки валидации (422) и возвращает ответ клиенту."""
    errors = exc.errors()
    logger.warning(
        "Validation error (422) %s %s: %s",
        request.method,
        request.url.path,
        errors,
    )
    detail = _sanitize_validation_errors(errors)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": detail},
    )


async def unicode_decode_error_handler(request: Request, exc: UnicodeDecodeError) -> JSONResponse:
    """Ошибки декодирования файлов."""
    logger.error("Unicode decode error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "File must be valid UTF-8 text"},
    )


async def httpx_status_error_handler(request: Request, exc: httpx.HTTPStatusError) -> JSONResponse:
    """HTTP ошибки от внешних API."""
    logger.error("HTTP error from external API: %s - %s", exc.response.status_code, exc.response.text)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": f"External API error: {exc.response.status_code}"},
    )


async def httpx_request_error_handler(request: Request, exc: httpx.RequestError) -> JSONResponse:
    """Ошибки сетевых запросов."""
    logger.error("Request error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"Service unavailable: {str(exc)}"},
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Все необработанные исключения (в т.ч. из репозиториев и БД)."""
    logger.exception("Unexpected error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )
