import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Настройка логирования с временем
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
import httpx

from app.core.config import settings
from app.core.exceptions import AppException
from app.core.middleware import (
    app_exception_handler,
    unicode_decode_error_handler,
    httpx_status_error_handler,
    httpx_request_error_handler,
    generic_exception_handler,
)
from app.infrastructure.db.base import init_db, setup_async_engine
from app.api import api_router
from app.core.dependencies import get_intent_classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    setup_async_engine()
    await init_db()
    
    # Предзагрузка sentence-transformers модели (singleton)
    startup_logger = logging.getLogger(__name__)
    try:
        get_intent_classifier()  # Инициализация singleton
        startup_logger.info("IntentClassifier initialized successfully")
    except Exception as e:
        startup_logger.warning(f"IntentClassifier failed to load: {e}. Falling back to keyword matching.")

    yield

    from app.infrastructure.db.base import engine
    if engine is not None:
        try:
            await engine.dispose()
        except Exception:
            pass


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="0.1.0",
    lifespan=lifespan,
)

# Обработчики исключений (видят все исключения из эндпоинтов, без BaseHTTPMiddleware)
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(UnicodeDecodeError, unicode_decode_error_handler)
app.add_exception_handler(httpx.HTTPStatusError, httpx_status_error_handler)
app.add_exception_handler(httpx.RequestError, httpx_request_error_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# CORS настройки для Telegram Mini App
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роутеры
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Корневой endpoint для проверки работы API"""
    return {
        "message": "MindBuddy API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

