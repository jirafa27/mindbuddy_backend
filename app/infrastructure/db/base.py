from typing import Optional

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from app.core.config import settings


class Base(DeclarativeBase):
    pass


engine: Optional[AsyncEngine] = None
AsyncSessionLocal: Optional[async_sessionmaker[AsyncSession]] = None


def setup_async_engine() -> None:
    """Создать engine и AsyncSessionLocal. Вызывать из lifespan при старте приложения."""
    global engine, AsyncSessionLocal
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        future=True,
    )
    AsyncSessionLocal = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )


async def init_db() -> None:
    """Инициализация БД: создание расширения vector и всех таблиц. Вызывать после setup_async_engine()."""
    if engine is None:
        raise RuntimeError("setup_async_engine() must be called before init_db()")
    from app.infrastructure.db import models  # noqa: F401 — регистрация моделей в Base.metadata
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
