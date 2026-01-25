from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
from app.core.config import settings


class Base(DeclarativeBase):
    pass


engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine, expire_on_commit=False
)


async def init_db() -> None:
    """Инициализация БД: создание расширения vector и всех таблиц"""
    async with engine.begin() as conn:
        # Создаем расширение pgvector
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Создаем все таблицы
        await conn.run_sync(Base.metadata.create_all)

