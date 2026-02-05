from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.db import base


async def get_db() -> AsyncSession:
    """
    Dependency для получения сессии БД в FastAPI.
    """
    if base.AsyncSessionLocal is None:
        raise RuntimeError("Database not initialized: lifespan has not run")
    session = base.AsyncSessionLocal()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
