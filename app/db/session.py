from sqlalchemy.ext.asyncio import AsyncSession
from app.db.base import AsyncSessionLocal


async def get_db() -> AsyncSession:
    """Dependency для получения сессии БД в FastAPI"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

