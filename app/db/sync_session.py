from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings


# Создаем синхронную сессию для Celery (Celery не поддерживает async напрямую)
sync_engine = create_engine(
    settings.DATABASE_URL.replace("+asyncpg", "").replace("postgresql+asyncpg", "postgresql"),
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=sync_engine, autocommit=False, autoflush=False)


@contextmanager
def get_sync_db() -> Session:
    """
    Context manager для получения синхронной сессии БД в Celery задачах.
    Автоматически делает commit при успехе или rollback при ошибке.
    
    Usage:
        with get_sync_db() as db:
            # работа с БД
    """
    db = SessionLocal()
    try:
        yield db
        # Если дошли сюда без исключения - делаем commit
        db.commit()
    except Exception:
        # При любой ошибке - rollback
        db.rollback()
        raise
    finally:
        # Всегда закрываем сессию
        db.close()
