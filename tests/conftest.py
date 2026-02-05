import os

# До импорта app: подставляем тестовые сервисы (БД и MinIO)
os.environ["DATABASE_URL"] = os.environ.get(
    "DATABASE_TEST_URL", "postgresql+asyncpg://user:password@localhost:5433/mindbuddy_test"
)
# Тестовый Redis (docker-compose.test.yml, порт 6380)
os.environ["REDIS_URL"] = os.environ.get("REDIS_TEST_URL", "redis://localhost:6380/0")
# Тестовый RabbitMQ (docker-compose.test.yml, порт 5673)
os.environ["RABBITMQ_URL"] = os.environ.get("RABBITMQ_TEST_URL", "amqp://guest:guest@localhost:5673//")
# Тестовый MinIO (docker-compose.test.yml, порт 9002)
os.environ["MINIO_ENDPOINT"] = os.environ.get("MINIO_TEST_ENDPOINT", "localhost:9002")
os.environ["MINIO_ACCESS_KEY"] = os.environ.get("MINIO_TEST_ACCESS_KEY", "minioadmin")
os.environ["MINIO_SECRET_KEY"] = os.environ.get("MINIO_TEST_SECRET_KEY", "minioadmin")
os.environ["MINIO_BUCKET_NAME"] = os.environ.get("MINIO_TEST_BUCKET", "mindbuddy-files")

import pytest
from typing import AsyncGenerator, List
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from alembic.config import Config
from alembic import command
import httpx

from app.infrastructure.db.models import User, Namespace
from app.infrastructure.db.session import get_db
from app.core.config import settings
from app.core.dependencies import get_llm_provider
from app.main import app
from app.infrastructure.repositories.vector_queries import VECTOR_SEARCH_SQL


class MockLLMProvider:
    """Мок LLM для тестов: возвращает фиксированный SQL для SQLAgent и короткий ответ для MindBuddyAgent."""

    async def complete(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        system_text = (messages[0].get("text") or "") if messages else ""
        if "SQL" in system_text or "sql" in system_text:
            return VECTOR_SEARCH_SQL.strip()
        return "Mocked answer based on your question."

# Тестовая БД (для миграций и фикстуры engine)
TEST_DATABASE_URL = settings.DATABASE_TEST_URL or settings.DATABASE_URL

# Порядок таблиц для TRUNCATE: дочерние перед родительскими (CASCADE учтёт FK)
TRUNCATE_TABLES = "vector_embeddings, files, namespaces, users"


@pytest.fixture(scope="session", autouse=True)
def run_migrations():
    """Автоматически прогоняем миграции перед началом всех тестов."""
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    command.upgrade(alembic_cfg, "head")
    yield


@pytest.fixture
async def engine():
    """Движок на каждый тест — создаётся в том же loop, что и запрос (избегаем "attached to a different loop")."""
    eng = create_async_engine(TEST_DATABASE_URL, echo=False)
    yield eng
    await eng.dispose()


@pytest.fixture
async def db_session(engine) -> AsyncGenerator[AsyncSession, None]:
    """Сессия для каждого теста. Перед тестом все таблицы очищаются."""
    connection = await engine.connect()
    await connection.execute(text(f"TRUNCATE {TRUNCATE_TABLES} RESTART IDENTITY CASCADE"))
    await connection.commit()

    Session = async_sessionmaker(bind=connection, expire_on_commit=False)
    session = Session()

    yield session

    await session.close()
    await connection.close()


@pytest.fixture
async def test_user(db_session):
    user = User(
        telegram_id=123,
        username="testuser",
        full_name="Test User",
        watcher_token="test_watcher_token_123",
    )
    db_session.add(user)
    await db_session.commit()
    return user


@pytest.fixture
async def test_user_2(db_session):
    """Второй пользователь для тестов доступа (чужой namespace/file)."""
    user = User(telegram_id=456, username="otheruser", full_name="Other User")
    db_session.add(user)
    await db_session.commit()
    return user


@pytest.fixture
async def test_namespace(db_session, test_user):
    namespace = Namespace(user_id=test_user.id, name="test_namespace", description="test_description")
    db_session.add(namespace)
    await db_session.commit()
    return namespace


@pytest.fixture
async def client(db_session):
    """Клиент API; get_db и get_llm_provider подменены (тестовая БД и MockLLMProvider)."""
    async def override_get_db():
        yield db_session

    def override_get_llm_provider():
        return MockLLMProvider()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_llm_provider] = override_get_llm_provider
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            yield ac
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_llm_provider, None)