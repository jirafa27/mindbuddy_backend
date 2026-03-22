import json
import os
from contextlib import asynccontextmanager

# До импорта app: подставляем тестовые сервисы (БД и MinIO)
os.environ["DATABASE_URL"] = os.environ.get(
    "DATABASE_TEST_URL", "postgresql+asyncpg://user:password@localhost:5433/mindbuddy_test"
)
os.environ["REDIS_URL"] = os.environ.get("REDIS_TEST_URL", "redis://localhost:6380/0")
os.environ["RABBITMQ_URL"] = os.environ.get("RABBITMQ_TEST_URL", "amqp://guest:guest@localhost:5673//")
os.environ["MINIO_ENDPOINT"] = os.environ.get("MINIO_TEST_ENDPOINT", "localhost:9002")
os.environ["MINIO_ACCESS_KEY"] = os.environ.get("MINIO_TEST_ACCESS_KEY", "minioadmin")
os.environ["MINIO_SECRET_KEY"] = os.environ.get("MINIO_TEST_SECRET_KEY", "minioadmin")
os.environ["MINIO_BUCKET_NAME"] = os.environ.get("MINIO_TEST_BUCKET", "mindbuddy-files")

import pytest
from typing import AsyncGenerator, List, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from alembic.config import Config
from alembic import command
import httpx

from app.infrastructure.db.models import User, Namespace
from app.infrastructure.db.session import get_db
from app.core.config import settings
from app.core.dependencies import get_llm_provider, get_task_publisher, get_storage_service, get_blob_storage, get_embedding_service
from app.core.security import create_access_token, hash_password
from app.main import app
from app.infrastructure.repositories.vector_queries import VECTOR_SEARCH_SQL


class MockLLMProvider:
    """
    Мок LLM для тестов.

    Когда LLMIntentClassifier запрашивает классификацию (системный промпт содержит
    «классификатор намерений»), возвращает JSON с настроенным intent.
    Когда SQLAgent запрашивает SQL — возвращает заготовленный VECTOR_SEARCH_SQL.
    Всё остальное — заглушка-ответ.
    """

    def __init__(self, intent: str = "general_chat", **intent_params):
        self.intent = intent
        self.intent_params = intent_params

    async def complete(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        system_text = (messages[0].get("text") or "") if messages else ""

        if "классификатор намерений" in system_text or "КЛЮЧЕВЫЕ ПРАВИЛА" in system_text:
            result = {
                "intent": self.intent,
                "search_query": self.intent_params.get("search_query"),
                "namespace_hint": self.intent_params.get("namespace_hint"),
                "search_mode": self.intent_params.get("search_mode"),
                "entity_name": self.intent_params.get("entity_name"),
                "entity_description": self.intent_params.get("entity_description"),
                "entity_content": self.intent_params.get("entity_content"),
            }
            return json.dumps(result, ensure_ascii=False)

        if "SQL" in system_text or "sql" in system_text:
            return VECTOR_SEARCH_SQL.strip()

        return "Mocked answer based on your question."


class MockTaskPublisher:
    """Мок TaskPublisher: не публикует задачи в RabbitMQ/Celery."""

    def send_embeddings_task(self, **kwargs) -> str:
        return "mock-task-id"

    def send_bulk_edit_task(self, **kwargs) -> str:
        return "mock-bulk-task-id"


class MockEmbeddingProvider:
    """Мок EmbeddingProvider: возвращает нулевые векторы без вызова Ollama."""

    async def generate_embedding(self, text: str) -> List[float]:
        return [0.0] * 3584

    async def generate_query_embedding(self, text: str) -> List[float]:
        return [0.0] * 3584

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] * 3584 for _ in texts]


class MockFileStorage:
    """
    Мок MinIO-хранилища: операции выполняются в памяти.
    Используется в тестах, где нужно проверять логику БД без реального MinIO.
    """

    def generate_object_name(
        self,
        user_id: Optional[int] = None,
        namespace_id: Optional[int] = None,
        filename: Optional[str] = None,
    ) -> str:
        return f"test/{user_id}/{namespace_id}/{filename}"

    async def upload_file(
        self,
        file_content: bytes,
        object_name: str,
        content_type: str = "application/octet-stream",
        metadata: Optional[dict] = None,
    ) -> str:
        return object_name

    async def get_file(self, object_name: str) -> bytes:
        return b"mock file content from storage"

    async def delete_file(self, object_name: str) -> None:
        pass

    async def file_exists(self, object_name: str) -> bool:
        return True

    async def put_blob(self, data) -> str:
        return "mock-blob-key"

    async def get_blob(self, key: str):
        return None

    async def delete_blob(self, key: str) -> None:
        pass

    @property
    def blob_bucket_name(self) -> str:
        return "mock-blobs"


# Тестовая БД (для миграций и фикстуры engine)
TEST_DATABASE_URL = settings.DATABASE_TEST_URL or settings.DATABASE_URL

# Порядок: дочерние таблицы перед родительскими (CASCADE FK учитывается автоматически)
TRUNCATE_TABLES = (
    "chat_messages, chats, vector_embeddings, summaries, "
    "user_files, files, namespaces, users"
)


@pytest.fixture(scope="session", autouse=True)
def run_migrations():
    """Автоматически прогоняем миграции перед началом всех тестов."""
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    command.upgrade(alembic_cfg, "head")
    yield


@pytest.fixture
async def engine():
    """Движок на каждый тест."""
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
    """Основной тестовый пользователь с email/паролем."""
    user = User(
        email="testuser@example.com",
        password_hash=hash_password("testpassword"),
        username="testuser",
        full_name="Test User",
        watcher_token="test_watcher_token_123",
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def test_user_2(db_session):
    """Второй пользователь для тестов доступа (чужой namespace/file)."""
    user = User(
        email="otheruser@example.com",
        password_hash=hash_password("otherpassword"),
        username="otheruser",
        full_name="Other User",
        watcher_token="other_watcher_token_456",
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def test_namespace(db_session, test_user):
    namespace = Namespace(
        user_id=test_user.id,
        name="test_namespace",
        description="test_description",
    )
    db_session.add(namespace)
    await db_session.commit()
    await db_session.refresh(namespace)
    return namespace


@pytest.fixture
async def auth_headers(test_user):
    """JWT-заголовки для аутентификации test_user."""
    token = create_access_token(test_user.id)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def auth_headers_2(test_user_2):
    """JWT-заголовки для аутентификации test_user_2."""
    token = create_access_token(test_user_2.id)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def client(db_session):
    """
    HTTP-клиент с тестовой БД, мок LLM (intent=general_chat) и мок TaskPublisher.
    Для тестов, которым нужен специфический intent, используй make_ask_client.
    """
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_llm_provider] = lambda: MockLLMProvider()
    app.dependency_overrides[get_task_publisher] = lambda: MockTaskPublisher()
    app.dependency_overrides[get_blob_storage] = lambda: MockFileStorage()
    app.dependency_overrides[get_embedding_service] = lambda: MockEmbeddingProvider()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            yield ac
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_llm_provider, None)
        app.dependency_overrides.pop(get_task_publisher, None)
        app.dependency_overrides.pop(get_blob_storage, None)
        app.dependency_overrides.pop(get_embedding_service, None)


@pytest.fixture
def make_ask_client(db_session):
    """
    Фабрика клиентов с настраиваемым MockLLMProvider.

    Использование в тестах:
        async with make_ask_client(intent="create_namespace", entity_name="X") as client:
            response = await client.post(...)
    """

    @asynccontextmanager
    async def _make(intent: str = "general_chat", mock_storage: bool = False, **intent_params):
        async def override_get_db():
            yield db_session

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_llm_provider] = lambda: MockLLMProvider(intent=intent, **intent_params)
        app.dependency_overrides[get_task_publisher] = lambda: MockTaskPublisher()
        app.dependency_overrides[get_blob_storage] = lambda: MockFileStorage()
        app.dependency_overrides[get_embedding_service] = lambda: MockEmbeddingProvider()
        if mock_storage:
            app.dependency_overrides[get_storage_service] = lambda: MockFileStorage()

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as ac:
                yield ac
        finally:
            app.dependency_overrides.pop(get_db, None)
            app.dependency_overrides.pop(get_llm_provider, None)
            app.dependency_overrides.pop(get_task_publisher, None)
            app.dependency_overrides.pop(get_blob_storage, None)
            app.dependency_overrides.pop(get_embedding_service, None)
            app.dependency_overrides.pop(get_storage_service, None)

    return _make
