from functools import lru_cache
from typing import Callable
from fastapi import Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.domain.protocols import (
    AsyncFileRepository,
    BlobStorage,
    EmbeddingProvider,
    FileStorage,
    NamespaceRepository,
    VectorRepository,
    WatcherTaskPublisher,
    LLMProvider,
)
from app.infrastructure.db.session import get_db
from app.schemas.user import UserResponse
from app.infrastructure.repositories import (
    UserRepository,
    NamespaceRepository,
    AsyncFileRepository,
    PgVectorRepository,
)
from app.services.text_chunker import TextChunkerService
from app.infrastructure.llm.yandex_embedding import YandexEmbeddingService
from app.infrastructure.llm.yandex_completion import YandexCompletionService
from app.services.file_service import FileService
from app.infrastructure.llm.yandex_iam import YandexIAMService
from app.services.user_service import UserService
from app.services.namespace_service import NamespaceService
from app.infrastructure.message_broker.rabbitmq import RabbitMQService
from app.infrastructure.storage.minio import MinIOStorage
from app.services.websocket_manager import WebSocketManager
from app.services.search_service import SearchService
from app.utils.file_readers import FileReaderFactory
from app.core.config import settings
from app.core.exceptions import NotFoundError
from app.services.chat_service import ChatService




def get_file_repository(db: AsyncSession = Depends(get_db)) -> AsyncFileRepository:
    """Провайдер для AsyncFileRepository"""
    return AsyncFileRepository(db)


def get_namespace_repository(db: AsyncSession = Depends(get_db)) -> NamespaceRepository:
    """Провайдер для NamespaceRepository"""
    return NamespaceRepository(db)


def get_vector_repository(db: AsyncSession = Depends(get_db)) -> VectorRepository:
    """Провайдер для VectorRepository"""
    return PgVectorRepository(db)


def get_user_repository(db: AsyncSession = Depends(get_db)) -> UserRepository:
    """Провайдер для UserRepository"""
    return UserRepository(db)

@lru_cache
def get_file_reader_factory() -> FileReaderFactory:
    """Провайдер для FileReaderFactory. Кэшируется — фабрика без состояния."""
    return FileReaderFactory()


@lru_cache
def get_yandex_iam_service() -> YandexIAMService:
    """
    Провайдер для YandexIAMService.
    Кэшируется глобально для переиспользования IAM токенов.
    """
    return YandexIAMService()


def get_embedding_service(iam_service: YandexIAMService = Depends(get_yandex_iam_service)) -> EmbeddingProvider:
    """
    Провайдер для эмбеддингов (EmbeddingProvider).
    IAM и прочие детали Яндекса скрыты внутри; снаружи — только абстракция.
    """
    return YandexEmbeddingService(iam_service=iam_service)


@lru_cache
def get_text_chunker_service() -> TextChunkerService:
    """Провайдер для TextChunkerService. Кэшируется — сервис без состояния."""
    return TextChunkerService()


def get_llm_provider(iam_service: YandexIAMService = Depends(get_yandex_iam_service)) -> LLMProvider:
    """Провайдер для LLM (completion). Использует общий IAM сервис."""
    return YandexCompletionService(iam_service=iam_service)


@lru_cache
def _get_minio_storage() -> MinIOStorage:
    """Один экземпляр MinIOStorage с двумя бакетами (файлы + блобы)."""
    return MinIOStorage(
        bucket=settings.MINIO_BUCKET_NAME,
        blob_bucket=settings.MINIO_BUCKET_TEMP_BLOBS,
    )


def get_storage_service() -> FileStorage:
    """Провайдер FileStorage (постоянные файлы пользователя)."""
    return _get_minio_storage()


def get_file_service(
    storage: FileStorage = Depends(get_storage_service),
    file_repository: AsyncFileRepository = Depends(get_file_repository),
    file_reader_factory: FileReaderFactory = Depends(get_file_reader_factory),
    vector_repository: VectorRepository = Depends(get_vector_repository),
    text_chunker: TextChunkerService = Depends(get_text_chunker_service),
    embedding_service: EmbeddingProvider = Depends(get_embedding_service),
    namespace_repository: NamespaceRepository = Depends(get_namespace_repository),
    db: AsyncSession = Depends(get_db),
) -> FileService:
    """
    Провайдер для FileService. Зависит от Storage, AsyncFileRepository, сессии БД и др.
    """
    return FileService(
        storage=storage,
        file_repository=file_repository,
        file_reader_factory=file_reader_factory,
        vector_repository=vector_repository,
        text_chunker=text_chunker,
        embedding_service=embedding_service,
        namespace_repository=namespace_repository,
        db=db,
    )


def create_file_service_for_celery(db: Session) -> FileService:
    """
    Фабрика FileService для Celery: синхронная сессия, commit делает воркер снаружи.
    Сессия db не передаётся в сервис (self.db = None), транзакцией управляет воркер.
    """
    return FileService(
        storage=get_storage_service(),
        file_repository=None,
        file_reader_factory=get_file_reader_factory(),
        vector_repository=PgVectorRepository(db),
        text_chunker=get_text_chunker_service(),
        embedding_service=get_embedding_service(get_yandex_iam_service()),
        namespace_repository=None,
        db=None,
    )


def get_rabbitmq_service() -> WatcherTaskPublisher:
    """
    Провайдер для публикации задач в очередь (Watcher).
    НЕ кэшируется: создает новое подключение в каждом методе.
    """
    return RabbitMQService()


@lru_cache
def get_websocket_manager() -> WebSocketManager:
    """
    Провайдер для WebSocketManager.
    Кэшируется глобально (singleton): один менеджер для всего приложения.
    """
    return WebSocketManager()


def get_user_service(
    db: AsyncSession = Depends(get_db),
    repository: UserRepository = Depends(get_user_repository),
) -> UserService:
    """Провайдер для UserService"""
    return UserService(repository, db)


def get_namespace_service(
    db: AsyncSession = Depends(get_db),
    namespace_repository: NamespaceRepository = Depends(get_namespace_repository),
) -> NamespaceService:
    """Провайдер для NamespaceService"""
    return NamespaceService(namespace_repository, db)


async def get_user_by_telegram_id(
    telegram_id: int = Query(..., description="Telegram ID пользователя"),
    user_service: UserService = Depends(get_user_service),
) -> UserResponse:
    """Dependency: пользователь по Telegram ID (UserResponse). NotFoundError если не найден."""
    return await user_service.get_user_by_telegram_id(telegram_id)


async def get_user_by_watcher_token(
    token: str = Query(..., description="Токен аутентификации Watcher"),
    user_service: UserService = Depends(get_user_service),
) -> UserResponse:
    """Dependency: пользователь по Watcher токену (UserResponse). NotFoundError если не найден."""
    return await user_service.get_user_by_watcher_token(token)


@lru_cache
def get_search_service_factory() -> Callable[[AsyncSession], SearchService]:
    """Фабрика SearchService по сессии (для графа /ask)."""
    return lambda db: SearchService(db)


def get_blob_storage() -> BlobStorage:
    """Провайдер BlobStorage (Claim Check). Тот же экземпляр MinIOStorage."""
    return _get_minio_storage()


def get_chat_service(
    file_reader_factory: FileReaderFactory = Depends(get_file_reader_factory),
    text_chunker: TextChunkerService = Depends(get_text_chunker_service),
    embedding_service: EmbeddingProvider = Depends(get_embedding_service),
    file_service: FileService = Depends(get_file_service),
    llm_service: LLMProvider = Depends(get_llm_provider),
    search_service_factory: Callable[[AsyncSession], SearchService] = Depends(get_search_service_factory),
    blob_storage: BlobStorage = Depends(get_blob_storage),
) -> ChatService:
    return ChatService(
        file_reader_factory=file_reader_factory,
        text_chunker=text_chunker,
        embedding_service=embedding_service,
        file_service=file_service,
        llm_service=llm_service,
        search_service_factory=search_service_factory,
        blob_storage=blob_storage,
    )