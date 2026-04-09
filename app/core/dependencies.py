from functools import lru_cache
from typing import Optional

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import decode_access_token

from app.domain.protocols import (
    UserFileRepository,
    FileRepository,
    BlobStorage,
    EmbeddingProvider,
    TaskPublisher,
    FileStorage,
    NamespaceRepository,
    UserRepository,
    VectorRepository,
    SummaryRepository,
    LLMProvider,
    ChatRepository,
    SyncRepository,
)
from app.infrastructure.db.session import get_db
from app.schemas.user import UserResponse
from app.infrastructure.repositories import (
    PgFileRepository,
    PgUserFileRepository,
    PgNamespaceRepository,
    PgUserRepository,
    PgSummaryRepository,
    PgChatRepository,
    PgSyncRepository,
)
from app.infrastructure.repositories.vector_embedding_repository import PgVectorRepository
from app.services.text_chunker import TextChunkerService
from app.infrastructure.llm.ollama_embedding import OllamaEmbeddingService
from app.infrastructure.llm.ollama_completion import OllamaCompletionService
from app.infrastructure.llm.openrouter_completion import OpenRouterCompletionService
from app.services.file_service import FileService
from app.infrastructure.llm.yandex_iam import YandexIAMService
from app.services.user_service import UserService
from app.services.namespace_service import NamespaceService
from app.infrastructure.storage.minio import MinIOStorage
from app.services.search_service import SearchService
from app.utils.file_readers import FileReaderFactory
from app.core.config import settings
from app.services.chat_service import ChatService
from app.services.content_extractor import ContentExtractorService
from app.services.summary_service import SummaryService
from app.graph.nodes.summary_agent import SummaryAgent
from app.infrastructure.workers.celery_app import celery_app
from app.infrastructure.workers.task_manager import TaskManager
from app.services.sync_service import SyncService




def get_file_repository(db: AsyncSession = Depends(get_db)) -> FileRepository:
    """Провайдер для FileRepository. Возвращает протокол, создаёт Pg-реализацию."""
    return PgFileRepository(db)


def get_user_file_repository(db: AsyncSession = Depends(get_db)) -> UserFileRepository:
    """Провайдер для UserFileRepository. Возвращает протокол, создаёт Pg-реализацию."""
    return PgUserFileRepository(db)


def get_namespace_repository(db: AsyncSession = Depends(get_db)) -> NamespaceRepository:
    """Провайдер для NamespaceRepository. Возвращает протокол, создаёт Pg-реализацию."""
    return PgNamespaceRepository(db)


def get_vector_repository(db: AsyncSession = Depends(get_db)) -> VectorRepository:
    """Провайдер для VectorRepository. Возвращает протокол, создаёт Pg-реализацию."""
    return PgVectorRepository(db)


def get_user_repository(db: AsyncSession = Depends(get_db)) -> UserRepository:
    """Провайдер для UserRepository. Возвращает протокол, создаёт Pg-реализацию."""
    return PgUserRepository(db)

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


@lru_cache
def get_embedding_service() -> EmbeddingProvider:
    """Провайдер для эмбеддингов (EmbeddingProvider). Использует локальный Ollama (Qwen)."""
    return OllamaEmbeddingService()


@lru_cache
def get_text_chunker_service() -> TextChunkerService:
    """Провайдер для TextChunkerService. Кэшируется — сервис без состояния."""
    return TextChunkerService()


@lru_cache
def get_llm_provider() -> LLMProvider:
    """Провайдер для LLM (completion). OpenRouter если ключ задан, иначе Ollama."""
    if settings.OPENROUTER_API_KEY:
        return OpenRouterCompletionService()
    return OllamaCompletionService()


@lru_cache
def get_summary_llm_provider() -> LLMProvider:
    """Провайдер LLM для суммаризации: меньшая модель с увеличенным таймаутом."""
    return OllamaCompletionService(
        model=settings.OLLAMA_SUMMARY_MODEL,
        timeout=settings.OLLAMA_SUMMARY_TIMEOUT,
    )


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


def get_summary_repository(db: AsyncSession = Depends(get_db)) -> SummaryRepository:
    """Провайдер для SummaryRepository. Возвращает протокол, создаёт Pg-реализацию."""
    return PgSummaryRepository(db)


@lru_cache
def get_task_publisher() -> TaskPublisher:
    """Провайдер постановки фоновых задач (Celery)."""
    return TaskManager(celery_app)


def get_sync_repository(db: AsyncSession = Depends(get_db)) -> SyncRepository:
    """Провайдер для SyncRepository. Возвращает протокол, создаёт Pg-реализацию."""
    return PgSyncRepository(db)



def get_namespace_service(
    db: AsyncSession = Depends(get_db),
    namespace_repository: NamespaceRepository = Depends(get_namespace_repository),
    user_file_repository: UserFileRepository = Depends(get_user_file_repository),
    sync_notifier: SyncService = Depends(get_sync_service),
) -> NamespaceService:
    """Провайдер для NamespaceService"""
    return NamespaceService(
        namespace_repository,
        db,
        user_file_repository=user_file_repository,
        sync_notifier=sync_notifier,
    )

def get_sync_service(
    storage: FileStorage = Depends(get_storage_service),
    user_file_repository: UserFileRepository = Depends(get_user_file_repository),
    file_repository: FileRepository = Depends(get_file_repository),
    namespace_repository: NamespaceRepository = Depends(get_namespace_repository),
    vector_repository: VectorRepository = Depends(get_vector_repository),
    summary_repository: SummaryRepository = Depends(get_summary_repository),
    sync_repository: SyncRepository = Depends(get_sync_repository),
    task_publisher: TaskPublisher = Depends(get_task_publisher),
    file_reader_factory: FileReaderFactory = Depends(get_file_reader_factory),
    db: AsyncSession = Depends(get_db),
) -> SyncService:
    return SyncService(
        db=db,
        storage=storage,
        file_repository=file_repository,
        user_file_repository=user_file_repository,
        namespace_repository=namespace_repository,
        sync_repository=sync_repository,
        vector_repository=vector_repository,
        summary_repository=summary_repository,
        task_publisher=task_publisher,
        file_reader_factory=file_reader_factory,
    )


def get_file_service(
    storage: FileStorage = Depends(get_storage_service),
    user_file_repository: UserFileRepository = Depends(get_user_file_repository),
    file_repository: FileRepository = Depends(get_file_repository),
    file_reader_factory: FileReaderFactory = Depends(get_file_reader_factory),
    vector_repository: VectorRepository = Depends(get_vector_repository),
    text_chunker: TextChunkerService = Depends(get_text_chunker_service),
    embedding_service: EmbeddingProvider = Depends(get_embedding_service),
    namespace_repository: NamespaceRepository = Depends(get_namespace_repository),
    summary_repository: SummaryRepository = Depends(get_summary_repository),
    task_publisher: TaskPublisher = Depends(get_task_publisher),
    sync_notifier: SyncService = Depends(get_sync_service),
    db: AsyncSession = Depends(get_db),
) -> FileService:
    return FileService(
        storage=storage,
        file_repository=file_repository,
        user_file_repository=user_file_repository,
        file_reader_factory=file_reader_factory,
        vector_repository=vector_repository,
        text_chunker=text_chunker,
        embedding_service=embedding_service,
        namespace_repository=namespace_repository,
        summary_repository=summary_repository,
        task_publisher=task_publisher,
        sync_notifier=sync_notifier,
        db=db,
    )


def create_file_service_for_celery(db: AsyncSession) -> FileService:
    """Создаёт FileService для Celery воркеров."""
    return FileService(
        storage=get_storage_service(),
        file_repository=PgFileRepository(db),
        user_file_repository=PgUserFileRepository(db),
        file_reader_factory=get_file_reader_factory(),
        vector_repository=PgVectorRepository(db),
        text_chunker=get_text_chunker_service(),
        embedding_service=get_embedding_service(),
        namespace_repository=PgNamespaceRepository(db),
        db=db,
    )




def get_user_service(
    db: AsyncSession = Depends(get_db),
    repository: UserRepository = Depends(get_user_repository),
    namespace_repository: NamespaceRepository = Depends(get_namespace_repository),
) -> UserService:
    """Провайдер для UserService"""
    return UserService(repository, db, namespace_repository)





async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=True)),
    user_service: UserService = Depends(get_user_service),
) -> UserResponse:
    """Текущий пользователь по JWT из заголовка Authorization: Bearer <token>."""
    token = credentials.credentials
    user_id = decode_access_token(token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Недействительный или истёкший токен",
        )
    user = await user_service.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь не найден",
        )
    return user


async def get_user_by_watcher_token(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=True)),
    user_service: UserService = Depends(get_user_service),
) -> UserResponse:
    """Пользователь по watcher-токену: заголовок Authorization: Bearer <watcher_token>."""
    return await user_service.get_user_by_watcher_token(credentials.credentials)


async def get_current_user_or_watcher(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    user_service: UserService = Depends(get_user_service),
) -> UserResponse:
    """
    JWT (веб/клиент) или токен Desktop Watcher — только заголовок Authorization: Bearer <token>.
    Сначала пробуем JWT; если не подошёл — ищем пользователя по watcher_token.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Требуется заголовок Authorization: Bearer <token>",
        )
    token = credentials.credentials
    user_id = decode_access_token(token)
    if user_id is not None:
        user = await user_service.get_user(user_id)
        if user:
            return user
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь не найден",
        )
    return await user_service.get_user_by_watcher_token(token)


def get_search_service(
    vector_repository: VectorRepository = Depends(get_vector_repository),
) -> SearchService:
    """Провайдер для SearchService. Векторный репозиторий внедряется через конструктор."""
    return SearchService(vector_repository=vector_repository)


def get_blob_storage() -> BlobStorage:
    """Провайдер BlobStorage (Claim Check). Тот же экземпляр MinIOStorage."""
    return _get_minio_storage()


def get_chat_repository(db: AsyncSession = Depends(get_db)) -> ChatRepository:
    """Провайдер для ChatRepository."""
    return PgChatRepository(db)


def get_content_extractor() -> ContentExtractorService:
    """Провайдер для ContentExtractorService."""
    return ContentExtractorService()


def get_summary_agent(
    llm_service: LLMProvider = Depends(get_summary_llm_provider),
    text_chunker: TextChunkerService = Depends(get_text_chunker_service),
) -> SummaryAgent:
    """Провайдер для SummaryAgent. Использует меньшую модель с увеличенным таймаутом."""
    return SummaryAgent(llm_service=llm_service, text_chunker=text_chunker)


def get_summary_service(
    db: AsyncSession = Depends(get_db),
    user_file_repository: UserFileRepository = Depends(get_user_file_repository),
    file_repository: FileRepository = Depends(get_file_repository),
    summary_repository: SummaryRepository = Depends(get_summary_repository),
    content_extractor: ContentExtractorService = Depends(get_content_extractor),
    file_service: FileService = Depends(get_file_service),
) -> SummaryService:
    """Провайдер для SummaryService (без агента; дирижёр вызывает агента отдельно)."""
    return SummaryService(
        db=db,
        user_file_repository=user_file_repository,
        file_repository=file_repository,
        summary_repository=summary_repository,
        file_service=file_service,
        content_extractor=content_extractor,
    )


def create_summary_service_for_celery(db: AsyncSession) -> SummaryService:
    """Создаёт SummaryService для Celery воркеров (без агента)."""
    file_service = create_file_service_for_celery(db)
    return SummaryService(
        db=db,
        user_file_repository=PgUserFileRepository(db),
        file_repository=PgFileRepository(db),
        summary_repository=PgSummaryRepository(db),
        file_service=file_service,
        content_extractor=ContentExtractorService(),
    )


def get_chat_service(
    db: AsyncSession = Depends(get_db),
    file_repository: FileRepository = Depends(get_file_repository),
    user_file_repository: UserFileRepository = Depends(get_user_file_repository),
    vector_repository: VectorRepository = Depends(get_vector_repository),
    search_service: SearchService = Depends(get_search_service),
    summary_service: SummaryService = Depends(get_summary_service),
    summary_agent: SummaryAgent = Depends(get_summary_agent),
    file_reader_factory: FileReaderFactory = Depends(get_file_reader_factory),
    text_chunker: TextChunkerService = Depends(get_text_chunker_service),
    embedding_service: EmbeddingProvider = Depends(get_embedding_service),
    file_service: FileService = Depends(get_file_service),
    llm_service: LLMProvider = Depends(get_llm_provider),
    blob_storage: BlobStorage = Depends(get_blob_storage),
    namespace_service: NamespaceService = Depends(get_namespace_service),
    content_extractor: ContentExtractorService = Depends(get_content_extractor),
    task_publisher: TaskPublisher = Depends(get_task_publisher),
    chat_repository: ChatRepository = Depends(get_chat_repository),
) -> ChatService:
    """Провайдер для ChatService"""
    return ChatService(
        db=db,
        file_repository=file_repository,
        user_file_repository=user_file_repository,
        vector_repository=vector_repository,
        search_service=search_service,
        summary_service=summary_service,
        summary_agent=summary_agent,
        file_reader_factory=file_reader_factory,
        text_chunker=text_chunker,
        embedding_service=embedding_service,
        file_service=file_service,
        llm_service=llm_service,
        blob_storage=blob_storage,
        namespace_service=namespace_service,
        content_extractor=content_extractor,
        task_publisher=task_publisher,
        chat_repository=chat_repository,
    )