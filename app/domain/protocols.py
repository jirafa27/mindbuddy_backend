"""Domain protocols (ports): контракты для внешних адаптеров."""
from typing import Any, Protocol, Optional, List

from app.domain.entities import (
    FileEntity,
    UserEntity,
    NamespaceEntity,
    ChunkEntity,
    SearchResultRow,
)


class EmbeddingProvider(Protocol):
    """Поставщик эмбеддингов (документ и запрос)."""

    async def generate_embedding(self, text: str) -> List[float]:
        ...
    
    async def generate_query_embedding(self, text: str) -> List[float]:
        ...

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        ...


class LLMProvider(Protocol):
    """LLM для генерации текста (completion)."""

    async def complete(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        ...


class BlobStorage(Protocol):
    """Для быстрой переброски данных между агентами (Claim Check)."""

    @property
    def blob_bucket_name(self) -> str:
        """Имя бакета для логирования."""
        ...

    async def put_blob(self, data: Any) -> str:
        ...
    async def get_blob(self, key: str) -> Any:
        ...
    async def delete_blob(self, key: str) -> None:
        ...


class FileStorage(Protocol):
    """Для работы с постоянными файлами пользователя."""

    def generate_object_name(
        self,
        user_id: Optional[int] = None,
        namespace_id: Optional[int] = None,
        filename: Optional[str] = None,
    ) -> str:
        ...

    async def upload_file(
        self,
        file_content: bytes,
        object_name: str,
        content_type: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        ...

    async def download_file(self, object_name: str) -> bytes:
        ...

    async def delete_file(self, object_name: str) -> None:
        ...

    def get_file_url(self, object_name: str, expires_in: int = 3600) -> str:
        ...


class AsyncFileRepository(Protocol):
    """Асинхронный репозиторий файлов."""

    async def get_by_id(self, file_id: int) -> Optional[FileEntity]:
        ...

    async def create(
        self,
        user_id: int,
        namespace_id: int,
        filename: str,
        file_path: str,
        file_type: str,
        file_size: int,
    ) -> FileEntity:
        ...

    async def delete(self, file: FileEntity) -> None:
        ...


class UserRepository(Protocol):
    """Репозиторий пользователей."""

    async def get_by_id(self, user_id: int) -> Optional[UserEntity]:
        ...

    async def get_by_telegram_id(self, telegram_id: int) -> Optional[UserEntity]:
        ...

    async def get_by_watcher_token(self, token: str) -> Optional[UserEntity]:
        ...

    async def create(
        self,
        telegram_id: Optional[int] = None,
        username: Optional[str] = None,
        full_name: Optional[str] = None,
    ) -> UserEntity:
        ...


class NamespaceRepository(Protocol):
    """Репозиторий пространств знаний."""

    async def get_by_id(self, namespace_id: int) -> Optional[NamespaceEntity]:
        ...

    async def get_by_user_with_files(self, user_id: int) -> List[NamespaceEntity]:
        ...


class VectorRepository(Protocol):
    """Синхронный репозиторий эмбеддингов (create_batch, search)."""

    def create_batch(
        self,
        file_id: int,
        chunks: List[str],
        embeddings: List[List[float]],
        namespace_id: Optional[int] = None,
    ) -> List[ChunkEntity]:
        ...

    def search_by_embedding(
        self,
        query_embedding: List[float],
        limit: int = 5,
        namespace_id: Optional[int] = None,
    ) -> List[SearchResultRow]:
        ...


class AsyncVectorEmbeddingRepository(Protocol):
    """Асинхронное выполнение SQL векторного поиска."""

    async def execute_search_sql(
        self,
        sql: str,
        query_embedding: Optional[List[float]],
        user_id: int,
        limit: int = 5,
        namespace_id: Optional[int] = None,
    ) -> List[SearchResultRow]:
        ...


class WatcherTaskPublisher(Protocol):
    """Публикация задач для Desktop Watcher (очередь сообщений)."""

    def send_watcher_task(
        self,
        file_id: int,
        user_id: int,
        filename: str,
        file_type: str,
        file_size: int,
        download_url: str,
        local_path: Optional[str] = None,
    ) -> bool:
        ...
