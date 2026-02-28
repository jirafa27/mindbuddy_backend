from typing import Any, Protocol, Optional, List

from app.domain.entities import (
    FileEntity,
    UserFileEntity,
    UserEntity,
    NamespaceEntity,
    ChunkEntity,
    SearchResultRow,
    ParsedContent,
    SummaryEntity,
)


class EmbeddingProvider(Protocol):
    """
    Протокол для работы с эмбеддингами
    """

    async def generate_embedding(self, text: str) -> List[float]:
        ...
    
    async def generate_query_embedding(self, text: str) -> List[float]:
        ...

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        ...


class LLMProvider(Protocol):
    """
    Протокол для работы с LLM
    """

    async def complete(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        ...


class BlobStorage(Protocol):
    """
    Протокол для работы с хранилищем, чтобы обмениваться данными между агентами
    """

    @property
    def blob_bucket_name(self) -> str:
        """
        Имя бакета для логирования.
        """
        ...

    async def put_blob(self, data: Any) -> str:
        ...
    async def get_blob(self, key: str) -> Any:
        ...
    async def delete_blob(self, key: str) -> None:
        ...


class FileStorage(Protocol):
    """Протокол для работы с постоянными файлами пользователя."""

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


class FileRepository(Protocol):
    """
    Протокол репозитория файлов
    """

    async def get_by_id(self, file_id: int) -> Optional[FileEntity]:
        ...

    async def get_by_content_hash(self, content_hash: str) -> Optional[FileEntity]:
        ...

    async def create(
        self,
        content_hash: str,
        source_url: Optional[str] = None,
        transcript_text: Optional[str] = None,
        file_path: Optional[str] = None,
        media_metadata: Optional[dict] = None,
        processing_status: str = "pending",
    ) -> FileEntity:
        ...

    async def delete(self, file: FileEntity) -> None:
        ...


class UserFileRepository(Protocol):
    """
    Протокол репозитория пользовательских файлов
    """
    async def get_by_id(self, user_file_id: int) -> Optional[UserFileEntity]:
        ...

    async def create(self, user_file: UserFileEntity) -> UserFileEntity:
        ...

    async def delete(self, user_file: UserFileEntity) -> None:
        ...

    async def update_namespace(self, user_file_id: int, namespace_id: Optional[int] = None) -> Optional[UserFileEntity]:
        ...

    async def find_by_source_url(self, source_url: str, user_id: int) -> Optional[UserFileEntity]:
        ...

    async def find_by_content_hash(self, content_hash: str, user_id: int) -> Optional[UserFileEntity]:
        ...

    async def find_by_user_and_file(self, user_id: int, file_id: int) -> Optional[UserFileEntity]:
        ...


class UserRepository(Protocol):
    """Протокол репозитория пользователей."""

    async def get_by_id(self, user_id: int) -> Optional[UserEntity]:
        ...

    async def get_by_email(self, email: str) -> Optional[UserEntity]:
        ...

    async def get_by_watcher_token(self, token: str) -> Optional[UserEntity]:
        ...

    async def create(
        self,
        email: str,
        password_hash: str,
        full_name: Optional[str] = None,
        watcher_token: Optional[str] = None,
    ) -> UserEntity:
        ...


class NamespaceRepository(Protocol):
    """Протокол репозитория пространств знаний."""

    async def get_by_id(self, namespace_id: int) -> Optional[NamespaceEntity]:
        ...

    async def get_by_name_and_user(self, name: str, user_id: int) -> Optional[NamespaceEntity]:
        ...

    async def create(
        self,
        name: str,
        user_id: int,
        description: Optional[str] = None,
    ) -> NamespaceEntity:
        ...

    async def get_by_user_with_files(self, user_id: int) -> List[NamespaceEntity]:
        ...

    async def update(self, namespace: NamespaceEntity) -> NamespaceEntity:
        ...

    async def delete(self, id: int) -> None:
        ...


class VectorRepository(Protocol):
    """Репозиторий эмбеддингов: создание чанков и семантический поиск."""

    async def create_batch(
        self,
        file_id: int,
        chunks: List[str],
        embeddings: List[List[float]],
        namespace_id: Optional[int] = None,
    ) -> List[ChunkEntity]:
        ...

    async def search(
        self,
        query_embedding: List[float],
        user_id: int,
        limit: int = 5,
        namespace_id: Optional[int] = None,
    ) -> List[SearchResultRow]:
        ...


class SummaryRepository(Protocol):
    """Протокол репозитория суммаризаций."""

    async def get_by_file_id(self, file_id: int) -> Optional[SummaryEntity]:
        ...

    async def get_by_file_and_lookup_key(
        self, file_id: int, lookup_key: str
    ) -> Optional[SummaryEntity]:
        ...

    async def create(
        self,
        file_id: int,
        text: str,
        lookup_key: str = "standard_v1",
        used_prompt: Optional[str] = None,
        model_name: Optional[str] = "yandexgpt",
        **kwargs: Any,
    ) -> SummaryEntity:
        ...


class TaskPublisher(Protocol):
    """Общий протокол постановки фоновых задач (Celery и т.д.)."""

    def send_embeddings_task(
        self,
        content_file_id: int,
        text: str,
        namespace_id: Optional[int],
        filename: str,
        user_file_id: int,
    ) -> Optional[str]:
        """Постановка задачи на построение эмбеддингов. Возвращает task_id или None."""
        ...

    def send_summary_url_task(self, url: str, user_id: int) -> Optional[str]:
        """Постановка задачи на суммаризацию по URL. Возвращает task_id или None."""
        ...


class ContentParser(Protocol):
    """Протокол парсера контента"""

    def can_handle(self, url: str) -> bool:
        """Проверяет, может ли парсер обработать данный URL."""
        ...

    async def parse(self, url: str) -> ParsedContent:
        """Извлекает контент из URL"""
        ...
