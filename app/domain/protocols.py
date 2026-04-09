from typing import Any, Dict, Protocol, Optional, List, Tuple, Sequence

from app.domain.entities import (
    FileEntity,
    UserFileEntity,
    UserEntity,
    NamespaceEntity,
    ChunkEntity,
    SearchResultRow,
    ParsedContent,
    SummaryEntity,
    ChatEntity,
    ChatMessageEntity,
    SyncCommandEntity,
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

    async def get_by_source_url(self, source_url: str) -> Optional[FileEntity]:
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

    async def update_content_metadata(
        self,
        file_id: int,
        *,
        content_hash: str,
        media_metadata: Optional[dict] = None,
        transcript_text: Optional[str] = None,
    ) -> Optional[FileEntity]:
        """Обновляет content_hash и media_metadata файла."""
        ...


class UserFileRepository(Protocol):
    """
    Протокол репозитория пользовательских файлов
    """
    async def get_by_id(self, user_file_id: int) -> Optional[UserFileEntity]:
        ...

    async def list_ids_by_user_and_namespace(
        self, user_id: int, namespace_id: int
    ) -> Sequence[int]:
        """user_files.id в пространстве пользователя (порядок — по created_at)."""
        ...

    async def create(self, user_file: UserFileEntity) -> UserFileEntity:
        ...

    async def delete(self, user_file_id: int) -> None:
        ...

    async def update_namespace(self, user_file_id: int, namespace_id: Optional[int] = None) -> Optional[UserFileEntity]:
        ...

    async def find_by_source_url(self, source_url: str, user_id: int) -> Optional[UserFileEntity]:
        ...

    async def find_by_content_hash(self, content_hash: str, user_id: int) -> Optional[UserFileEntity]:
        ...

    async def find_by_user_and_file(self, user_id: int, file_id: int, namespace_id: Optional[int] = None) -> Optional[UserFileEntity]:
        ...

    async def count_by_file_id(self, file_id: int) -> int:
        """Количество UserFile, ссылающихся на данный File (content file)."""
        ...

    async def update_file_id(self, user_file_id: int, new_file_id: int) -> Optional[UserFileEntity]:
        """Переключает UserFile на другой File (для Copy-on-Write при редактировании)."""
        ...

    async def update_custom_title(self, user_file_id: int, new_title: str) -> Optional[UserFileEntity]:
        """Обновляет отображаемое имя файла (custom_title)."""
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

    async def get_by_name_and_parent(
        self, *, user_id: int, parent_id: Optional[int], name: str
    ) -> Optional[NamespaceEntity]:
        ...

    async def create(
        self,
        name: str,
        user_id: int,
        parent_id: Optional[int] = None,
        kind: str = "regular",
        description: Optional[str] = None,
    ) -> NamespaceEntity:
        ...

    async def get_by_user_with_files(self, user_id: int) -> List[NamespaceEntity]:
        ...

    async def get_children(
        self, *, user_id: int, parent_id: Optional[int]
    ) -> List[NamespaceEntity]:
        ...

    async def get_descendant_ids(self, *, user_id: int, namespace_id: int) -> List[int]:
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

    async def delete_by_file_id(self, file_id: int) -> int:
        """Удаляет все эмбеддинги для файла."""
        ...

    async def search(
        self,
        query_embedding: Optional[List[float]],
        user_id: int,
        limit: int = 5,
        namespace_id: Optional[int] = None,
        file_ids: Optional[List[int]] = None,
        *,
        sql: Optional[str] = None,
    ) -> List[SearchResultRow]:
        ...


class ChatRepository(Protocol):
    """Протокол репозитория чатов."""

    async def create_chat(
        self, user_id: int, name: Optional[str] = None
    ) -> ChatEntity:
        """Создать новый чат."""
        ...

    async def get_chat_by_id(self, chat_id: int, user_id: int) -> Optional[ChatEntity]:
        """Получить чат по id с проверкой владельца."""
        ...

    async def update_chat_name(
        self, chat_id: int, user_id: int, name: Optional[str]
    ) -> Optional[ChatEntity]:
        """Обновить название чата."""
        ...

    async def add_message(
        self,
        chat_id: int,
        role: str,
        text: str,
        file_ids: Optional[List[int]] = None,
    ) -> ChatMessageEntity:
        """Добавить сообщение в чат."""
        ...

    async def get_messages(
        self,
        chat_id: int,
        user_id: int,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ChatMessageEntity]:
        """Получить историю сообщений чата с пагинацией."""
        ...

    async def get_messages_count(self, chat_id: int, user_id: int) -> int:
        """Общее количество сообщений в чате (для пагинации)."""
        ...

    async def get_user_chats(
        self,
        user_id: int,
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[List[Tuple[ChatEntity, int]], int]:
        """Список чатов пользователя с количеством сообщений и общее число чатов."""
        ...

    async def delete_chat(self, chat_id: int, user_id: int) -> bool:
        """Удалить чат и все его сообщения. Возвращает True если удалён, False если не найден или не принадлежит пользователю."""
        ...

    async def get_pending_action(self, chat_id: int) -> Optional[Dict[str, Any]]:
        """Получить отложенное действие чата (pending_action)."""
        ...

    async def set_pending_action(self, chat_id: int, action: Dict[str, Any]) -> None:
        """Сохранить отложенное действие в чате."""
        ...

    async def clear_pending_action(self, chat_id: int) -> None:
        """Очистить отложенное действие чата."""
        ...

    async def update_context(self, chat_id: int, context: Dict[str, Any]) -> None:
        """Обновить персистентный контекст диалога (ConversationContext)."""
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

    async def delete_by_file_id(self, file_id: int) -> int:
        """Удаляет все суммаризации для файла."""
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

    def send_bulk_edit_task(
        self,
        file_ids: list[int],
        user_id: int,
        edit_instruction: str,
        namespace_id: Optional[int] = None,
    ) -> Optional[str]:
        """Постановка задачи на массовое редактирование файлов. Возвращает task_id или None."""
        ...


class SyncRepository(Protocol):
    """
    Репозиторий команд синхронизации
    """

    async def create_command(
        self,
        *,
        user_id: int,
        user_file_id: int,
        command_type: str,
        payload_json: dict,
        status: str = "pending",
    ) -> SyncCommandEntity:
        ...

    async def get_pending_commands(
        self, user_id: int, limit: int = 100
    ) -> List[SyncCommandEntity]:
        ...

    async def get_command(
        self, command_id: int, user_id: int
    ) -> Optional[SyncCommandEntity]:
        ...

    async def ack_command(
        self, command_id: int, user_id: int, status: str
    ) -> Optional[SyncCommandEntity]:
        ...

    async def get_namespaces_with_files(self, user_id: int) -> List[Any]:
        ...


class FileSyncNotifier(Protocol):
    """Протокол для уведомления подсистемы синхронизации об изменениях файлов.
    Позволяет FileService / NamespaceService создавать sync-команды,
    не зная о конкретной реализации (SyncService)."""

    async def add_upsert_command_to_queue(
        self,
        *,
        user_file_id: int,
        user_id: int,
        command_type: Any = None,
        vault_relative_path: Optional[str] = None,
    ) -> Any:
        """Добавляет команду на обновление файла в очередь синхронизации."""
        ...

    async def add_trash_command_to_queue(
        self,
        *,
        user_file_id: int,
        user_id: int,
    ) -> None:
        """Добавляет команду на перемещение файла в корзину в очередь синхронизации."""
        ...


    async def add_rename_command_to_queue(
        self,
        *,
        user_file_id: int,
        user_id: int,
        new_title: str,
    ) -> Any:
        """Добавляет команду на переименование файла в очередь синхронизации."""
        ...

    async def get_file_version(
        self,
        user_file_id: int,
        user_id: int,
    ) -> Any:
        """Получает версию файла."""
        ...

    async def assert_can_save(
        self,
        *,
        user_file_id: int,
        user_id: int,
        base_hash: Optional[str] = None,
        force_overwrite: bool = False,
    ) -> None:
        """
        Проверяет, был ли файл изменен после открытия редактора
        """
        ...


class ContentParser(Protocol):
    """Протокол парсера контента"""

    def can_handle(self, url: str) -> bool:
        """Проверяет, может ли парсер обработать данный URL."""
        ...

    async def parse(self, url: str) -> ParsedContent:
        """Извлекает контент из URL"""
        ...
