from pydantic import BaseModel, Field, model_validator
from datetime import datetime
from typing import Optional, List

from app.schemas.file import FileStructureItem, FileWithUrl


class NamespaceCreate(BaseModel):
    """Схема создания namespace"""
    name: str = Field(..., min_length=1, max_length=255, description="Название namespace")
    description: Optional[str] = Field(None, max_length=1000, description="Описание namespace")
    parent_id: Optional[int] = Field(None, description="ID родительского пространства")


class SyncNamespaceCreate(BaseModel):
    """Схема создания namespace для desktop sync."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Название namespace")
    vault_name: Optional[str] = Field(None, min_length=1, max_length=255, description="Имя корневого vault")
    description: Optional[str] = Field(None, max_length=1000, description="Описание namespace")
    parent_id: Optional[int] = Field(None, description="ID родительского пространства")
    relative_path: Optional[str] = Field(
        None,
        min_length=1,
        max_length=1024,
        description="Относительный путь namespace внутри vault для sync-сценариев",
    )

    @model_validator(mode="after")
    def validate_payload(self) -> "SyncNamespaceCreate":
        if self.relative_path:
            return self
        if self.name:
            return self
        raise ValueError("Нужно передать либо name, либо relative_path")


class NamespaceUpdate(BaseModel):
    """Схема обновления namespace"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)


class NamespaceMoveRequest(BaseModel):
    """Схема перемещения namespace."""
    target_parent_id: int = Field(..., description="ID нового родительского пространства")


class NamespaceRenameRequest(BaseModel):
    """Схема переименования namespace."""
    new_name: str = Field(..., min_length=1, max_length=255, description="Новое имя пространства")


class NamespaceListItem(BaseModel):
    """Схема namespace для списка (без файлов, с количеством)"""
    id: int
    user_id: int
    name: str
    parent_id: Optional[int]
    kind: str
    description: Optional[str]
    created_at: datetime
    files_count: int = Field(0, description="Количество файлов в namespace")

    class Config:
        from_attributes = True


class NamespaceResponse(BaseModel):
    """Схема ответа с детальной информацией о namespace (с файлами)"""
    id: int
    user_id: int
    name: str
    parent_id: Optional[int]
    kind: str
    description: Optional[str]
    created_at: datetime
    files: List[FileWithUrl] = Field(default_factory=list, description="Список файлов с ссылками на скачивание")

    class Config:
        from_attributes = True


class NamespaceTreeItem(BaseModel):
    id: int
    user_id: int
    name: str
    parent_id: Optional[int]
    kind: str = Field(..., description="Тип пространства: vault_root, regular, inbox, trash")
    description: Optional[str]
    created_at: datetime
    files_count: int = 0
    children: List["NamespaceTreeItem"] = Field(default_factory=list)


class NamespaceStructureItem(BaseModel):
    """Элемент структуры: namespace как «папка» с файлами"""
    id: int = Field(..., description="ID пространства знаний")
    name: str = Field(..., description="Имя пространства (имя папки на диске)")
    parent_id: Optional[int] = Field(None, description="ID родительского пространства")
    kind: str = Field("regular", description="Тип пространства: vault_root, regular, inbox, trash")
    files: list[FileStructureItem] = Field(default_factory=list, description="Файлы в этом пространстве")

    class Config:
        from_attributes = True




NamespaceTreeItem.model_rebuild()
