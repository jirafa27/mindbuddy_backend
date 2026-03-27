from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ChatListItem(BaseModel):
    """Схема чата для списка."""
    id: int
    user_id: int
    name: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    messages_count: int = Field(0, description="Количество сообщений в чате")


class ChatUpdate(BaseModel):
    """Схема обновления чата (название)."""
    name: Optional[str] = Field(None, max_length=255, description="Название чата")


class ChatMessageItem(BaseModel):
    """Схема сообщения в чате."""
    id: int
    chat_id: int
    role: str = Field(..., description="user | assistant")
    text: str
    file_ids: List[int] = Field(default_factory=list)
    namespace_id: Optional[int] = Field(None, description="ID пространства, в контексте которого отправлено сообщение")
    created_at: datetime
