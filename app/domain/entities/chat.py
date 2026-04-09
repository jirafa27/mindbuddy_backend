from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.enums import ChatMessageRole


@dataclass
class ConversationContext:
    """Персистентный контекст диалога — хранится в Chat.context (JSONB).

    Работает как «рабочая директория» чата: всегда явный, не требует парсинга.
    Обновляется ChatService после каждого прохода графа.
    """
    active_namespace_id: Optional[int] = None
    active_file_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_namespace_id": self.active_namespace_id,
            "active_file_ids": self.active_file_ids,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ConversationContext":
        if not data:
            return cls()
        return cls(
            active_namespace_id=data.get("active_namespace_id"),
            active_file_ids=data.get("active_file_ids") or [],
        )


@dataclass
class ChatMessageEntity:
    """Сообщение в чате."""
    id: int
    chat_id: int
    role: ChatMessageRole
    text: str
    file_ids: List[int] = field(default_factory=list)
    namespace_id: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class ChatEntity:
    """Чат (диалог пользователя)."""
    id: int
    user_id: int
    name: Optional[str] = None
    pending_action: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
