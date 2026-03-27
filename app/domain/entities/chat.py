from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.enums import ChatMessageRole


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
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
