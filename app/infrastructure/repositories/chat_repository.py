from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.db.models import Chat, ChatMessage
from app.domain.entities import ChatEntity, ChatMessageEntity
from app.core.enums import ChatMessageRole


class PgChatRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    def _to_chat_entity(self, model: Chat) -> ChatEntity:
        return ChatEntity(
            id=model.id,
            user_id=model.user_id,
            name=model.name,
            pending_action=model.pending_action,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    def _to_message_entity(self, model: ChatMessage) -> ChatMessageEntity:
        return ChatMessageEntity(
            id=model.id,
            chat_id=model.chat_id,
            role=ChatMessageRole(model.role),
            text=model.text,
            file_ids=model.file_ids or [],
            namespace_id=model.namespace_id,
            created_at=model.created_at,
        )

    async def create_chat(
        self, user_id: int, name: Optional[str] = None
    ) -> ChatEntity:
        chat = Chat(user_id=user_id, name=name)
        self.db.add(chat)
        await self.db.flush()
        return self._to_chat_entity(chat)

    async def get_chat_by_id(self, chat_id: int, user_id: int) -> Optional[ChatEntity]:
        result = await self.db.execute(
            select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_chat_entity(row)

    async def update_chat_name(
        self, chat_id: int, user_id: int, name: Optional[str]
    ) -> Optional[ChatEntity]:
        result = await self.db.execute(
            select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id)
        )
        chat = result.scalar_one_or_none()
        if chat is None:
            return None
        chat.name = name
        self.db.add(chat)
        await self.db.flush()
        return self._to_chat_entity(chat)

    async def add_message(
        self,
        chat_id: int,
        role: str,
        text: str,
        file_ids: Optional[List[int]] = None,
        namespace_id: Optional[int] = None,
    ) -> ChatMessageEntity:
        msg = ChatMessage(
            chat_id=chat_id,
            role=role,
            text=text,
            file_ids=file_ids or [],
            namespace_id=namespace_id,
        )
        self.db.add(msg)
        await self.db.flush()
        return self._to_message_entity(msg)

    async def get_messages(
        self,
        chat_id: int,
        user_id: int,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ChatMessageEntity]:
        stmt = (
            select(
                ChatMessage.id,
                ChatMessage.chat_id,
                ChatMessage.role,
                ChatMessage.text,
                ChatMessage.file_ids,
                ChatMessage.namespace_id,
                ChatMessage.created_at,
            )
            .join(Chat, ChatMessage.chat_id == Chat.id)
            .where(Chat.id == chat_id, Chat.user_id == user_id)
            .order_by(ChatMessage.created_at.asc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        rows = result.all()
        return [
            ChatMessageEntity(
                id=r.id,
                chat_id=r.chat_id,
                role=ChatMessageRole(r.role),
                text=r.text,
                file_ids=r.file_ids or [],
                namespace_id=r.namespace_id,
                created_at=r.created_at,
            )
            for r in rows
        ]

    async def get_messages_count(self, chat_id: int, user_id: int) -> int:
        result = await self.db.execute(
            select(func.count(ChatMessage.id))
            .join(Chat, ChatMessage.chat_id == Chat.id)
            .where(Chat.id == chat_id, Chat.user_id == user_id)
        )
        return result.scalar() or 0

    async def get_user_chats(
        self,
        user_id: int,
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[List[Tuple[ChatEntity, int]], int]:
        msg_count = (
            select(func.count(ChatMessage.id))
            .where(ChatMessage.chat_id == Chat.id)
            .correlate(Chat)
            .scalar_subquery()
        )
        result = await self.db.execute(
            select(Chat, msg_count.label("messages_count"))
            .where(Chat.user_id == user_id)
            .order_by(Chat.updated_at.desc())
            .offset(offset)
            .limit(limit)
        )
        rows = result.all()
        items = [(self._to_chat_entity(row[0]), int(row[1] or 0)) for row in rows]
        count_result = await self.db.execute(
            select(func.count(Chat.id)).where(Chat.user_id == user_id)
        )
        total = count_result.scalar() or 0
        return items, total

    async def delete_chat(self, chat_id: int, user_id: int) -> bool:
        """Удаляет чат и все сообщения (cascade). Возвращает True если удалён."""
        result = await self.db.execute(
            select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id)
        )
        chat = result.scalar_one_or_none()
        if chat is None:
            return False
        await self.db.delete(chat)
        await self.db.flush()
        return True

    async def get_pending_action(self, chat_id: int) -> Optional[Dict[str, Any]]:
        """Получить отложенное действие чата."""
        result = await self.db.execute(
            select(Chat.pending_action).where(Chat.id == chat_id)
        )
        row = result.scalar_one_or_none()
        return row

    async def set_pending_action(self, chat_id: int, action: Dict[str, Any]) -> None:
        """Сохранить отложенное действие в чате."""
        result = await self.db.execute(
            select(Chat).where(Chat.id == chat_id)
        )
        chat = result.scalar_one_or_none()
        if chat is not None:
            chat.pending_action = action
            self.db.add(chat)
            await self.db.flush()

    async def clear_pending_action(self, chat_id: int) -> None:
        """Очистить отложенное действие чата."""
        result = await self.db.execute(
            select(Chat).where(Chat.id == chat_id)
        )
        chat = result.scalar_one_or_none()
        if chat is not None:
            chat.pending_action = None
            self.db.add(chat)
            await self.db.flush()
