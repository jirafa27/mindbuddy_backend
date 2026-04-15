from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.domain.entities import (
    SyncCommandEntity,
)
from app.infrastructure.db.models import Namespace, SyncCommand, UserFile
from app.schemas.file import CommandStatus


class PgSyncRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    @staticmethod
    def _cmd_to_entity(model: SyncCommand) -> SyncCommandEntity:
        return SyncCommandEntity(
            id=model.id,
            user_id=model.user_id,
            user_file_id=model.user_file_id,
            command_type=model.command_type,
            payload_json=model.payload_json or {},
            status=model.status,
            created_at=model.created_at,
            acked_at=model.acked_at,
        )


    async def create_command(
        self,
        *,
        user_id: int,
        user_file_id: Optional[int],
        command_type: str,
        payload_json: dict,
        status: str = "pending",
    ) -> SyncCommandEntity:
        cmd = SyncCommand(
            user_id=user_id,
            user_file_id=user_file_id,
            command_type=command_type,
            payload_json=payload_json,
            status=status,
        )
        self.db.add(cmd)
        await self.db.flush()
        return self._cmd_to_entity(cmd)

    async def get_pending_commands(
        self, user_id: int, limit: int = 100
    ) -> list[SyncCommandEntity]:
        result = await self.db.execute(
            select(SyncCommand)
            .where(
                SyncCommand.user_id == user_id,
                SyncCommand.status == CommandStatus.PENDING.value,
            )
            .order_by(SyncCommand.created_at.asc())
            .limit(limit)
        )
        return [self._cmd_to_entity(r) for r in result.scalars().all()]

    async def get_command(
        self, command_id: int, user_id: int
    ) -> Optional[SyncCommandEntity]:
        result = await self.db.execute(
            select(SyncCommand).where(
                SyncCommand.id == command_id,
                SyncCommand.user_id == user_id,
            )
        )
        row = result.scalar_one_or_none()
        return self._cmd_to_entity(row) if row else None

    async def ack_command(
        self, command_id: int, user_id: int, status: str
    ) -> Optional[SyncCommandEntity]:
        from datetime import datetime

        result = await self.db.execute(
            select(SyncCommand).where(
                SyncCommand.id == command_id,
                SyncCommand.user_id == user_id,
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        row.status = status
        row.acked_at = datetime.utcnow()
        await self.db.flush()
        return self._cmd_to_entity(row)


