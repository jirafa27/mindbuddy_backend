from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.entities import FileEntity
from app.infrastructure.db.models import File


class PgFileRepository:
    def __init__(self, db: AsyncSession):
        self.db = db


    def _to_entity(self, model: File) -> FileEntity:
        return FileEntity(
            id=model.id,
            content_hash=model.content_hash,
            source_url=model.source_url,
            transcript_text=model.transcript_text,
            file_path=model.file_path,
            media_metadata=model.media_metadata,
            processing_status=model.processing_status,
            created_at=model.created_at,
        )

    async def get_by_id(self, file_id: int) -> Optional[FileEntity]:
        result = await self.db.execute(
            select(File).where(File.id == file_id)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_entity(row)

    async def get_by_content_hash(self, content_hash: str) -> Optional[FileEntity]:
        result = await self.db.execute(
            select(File).where(File.content_hash == content_hash)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_entity(row)

    async def get_by_source_url(self, source_url: str) -> Optional[FileEntity]:
        result = await self.db.execute(
            select(File).where(File.source_url == source_url).limit(1)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_entity(row)

    async def create(
        self,
        content_hash: str,
        source_url: Optional[str] = None,
        transcript_text: Optional[str] = None,
        file_path: Optional[str] = None,
        media_metadata: Optional[dict] = None,
        processing_status: str = "pending",
    ) -> FileEntity:
        content_file = File(
            content_hash=content_hash,
            source_url=source_url,
            transcript_text=transcript_text or "",
            file_path=file_path,
            media_metadata=media_metadata,
            processing_status=processing_status or "pending",
        )
        self.db.add(content_file)
        await self.db.flush()
        return self._to_entity(content_file)

    async def update_content_metadata(
        self,
        file_id: int,
        *,
        content_hash: str,
        media_metadata: Optional[dict] = None,
        transcript_text: Optional[str] = None,
    ) -> Optional[FileEntity]:
        """Обновляет content_hash и media_metadata файла."""
        meta = media_metadata or {}
        values = {
            "content_hash": content_hash,
            "media_metadata": meta,
        }
        if transcript_text is not None:
            values["transcript_text"] = transcript_text
        await self.db.execute(
            update(File)
            .where(File.id == file_id)
            .values(**values)
        )
        await self.db.flush()
        return await self.get_by_id(file_id)