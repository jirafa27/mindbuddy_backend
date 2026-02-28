from typing import Any, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.db.models import Summary
from app.domain.entities import SummaryEntity


class PgSummaryRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    def _to_entity(self, model: Summary) -> SummaryEntity:
        return SummaryEntity(
            id=model.id,
            file_id=model.file_id,
            lookup_key=model.lookup_key,
            text=model.text,
            used_prompt=model.used_prompt,
            model_name=model.model_name,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )


    async def get_by_file_id(self, file_id: int) -> Optional[SummaryEntity]:
        result = await self.db.execute(
            select(Summary).where(Summary.file_id == file_id).limit(1)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_entity(row)

    async def get_by_file_and_lookup_key(
        self, file_id: int, lookup_key: str
    ) -> Optional[SummaryEntity]:
        result = await self.db.execute(
            select(Summary).where(
                Summary.file_id == file_id,
                Summary.lookup_key == lookup_key,
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return self._to_entity(row)

    async def create(
        self,
        file_id: int,
        text: str,
        lookup_key: str = "standard_v1",
        used_prompt: Optional[str] = None,
        model_name: Optional[str] = "yandexgpt",
        **kwargs: Any,
    ) -> SummaryEntity:
        if "content" in kwargs:
            text = kwargs["content"]
        if "configuration_hash" in kwargs:
            lookup_key = kwargs["configuration_hash"]
        summary = Summary(
            file_id=file_id,
            lookup_key=lookup_key,
            text=text,
            used_prompt=used_prompt,
            model_name=model_name,
        )
        self.db.add(summary)
        await self.db.flush()
        return self._to_entity(summary)
