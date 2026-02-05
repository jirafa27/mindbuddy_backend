from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.infrastructure.db.models import File as FileModel


class FileRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, file_id: int) -> Optional[FileModel]:
        return self.db.query(FileModel).filter(FileModel.id == file_id).first()

    def delete(self, file_id: int) -> bool:
        db_file = self.get_by_id(file_id)
        if db_file:
            self.db.delete(db_file)
            return True
        return False


class AsyncFileRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, file_id: int) -> Optional[FileModel]:
        result = await self.db.execute(
            select(FileModel).where(FileModel.id == file_id)
        )
        return result.scalar_one_or_none()

    async def create(
        self,
        user_id: int,
        filename: str,
        file_path: str,
        file_type: str,
        file_size: int,
        namespace_id: Optional[int] = None,
    ) -> FileModel:
        db_file = FileModel(
            user_id=user_id,
            namespace_id=namespace_id,
            filename=filename,
            file_path=file_path,
            file_type=file_type,
            file_size=file_size,
        )
        self.db.add(db_file)
        await self.db.flush()
        return db_file

    async def delete(self, file: FileModel) -> None:
        await self.db.delete(file)
