from typing import Optional

from fastapi import APIRouter, Depends, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.user import UserResponse
from app.core.dependencies import (
    get_user_by_telegram_id,
    get_chat_service,
    get_file_repository,
)
from app.domain.protocols import AsyncFileRepository
from app.services.chat_service import ChatService
from app.infrastructure.db.session import get_db
from app.utils.file import decode_filename
from app.schemas.base import ResponseMessage

router = APIRouter()


@router.post("/ask")
async def ask(
    question: str = Query(...),
	user: UserResponse = Depends(get_user_by_telegram_id),
    file: Optional[UploadFile] = File(None),
    namespace_id: Optional[int] = Query(None),
    chat_service: ChatService = Depends(get_chat_service),
    file_repository: AsyncFileRepository = Depends(get_file_repository),
    db: AsyncSession = Depends(get_db),
):
    file_content = None
    filename = None
    if file is not None:
        file_content = await file.read()
        filename = decode_filename(file.filename or "unnamed_file")

    answer = await chat_service.ask(
        question=question,
        user_id=user.id,
        namespace_id=namespace_id,
        file_content=file_content,
        filename=filename,
        async_db=db,
        file_repository=file_repository,
    )
    return ResponseMessage(data=answer)