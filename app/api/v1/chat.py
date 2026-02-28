from typing import Optional, List

from fastapi import APIRouter, Depends, UploadFile, File, Query, Body

from app.schemas.user import UserResponse
from app.schemas.base import HistoryMessage, ResponseMessage
from app.core.dependencies import get_current_user, get_chat_service
from app.services.chat_service import ChatService
from app.utils.file import decode_filename
from app.graph.schemas import AskResponse, OverrideIntentType

router = APIRouter()


@router.post("/ask", response_model=ResponseMessage[AskResponse])
async def ask(
    question: str = Query(...),
    user: UserResponse = Depends(get_current_user),
    file: Optional[UploadFile] = File(None),
    file_id: Optional[int] = Query(None, description="ID файла для суммаризации (опционально)"),
    namespace_id: Optional[int] = Query(None),
    override_intent: Optional[OverrideIntentType] = Query(
        None, description="Принудительный интент (пропускает классификацию)"
    ),
    history: Optional[List[HistoryMessage]] = Body(None, embed=True),
    chat_service: ChatService = Depends(get_chat_service),
) -> ResponseMessage[AskResponse]:
    """
    Универсальный эндпоинт для работы с базой знаний.
    
    Интенты (определяются автоматически или через override_intent):
    - rag_query: вопрос по базе знаний
    - save_file: сохранение файла
    - index_url: сохранение URL (YouTube, веб-страницы)
    - summarize: суммаризация контента
    
    Args:
        question: Текст запроса
        user: из JWT (Authorization: Bearer)
        file: Файл для загрузки/суммаризации (опционально)
        file_id: ID файла для суммаризации (опционально, вместе с override_intent=summarize)
        namespace_id: ID пространства знаний (опционально)
        override_intent: Принудительный интент (опционально)
        history: История сообщений для контекста
    """
    file_content = None
    filename = None
    if file is not None:
        file_content = await file.read()
        filename = decode_filename(file.filename or "unnamed_file")
    
    history_dicts = [
        {"role": h.role, "text": h.text, "file_id": h.file_id}
        for h in (history or [])
    ]

    result = await chat_service.ask(
        question=question,
        user_id=user.id,
        namespace_id=namespace_id,
        file_content=file_content,
        filename=filename,
        file_id=file_id,
        history=history_dicts,
        override_intent=override_intent,
    )
    
    return ResponseMessage[AskResponse](data=result)