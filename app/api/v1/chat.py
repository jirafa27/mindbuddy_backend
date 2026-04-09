import json
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, UploadFile, File, Query, Form, status

from app.schemas.user import UserResponse
from app.schemas.base import HistoryMessage, ResponseMessage, ListResponseData, PaginationInfo
from app.schemas.chat import ChatListItem, ChatMessageItem, ChatUpdate
from app.schemas.file import RawFileUpload
from app.core.dependencies import get_current_user, get_chat_service
from app.core.exceptions import NotFoundError
from app.services.chat_service import ChatService
from app.utils.file import decode_filename
from app.graph.schemas import AskResponse, OverrideIntentType

router = APIRouter()


@router.post("/ask", response_model=ResponseMessage[AskResponse])
async def ask(
    question: str = Form(...),
    user: UserResponse = Depends(get_current_user),
    files: List[UploadFile] = File(default=[]),
    file_ids: Optional[List[int]] = Query(None, description="Список ID файлов (user_files.id)"),
    namespace_id: Optional[int] = Query(None),
    override_intent: Optional[OverrideIntentType] = Query(
        None, description="Принудительный интент (пропускает классификацию)"
    ),
    history: Optional[str] = Form(None),
    chat_id: Optional[int] = Query(None, description="ID чата для продолжения диалога (если не передан — создаётся новый)"),
    chat_name: Optional[str] = Query(None, description="Название чата (при создании нового)"),
    chat_service: ChatService = Depends(get_chat_service),
) -> ResponseMessage[AskResponse]:
    files_data: list[RawFileUpload] = []
    for upload in files:
        content = await upload.read()
        name = decode_filename(upload.filename or "unnamed_file")
        files_data.append(RawFileUpload(
            content=content,
            filename=name,
            content_type=upload.content_type,
            size=len(content),
        ))

    parsed_history: List[HistoryMessage] = []
    if history:
        raw = json.loads(history)
        if isinstance(raw, list):
            parsed_history = [HistoryMessage(**item) for item in raw]
        elif isinstance(raw, dict):
            parsed_history = [HistoryMessage(**raw)]

    history_dicts = [
        {"role": h.role, "text": h.text, "file_ids": h.file_ids}
        for h in parsed_history
    ]

    result = await chat_service.ask(
        question=question,
        user_id=user.id,
        namespace_id=namespace_id,
        files=files_data or None,
        file_ids=list(file_ids) if file_ids else None,
        history=history_dicts,
        override_intent=override_intent,
        chat_id=chat_id,
        chat_name=chat_name,
    )

    return ResponseMessage[AskResponse](data=result)


@router.get(
    "/chats",
    response_model=ResponseMessage[ListResponseData[ChatListItem]],
    summary="Список чатов пользователя",
)
async def list_chats(
    page: int = Query(1, ge=1, description="Номер страницы"),
    page_size: int = Query(20, ge=1, le=100, description="Размер страницы"),
    user: UserResponse = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> ResponseMessage[ListResponseData[ChatListItem]]:
    """Список чатов текущего пользователя с пагинацией."""
    skip = (page - 1) * page_size
    items_with_count, total = await chat_service.get_user_chats(
        user_id=user.id, limit=page_size, offset=skip
    )
    list_items = [
        ChatListItem(
            id=chat.id,
            user_id=chat.user_id,
            name=chat.name,
            created_at=chat.created_at or datetime.utcnow(),
            updated_at=chat.updated_at or datetime.utcnow(),
            messages_count=count,
        )
        for chat, count in items_with_count
    ]
    pagination = PaginationInfo(total=total, page=page, page_size=page_size)
    return ResponseMessage(data=ListResponseData(items=list_items, pagination=pagination))


@router.delete(
    "/chats/{chat_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Удалить чат",
)
async def delete_chat(
    chat_id: int,
    user: UserResponse = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> None:
    """Удалить чат и все сообщения. Доступ только к своим чатам."""
    deleted = await chat_service.delete_chat(chat_id=chat_id, user_id=user.id)
    if not deleted:
        raise NotFoundError("Чат не найден или доступ запрещён")


@router.patch(
    "/chats/{chat_id}",
    response_model=ResponseMessage[ChatListItem],
    summary="Обновить чат (название)",
)
async def update_chat(
    chat_id: int,
    body: ChatUpdate,
    user: UserResponse = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> ResponseMessage[ChatListItem]:
    """Обновить название чата. Доступ только к своим чатам."""
    updated = await chat_service.update_chat_name(
        chat_id=chat_id, user_id=user.id, name=body.name
    )
    if not updated:
        raise NotFoundError("Чат не найден или доступ запрещён")
    return ResponseMessage(
        data=ChatListItem(
            id=updated.id,
            user_id=updated.user_id,
            name=updated.name,
            created_at=updated.created_at or datetime.utcnow(),
            updated_at=updated.updated_at or datetime.utcnow(),
            messages_count=0,
        )
    )


@router.get(
    "/chats/{chat_id}/messages",
    response_model=ResponseMessage[ListResponseData[ChatMessageItem]],
    summary="История сообщений чата",
)
async def get_chat_messages(
    chat_id: int,
    page: int = Query(1, ge=1, description="Номер страницы"),
    page_size: int = Query(50, ge=1, le=100, description="Размер страницы"),
    user: UserResponse = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> ResponseMessage[ListResponseData[ChatMessageItem]]:
    """Сообщения чата с пагинацией. Доступ только к своим чатам."""
    messages, total = await chat_service.get_chat_messages(
        chat_id=chat_id, user_id=user.id, limit=page_size, offset=(page - 1) * page_size
    )
    items = [
        ChatMessageItem(
            id=m.id,
            chat_id=m.chat_id,
            role=m.role.value,
            text=m.text,
            file_ids=m.file_ids,
            namespace_id=m.namespace_id,
            created_at=m.created_at or datetime.utcnow(),
        )
        for m in messages
    ]
    pagination = PaginationInfo(total=total, page=page, page_size=page_size)
    return ResponseMessage(data=ListResponseData(items=items, pagination=pagination))