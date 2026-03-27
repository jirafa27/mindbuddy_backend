from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Query, Form
from fastapi.responses import StreamingResponse
import io
import logging


from app.utils.file import decode_filename, encode_filename_for_header
from app.schemas.base import ResponseMessage
from app.schemas.file import (
    FileResponse,
    FileInfo,
)
from app.schemas.content import AttachFileRequest, AttachFileResponse
from app.services.file_service import FileService
from app.services.summary_service import SummaryService
from app.services.namespace_service import NamespaceService
from app.core.dependencies import (
    get_file_service,
    get_current_user,
    get_summary_service,
    get_namespace_service,
    get_task_publisher,
)
from app.domain.protocols import TaskPublisher
from app.schemas.user import UserResponse
from app.core.exceptions import ValidationError, NotFoundError, ForbiddenError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload", response_model=ResponseMessage[list[FileResponse]], status_code=status.HTTP_201_CREATED)
async def upload_file(
    files: list[UploadFile] = File(default=[]),
    url: Optional[str] = Form(None, description="URL для парсинга (YouTube, веб-страница)"),
    text: Optional[str] = Form(None, description="Текст для сохранения в пространство (Markdown)"),
    title: Optional[str] = Form(None, description="Заголовок MD-файла (если передан text)"),
    namespace_id: Optional[int] = Query(None, description="ID пространства (опционально; без значения — файл в Inbox пользователя)"),
    user: UserResponse = Depends(get_current_user),
    file_service: FileService = Depends(get_file_service),
    summary_service: SummaryService = Depends(get_summary_service),
    namespace_service: NamespaceService = Depends(get_namespace_service),
    task_publisher: TaskPublisher = Depends(get_task_publisher),
):
    """
    Универсальный эндпоинт загрузки контента (файл/файлы, URL или текст).
    Аутентификация: JWT в заголовке Authorization.
    """
    user_id = user.id
    all_files = files
    has_files = len(all_files) > 0
    has_url = bool(url and str(url).strip())
    has_text = bool(text is not None and str(text).strip())
    sources = has_files + has_url + has_text
    if sources == 0:
        raise ValidationError("Необходимо передать file, url или text")
    if sources > 1:
        raise ValidationError("Передайте только один из параметров: file, url или text")

    if namespace_id is None:
        namespace_id = await namespace_service.get_or_create_inbox(user_id)

    if url:
        logger.info("[Upload] Processing URL: %s (user=%d, namespace_id=%s)", url, user_id, namespace_id)
        ingest_result = await summary_service.ingest_url(
            url=url,
            user_id=user_id,
            namespace_id=namespace_id,
        )
        task_id = task_publisher.send_embeddings_task(
            content_file_id=ingest_result.content_file_id,
            text=ingest_result.text,
            namespace_id=namespace_id,
            filename=ingest_result.filename,
            user_file_id=ingest_result.file_id,
        ) or ""
        return ResponseMessage[list[FileResponse]](data=[FileResponse(
            file_id=ingest_result.file_id,
            filename=ingest_result.filename,
            task_id=task_id,
            status="duplicate" if ingest_result.is_duplicate else "processing",
            message=ingest_result.message,
        )])

    # Обработка текста (сохранение как MD)
    if text is not None:
        file_created = await file_service.create_file_from_text(
            text=text,
            user_id=user_id,
            namespace_id=namespace_id,
            title=title,
        )
        task_id = task_publisher.send_embeddings_task(
            content_file_id=file_created.content_file_id,
            text=file_created.text,
            namespace_id=namespace_id,
            filename=file_created.filename,
            user_file_id=file_created.file_id,
        ) or ""
        return ResponseMessage[list[FileResponse]](data=[FileResponse(
            file_id=file_created.file_id,
            filename=file_created.filename,
            task_id=task_id,
            status="processing",
            message="Текст сохранён в пространство как Markdown",
        )])

    # Обработка файлов
    results = []
    for upload in all_files:
        file_content = await upload.read()
        filename = decode_filename(upload.filename or "unnamed_file")
        file_created = await file_service.upload_file(
            file_content=file_content,
            filename=filename,
            namespace_id=namespace_id,
            user_id=user_id,
        )
        task_id = task_publisher.send_embeddings_task(
            content_file_id=file_created.content_file_id,
            text=file_created.text,
            namespace_id=namespace_id,
            filename=file_created.filename,
            user_file_id=file_created.file_id,
        ) or ""
        results.append(FileResponse(
            file_id=file_created.file_id,
            filename=file_created.filename,
            task_id=task_id,
            status="processing",
        ))

    return ResponseMessage[list[FileResponse]](data=results)



@router.get("/{file_id}", response_model=ResponseMessage[FileInfo])
async def get_file_info(
    file_id: int,
    user: UserResponse = Depends(get_current_user),
    file_service: FileService = Depends(get_file_service),
) -> ResponseMessage[FileInfo]:
    """Метаданные файла по ID (user_file_id)."""
    file_info = await file_service.get_file_info(file_id=file_id, user_id=user.id)
    return ResponseMessage[FileInfo](data=file_info)


@router.head("/download/{file_id}")
async def head_download_file(
    file_id: int,
    user: UserResponse = Depends(get_current_user),
    file_service: FileService = Depends(get_file_service),
):
    """
    HEAD-запрос для скачивания файла — возвращает метаданные без тела.
    """
    from fastapi.responses import Response

    file_content, filename, content_type = await file_service.download_file(
        file_id=file_id,
        user_id=user.id,
    )

    filename = decode_filename(filename)

    return Response(
        media_type=content_type,
        headers={
            "Content-Disposition": encode_filename_for_header(filename),
            "Content-Length": str(len(file_content)),
        },
    )


@router.get("/download/{file_id}")
async def download_file(
    file_id: int,
    user: UserResponse = Depends(get_current_user),
    file_service: FileService = Depends(get_file_service),
):
    """
    Скачивает файл из хранилища.
    
    Аутентификация: По Telegram ID
    
    Args:
        file_id: ID файла
        user: UserResponse с информацией о пользователе
        file_service: Сервис для работы с файлами
    
    Returns:
        Файл для скачивания (бинарный поток)
        
    Raises:
        404: Если пользователь или файл не найден
        403: Если нет доступа к файлу
    """
    file_content, filename, content_type = await file_service.download_file(
        file_id=file_id,
        user_id=user.id,
    )
    
    filename = decode_filename(filename)
    
    return StreamingResponse(
        io.BytesIO(file_content),
        media_type=content_type,
        headers={
            "Content-Disposition": encode_filename_for_header(filename),
            "Content-Length": str(len(file_content)),
        }
    )


@router.put("/{file_id}/content", response_model=ResponseMessage[FileInfo])
async def replace_file_content(
    file_id: int,
    file: UploadFile = File(..., description="Новое содержимое файла"),
    user: UserResponse = Depends(get_current_user),
    file_service: FileService = Depends(get_file_service),
) -> ResponseMessage[FileInfo]:
    """
    Заменяет содержимое существующего файла in-place, сохраняя file_id.

    Проверяет владельца, перезаписывает файл в хранилище, обновляет метаданные,
    инвалидирует суммаризацию и перезапускает конвейер индексации (эмбеддинги).
    """
    file_content = await file.read()
    filename = decode_filename(file.filename or "unnamed_file")
    if not filename or not file_content:
        raise ValidationError("Файл не передан или пуст")

    file_info = await file_service.replace_file_content(
        file_id=file_id,
        user_id=user.id,
        file_content=file_content,
        filename=filename,
    )
    return ResponseMessage[FileInfo](status="success", data=file_info)


@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    file_id: int,
    user: UserResponse = Depends(get_current_user),
    file_service: FileService = Depends(get_file_service),
):
    """
    Удаляет файл из хранилища и базы данных.
    
    Аутентификация: JWT.
    
    Args:
        file_id: ID файла
        user: UserResponse с информацией о пользователе
        file_service: Сервис для работы с файлами
        
    Raises:
        404: Если пользователь или файл не найден
        403: Если нет доступа к файлу
    """
    await file_service.delete_file(
        file_id=file_id,
        user_id=user.id,
    )


@router.patch("/{file_id}/move", response_model=ResponseMessage[FileInfo])
async def move_file_to_namespace(
    file_id: int,
    namespace_id: int = Query(..., description="ID пространства назначения"),
    user: UserResponse = Depends(get_current_user),
    file_service: FileService = Depends(get_file_service),
) -> ResponseMessage[FileInfo]:
    """
    Перемещает файл в указанное пространство.
    
    Аутентификация: JWT.
    
    Args:
        file_id: ID файла
        namespace_id: ID пространства назначения
        user: UserResponse с информацией о пользователе
        file_service: Сервис для работы с файлами
    
    Returns:
        ResponseMessage[FileInfo] с информацией о файле
        
    Raises:
        404: Если пользователь, файл или пространство не найдены
        403: Если нет доступа к файлу или пространству
    """
    file_info = await file_service.move_to_namespace(
        file_id=file_id,
        namespace_id=namespace_id,
        user_id=user.id,
    )
    return ResponseMessage[FileInfo](
        message=f"Файл {file_info.user_file_id} перемещен в пространство {namespace_id}",
        data=FileInfo(
                user_file_id=file_info.user_file_id,
                content_file_id=file_info.content_file_id,
                user_id=file_info.user_id,
                namespace_id=file_info.namespace_id,
                filename=file_info.filename,
                file_type=file_info.file_type,
                file_size=file_info.file_size,
                created_at=file_info.created_at,
                updated_at=file_info.updated_at,
                file_path=file_info.file_path,
    ))


@router.post(
    "/{file_id}/attach",
    response_model=ResponseMessage[AttachFileResponse],
    status_code=status.HTTP_201_CREATED,
)
async def attach_file_to_namespace(
    file_id: int,
    body: AttachFileRequest,
    user: UserResponse = Depends(get_current_user),
    file_service: FileService = Depends(get_file_service),
) -> ResponseMessage[AttachFileResponse]:
    """
    Привязывает контент-файл к пространству пользователя, создавая запись в user_files.

    `file_id` — это ID контент-файла (files.id), возвращённый из `POST /content/extract`.
    После успешной привязки файл появляется в пространстве пользователя.

    Если файл уже привязан к другому пространству — namespace_id обновляется.
    """
    try:
        user_file = await file_service.attach_file_to_namespace(
            content_file_id=file_id,
            user_id=user.id,
            namespace_id=body.namespace_id,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ForbiddenError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.exception("[API/Attach] Failed to attach file_id=%d for user=%d", file_id, user.id)
        raise HTTPException(status_code=500, detail="Ошибка привязки файла к пространству")

    content_file = await file_service.file_repository.get_by_id(file_id)
    filename = user_file.custom_title or (content_file.media_metadata or {}).get("title") if content_file else None

    return ResponseMessage[AttachFileResponse](
        data=AttachFileResponse(
            user_file_id=user_file.id,
            content_file_id=file_id,
            namespace_id=body.namespace_id,
            filename=filename,
        )
    )