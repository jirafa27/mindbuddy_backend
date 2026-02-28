"""Сервис суммаризации — подготовка контента и сохранение резюме (дирижёр — узел графа или API)."""
import hashlib
import logging
from typing import Optional, Union

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import SummaryMethod
from app.domain.protocols import SummaryRepository, UserFileRepository, FileRepository
from app.services.content_extractor import ContentExtractorService
from app.services.file_service import FileService
from app.schemas.summary import SummaryResponse, ContentToSummarize, SummaryResult
from app.schemas.file import IngestUrlResult

logger = logging.getLogger(__name__)


class SummaryService:
    """
    Сервис суммаризации: сам собирает контент для суммаризации, используя универсальные
    методы FileService (get_file_text, upload_file, save_extracted_content) и логику дедупа.
    FileService не знает о ContentToSummarize; SummaryService строит его из данных файлов.
    """
    
    def __init__(
        self,
        db: AsyncSession,
        user_file_repository: UserFileRepository,
        file_repository: FileRepository,
        summary_repository: SummaryRepository,
        file_service: FileService,
        content_extractor: ContentExtractorService,
    ):
        self.db = db
        self.user_file_repository = user_file_repository
        self.file_repository = file_repository
        self.summary_repository = summary_repository
        self.file_service = file_service
        self.content_extractor = content_extractor

    async def get_content_for_summarization_url(
        self,
        url: str,
        user_id: int,
    ) -> Union[SummaryResponse, ContentToSummarize]:
        """
        Получить контент для суммаризации по URL.
        Если уже есть кэш — возвращает SummaryResponse. Иначе извлекает контент,
        сохраняет файл и возвращает ContentToSummarize (дирижёр вызовет агента и save_summary).
        """
        dedup_result = await self.file_service.check_deduplication(
            user_id=user_id,
            source_url=url,
            file_repository=self.user_file_repository,
        )
        if dedup_result.is_duplicate and dedup_result.existing_file_id:
            existing_uf = await self.user_file_repository.get_by_id(dedup_result.existing_file_id)
            if existing_uf:
                existing_summary = await self.summary_repository.get_by_file_id(existing_uf.file_id)
                if existing_summary:
                    logger.info("[Summary] Cached summary for URL: %s", url)
                    title = existing_uf.custom_title or (existing_uf.file.media_metadata or {}).get("title") if existing_uf.file else "Document"
                    return SummaryResponse(
                        file_id=existing_uf.id,
                        summary=existing_summary.content,
                        title=title,
                        source_url=url,
                        is_cached=True,
                        method=SummaryMethod.CACHED,
                    )
        logger.info("[Summary] Extracting content from URL: %s", url)
        parsed = await self.content_extractor.extract(url)
        dedup_result = await self.file_service.check_deduplication(
            user_id=user_id,
            content_hash=parsed.content_hash,
            file_repository=self.user_file_repository,
        )
        if dedup_result.is_duplicate and dedup_result.existing_file_id:
            existing_uf = await self.user_file_repository.get_by_id(dedup_result.existing_file_id)
            if existing_uf:
                existing_summary = await self.summary_repository.get_by_file_id(existing_uf.file_id)
                if existing_summary:
                    logger.info("[Summary] Cached summary for hash: %s", parsed.content_hash[:16])
                    title = existing_uf.custom_title or (existing_uf.file.media_metadata or {}).get("title") if existing_uf.file else "Document"
                    return SummaryResponse(
                        file_id=existing_uf.id,
                        summary=existing_summary.content,
                        title=title,
                        source_url=url,
                        is_cached=True,
                        method=SummaryMethod.CACHED,
                    )
        user_file = await self.file_service.save_extracted_content(
            user_id=user_id,
            text=parsed.text,
            title=parsed.title,
            source_url=url,
            content_hash=parsed.content_hash,
            content_type=parsed.content_type,
        )
        return ContentToSummarize(
            text=parsed.text,
            title=parsed.title,
            source_url=url,
            content_file_id=user_file.file_id,
            user_file_id=user_file.id,
        )
    
    async def get_content_for_summarization_file(
        self,
        file_content: bytes,
        filename: str,
        user_id: int,
    ) -> Union[SummaryResponse, ContentToSummarize]:
        """
        Получить контент для суммаризации из загруженного файла.
        Кэш по хэшу контента; иначе сохраняет файл и возвращает ContentToSummarize.
        """
        file_ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if not file_ext:
            raise ValueError("Не удалось определить тип файла")
        text = self.file_service.extract_text(file_content, file_ext)
        if not text or not text.strip():
            raise ValueError("Не удалось извлечь текст из файла")
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        dedup_result = await self.file_service.check_deduplication(
            user_id=user_id,
            content_hash=content_hash,
            file_repository=self.user_file_repository,
        )
        if dedup_result.is_duplicate and dedup_result.existing_file_id:
            existing_uf = await self.user_file_repository.get_by_id(dedup_result.existing_file_id)
            if existing_uf:
                existing_summary = await self.summary_repository.get_by_file_id(existing_uf.file_id)
                if existing_summary:
                    logger.info("[Summary] Cached summary for file hash: %s", content_hash[:16])
                    title = existing_uf.custom_title or (existing_uf.file.media_metadata or {}).get("title") if existing_uf.file else "Document"
                    return SummaryResponse(
                        file_id=existing_uf.id,
                        summary=existing_summary.content,
                        title=title,
                        source_url=None,
                        is_cached=True,
                        method=SummaryMethod.CACHED,
                    )
        logger.info("[Summary] Uploading file: %s (user=%d)", filename, user_id)
        file_created = await self.file_service.upload_file(
            user_id=user_id,
            file_content=file_content,
            filename=filename,
            db=self.db,
            content_hash=content_hash,
        )
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        return ContentToSummarize(
            text=text,
            title=title,
            source_url=None,
            content_file_id=file_created.content_file_id,
            user_file_id=file_created.file_id,
        )

    async def get_summary_by_file_id(
        self,
        file_id: int,
        user_id: int,
    ) -> Optional[SummaryResponse]:
        """Получить суммаризацию по ID файла (user_file_id)."""
        uf = await self.user_file_repository.get_by_id(file_id)
        if not uf or uf.user_id != user_id:
            return None
        summary = await self.summary_repository.get_by_file_id(uf.file_id)
        if not summary:
            return None
        title = uf.custom_title or (uf.file.media_metadata or {}).get("title") if uf.file else "Document"
        source_url = uf.file.source_url if uf.file else None
        return SummaryResponse(
            file_id=uf.id,
            summary=summary.content,
            title=title,
            source_url=source_url,
            is_cached=True,
            method=SummaryMethod.CACHED,
        )

    async def get_content_for_summarization_existing_file(
        self,
        file_id: int,
        user_id: int,
    ) -> Union[SummaryResponse, ContentToSummarize]:
        """
        Получить контент для суммаризации по file_id из истории.
        Если суммаризация уже есть — возвращает SummaryResponse. Иначе загружает файл и возвращает ContentToSummarize.
        """
        uf = await self.user_file_repository.get_by_id(file_id)
        if not uf:
            raise ValueError(f"Файл с ID {file_id} не найден")
        if uf.user_id != user_id:
            raise ValueError("Файл не принадлежит пользователю")
        cf = uf.file
        if not cf:
            raise ValueError("Контент файла не найден")
        existing_summary = await self.summary_repository.get_by_file_id(cf.id)
        if existing_summary:
            logger.info("[Summary] Cached summary for file_id=%d", file_id)
            title = uf.custom_title or (cf.media_metadata or {}).get("title") or "Document"
            return SummaryResponse(
                file_id=uf.id,
                summary=existing_summary.content,
                title=title,
                source_url=cf.source_url,
                is_cached=True,
                method=SummaryMethod.CACHED,
            )
        file_path = cf.file_path
        if not file_path:
            raise ValueError("Путь к файлу не найден")
        logger.info("[Summary] Loading file content from storage: %s", file_path)
        file_content = await self.file_service.storage.download_file(file_path)
        if not file_content:
            raise ValueError("Не удалось загрузить файл из хранилища")
        filename = uf.custom_title or (cf.media_metadata or {}).get("title") or "document"
        file_ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "md"
        text = self.file_service.extract_text(file_content, file_ext)
        if not text or not text.strip():
            raise ValueError("Не удалось извлечь текст из файла")
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        return ContentToSummarize(
            text=text,
            title=title,
            source_url=cf.source_url,
            content_file_id=cf.id,
            user_file_id=uf.id,
        )

    async def save_summary(self, content_file_id: int, summary_result: SummaryResult) -> None:
        """Сохранить результат суммаризации в БД (вызывает дирижёр после агента)."""
        await self.summary_repository.create(
            file_id=content_file_id,
            text=summary_result.content,
            model_name=summary_result.model_name,
        )
        await self.db.commit()
        logger.info("[Summary] Saved summary for file_id=%d, method=%s", content_file_id, summary_result.method)



    async def ingest_url(
        self,
        url: str,
        user_id: int,
        namespace_id: Optional[int] = None,
    ) -> IngestUrlResult:
        """
        Индексирует URL без суммаризации.
        
        Парсит контент, сохраняет в MinIO и БД, возвращает данные
        для последующей векторизации через Celery.
        
        Args:
            url: URL источника (YouTube, веб-страница)
            user_id: ID пользователя
            namespace_id: ID пространства знаний (опционально)
            
        Returns:
            IngestUrlResult с file_id и текстом для эмбеддингов
        """
        # 1. Проверка дедупликации по URL
        dedup_result = await self.file_service.check_deduplication(
            user_id=user_id,
            source_url=url,
        )
        
        if dedup_result.is_duplicate and dedup_result.existing_file_id:
            existing_uf = await self.user_file_repository.get_by_id(dedup_result.existing_file_id)
            if existing_uf and existing_uf.file:
                logger.info("[Ingest] URL already indexed: %s (user_file_id=%d)", url, existing_uf.id)
                file_path = existing_uf.file.file_path
                text = ""
                if file_path:
                    file_content = await self.file_service.storage.download_file(file_path)
                    text = file_content.decode("utf-8") if file_content else ""
                filename = existing_uf.custom_title or (existing_uf.file.media_metadata or {}).get("title") or "document"
                return IngestUrlResult(
                    file_id=existing_uf.id,
                    content_file_id=existing_uf.file_id,
                    filename=filename,
                    text=text,
                    is_duplicate=True,
                )
        
        # 2. Извлечение контента
        logger.info("[Ingest] Extracting content from URL: %s", url)
        parsed = await self.content_extractor.extract(url)
        
        # 3. Проверка дедупликации по хэшу контента
        dedup_result = await self.file_service.check_deduplication(
            user_id=user_id,
            content_hash=parsed.content_hash,
            file_repository=self.user_file_repository,
        )
        
        if dedup_result.is_duplicate and dedup_result.existing_file_id:
            existing_uf = await self.user_file_repository.get_by_id(dedup_result.existing_file_id)
            if existing_uf:
                logger.info("[Ingest] Content hash match: %s (user_file_id=%d)", parsed.content_hash[:16], existing_uf.id)
                filename = existing_uf.custom_title or (existing_uf.file.media_metadata or {}).get("title") if existing_uf.file else parsed.title
                return IngestUrlResult(
                    file_id=existing_uf.id,
                    content_file_id=existing_uf.file_id,
                    filename=filename or parsed.title,
                    text=parsed.text,
                    is_duplicate=True,
                )
        
        user_file = await self.file_service.save_extracted_content(
            user_id=user_id,
            text=parsed.text,
            title=parsed.title,
            source_url=url,
            content_hash=parsed.content_hash,
            content_type=parsed.content_type,
            namespace_id=namespace_id,
        )
        logger.info("[Ingest] Saved URL content: user_file_id=%d, title=%s", user_file.id, parsed.title)
        message = None
        if getattr(parsed, "fallback_used", False):
            message = "Не удалось получить текст видео, сохранены только метаданные"
        return IngestUrlResult(
            file_id=user_file.id,
            content_file_id=user_file.file_id,
            filename=user_file.custom_title or parsed.title,
            text=parsed.text,
            is_duplicate=False,
            message=message,
        )
