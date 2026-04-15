"""Сервис суммаризации — подготовка контента и сохранение резюме (дирижёр — узел графа или API)."""
import hashlib
import logging
from typing import Optional, Union

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import SummaryMethod
from app.domain.protocols import SummaryRepository, UserFileRepository, FileRepository
from app.services.content_extractor import ContentExtractorService
from app.services.file_content_service import FileContentService
from app.services.file_service import FileService
from app.schemas.summary import SummaryResponse, ContentToSummarize, SummaryResult
from app.schemas.file import IngestUrlResult
from app.schemas.content import ContentExtractResponse

logger = logging.getLogger(__name__)


class SummaryService:
    """
    Сервис суммаризации: собирает контент для суммаризации, используя
    FileService для файловых операций высокого уровня и FileContentService
    для чтения/извлечения текста из content-файлов.
    """
    
    def __init__(
        self,
        db: AsyncSession,
        user_file_repository: UserFileRepository,
        file_repository: FileRepository,
        summary_repository: SummaryRepository,
        file_service: FileService,
        file_content_service: FileContentService,
        content_extractor: ContentExtractorService,
    ):
        self.db = db
        self.user_file_repository = user_file_repository
        self.file_repository = file_repository
        self.summary_repository = summary_repository
        self.file_service = file_service
        self.file_content_service = file_content_service
        self.content_extractor = content_extractor

    async def get_content_for_summarization_url(
        self,
        url: str,
        user_id: int,
        namespace_id: Optional[int] = None,
    ) -> Union[SummaryResponse, ContentToSummarize]:
        """
        Получить контент для суммаризации по URL.
        Если уже есть кэш — возвращает SummaryResponse. Иначе извлекает контент,
        сохраняет файл и возвращает ContentToSummarize (дирижёр вызовет агента и save_summary).
        """
        dedup_result = await self.file_service.check_deduplication(
            user_id=user_id,
            source_url=url,
        )
        if dedup_result.is_duplicate and dedup_result.existing_file_id:
            existing_uf = await self.user_file_repository.get_by_id(dedup_result.existing_file_id)
            if existing_uf:
                existing_summary = await self.summary_repository.get_by_file_id(existing_uf.file_id)
                if existing_summary:
                    logger.info("[Summary] Cached summary for URL: %s", url)
                    content_file = await self.file_repository.get_by_id(existing_uf.file_id)
                    title = existing_uf.custom_title or (content_file.media_metadata or {}).get("title") if content_file else "Document"
                    return SummaryResponse(
                        user_file_id=existing_uf.id,
                        content_file_id=existing_uf.file_id,
                        summary=existing_summary.text,
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
        )
        if dedup_result.is_duplicate and dedup_result.existing_file_id:
            existing_uf = await self.user_file_repository.get_by_id(dedup_result.existing_file_id)
            if existing_uf:
                existing_summary = await self.summary_repository.get_by_file_id(existing_uf.file_id)
                if existing_summary:
                    logger.info("[Summary] Cached summary for hash: %s", parsed.content_hash[:16])
                    content_file = await self.file_repository.get_by_id(existing_uf.file_id)
                    title = existing_uf.custom_title or (content_file.media_metadata or {}).get("title") if content_file else "Document"
                    return SummaryResponse(
                        user_file_id=existing_uf.id,
                        content_file_id=existing_uf.file_id,
                        summary=existing_summary.text,
                        title=title,
                        source_url=url,
                        is_cached=True,
                        method=SummaryMethod.CACHED,
                    )
                content_file = await self.file_repository.get_by_id(existing_uf.file_id)
                title = existing_uf.custom_title or (content_file.media_metadata or {}).get("title") if content_file else parsed.title
                logger.info("[Summary] Reusing existing user_file_id=%d for summarization (no cached summary)", existing_uf.id)
                return ContentToSummarize(
                    text=parsed.text,
                    title=title or parsed.title,
                    source_url=url,
                    content_file_id=existing_uf.file_id,
                    user_file_id=existing_uf.id,
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
        return ContentToSummarize(
            text=parsed.text,
            title=parsed.title,
            source_url=url,
            content_file_id=user_file.file_id,
            user_file_id=user_file.id,
        )
    
    _MIME_TO_EXT: dict = {
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
        "text/plain": "txt",
        "text/markdown": "md",
        "text/x-markdown": "md",
    }

    async def get_content_for_summarization_file(
        self,
        file_content: bytes,
        filename: str,
        user_id: int,
        content_type: Optional[str] = None,
        namespace_id: Optional[int] = None,
    ) -> Union[SummaryResponse, ContentToSummarize]:
        """
        Получить контент для суммаризации из загруженного файла.
        Кэш по хэшу контента; иначе сохраняет файл и возвращает ContentToSummarize.
        """
        file_ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if not file_ext and content_type:
            mime = content_type.split(";")[0].strip().lower()
            file_ext = self._MIME_TO_EXT.get(mime, "")
        if not file_ext:
            raise ValueError("Не удалось определить тип файла")
        text = self.file_content_service.extract_text(
            file_content,
            file_ext=file_ext,
            strict=True,
        )
        if not text or not text.strip():
            raise ValueError("Не удалось извлечь текст из файла")
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        dedup_result = await self.file_service.check_deduplication(
            user_id=user_id,
            content_hash=content_hash,
        )
        if dedup_result.is_duplicate and dedup_result.existing_file_id:
            existing_uf = await self.user_file_repository.get_by_id(dedup_result.existing_file_id)
            if existing_uf:
                existing_summary = await self.summary_repository.get_by_file_id(existing_uf.file_id)
                if existing_summary:
                    logger.info("[Summary] Cached summary for file hash: %s", content_hash[:16])
                    content_file = await self.file_repository.get_by_id(existing_uf.file_id)
                    title = existing_uf.custom_title or (content_file.media_metadata or {}).get("title") if content_file else "Document"
                    return SummaryResponse(
                        user_file_id=existing_uf.id,
                        content_file_id=existing_uf.file_id,
                        summary=existing_summary.text,
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
            content_hash=content_hash,
            namespace_id=namespace_id,
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
        content_file = await self.file_repository.get_by_id(uf.file_id)
        title = uf.custom_title or (content_file.media_metadata or {}).get("title") if content_file else "Document"
        source_url = content_file.source_url if content_file else None
        return SummaryResponse(
            user_file_id=uf.id,
            content_file_id=uf.file_id,
            summary=summary.text,
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
        cf = await self.file_repository.get_by_id(uf.file_id)
        if not cf:
            raise ValueError("Контент файла не найден")
        existing_summary = await self.summary_repository.get_by_file_id(cf.id)
        if existing_summary:
            logger.info("[Summary] Cached summary for file_id=%d", file_id)
            title = uf.custom_title or (cf.media_metadata or {}).get("title") or "Document"
            return SummaryResponse(
                user_file_id=uf.id,
                content_file_id=cf.id,
                summary=existing_summary.text,
                title=title,
                source_url=cf.source_url,
                is_cached=True,
                method=SummaryMethod.CACHED,
            )
        file_path = cf.file_path
        if not file_path:
            raise ValueError("Путь к файлу не найден")
        logger.info("[Summary] Loading file content from storage: %s", file_path)
        text = await self.file_content_service.get_text_content(
            cf,
            filename=uf.custom_title or (cf.media_metadata or {}).get("title") or "document",
            strict=True,
        )
        if not text or not text.strip():
            raise ValueError("Не удалось извлечь текст из файла")
        filename = uf.custom_title or (cf.media_metadata or {}).get("title") or "document"
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
        existing = await self.summary_repository.get_by_file_id(content_file_id)
        if existing:
            logger.info("[Summary] Summary already exists for file_id=%d, skipping save", content_file_id)
            return
        await self.summary_repository.create(
            file_id=content_file_id,
            text=summary_result.content,
            model_name=summary_result.model_name,
        )
        await self.db.commit()
        logger.info("[Summary] Saved summary for file_id=%d, method=%s", content_file_id, summary_result.method)

    @staticmethod
    def build_summary_response(content: ContentToSummarize, summary_result: SummaryResult) -> SummaryResponse:
        """Собрать SummaryResponse из контента и результата суммаризации."""
        return SummaryResponse(
            user_file_id=content.user_file_id,
            content_file_id=content.content_file_id,
            summary=summary_result.content,
            title=content.title,
            source_url=content.source_url,
            is_cached=False,
            method=summary_result.method,
        )

    async def extract_url_content(
        self,
        url: str,
        user_id: int,
    ) -> ContentExtractResponse:
        """
        Извлекает контент по URL и сохраняет без привязки к пользователю (без UserFile).
        При повторном запросе с тем же URL или контентом возвращает уже существующий file_id.

        Returns:
            ContentExtractResponse с file_id (content File.id), title, parsed_content, source_url.
        """
        existing_file = await self.file_repository.get_by_source_url(url)
        if existing_file:
            logger.info("[Extract] Reusing existing file for URL: %s (file_id=%d)", url, existing_file.id)
            title = (existing_file.media_metadata or {}).get("title")
            text = None
            if existing_file.transcript_text:
                text = existing_file.transcript_text
            elif existing_file.file_path:
                try:
                    text = await self.file_content_service.get_text_content(
                        existing_file,
                        strict=False,
                    ) or None
                except Exception:
                    pass
            return ContentExtractResponse(
                file_id=existing_file.id,
                title=title,
                parsed_content=text,
                source_url=url,
            )

        logger.info("[Extract] Parsing URL: %s (user=%d)", url, user_id)
        parsed = await self.content_extractor.extract(url)

        existing_by_hash = await self.file_repository.get_by_content_hash(parsed.content_hash)
        if existing_by_hash:
            logger.info("[Extract] Reusing existing file by hash (file_id=%d)", existing_by_hash.id)
            return ContentExtractResponse(
                file_id=existing_by_hash.id,
                title=(existing_by_hash.media_metadata or {}).get("title") or parsed.title,
                parsed_content=parsed.text,
                source_url=url,
            )

        content_file = await self.file_service.save_unattached_content(
            text=parsed.text,
            title=parsed.title,
            source_url=url,
            content_hash=parsed.content_hash,
            content_type=parsed.content_type,
            user_id=user_id,
        )
        return ContentExtractResponse(
            file_id=content_file.id,
            title=parsed.title,
            parsed_content=parsed.text,
            source_url=url,
        )

    async def get_content_for_summarization_by_content_file_id(
        self,
        content_file_id: int,
    ) -> Union[SummaryResponse, ContentToSummarize]:
        """
        Получить контент для суммаризации по ID контент-файла (File.id).
        Работает для непривязанных файлов (без записи в user_files).
        Если суммаризация уже есть — возвращает SummaryResponse. Иначе — ContentToSummarize.
        """
        cf = await self.file_repository.get_by_id(content_file_id)
        if not cf:
            raise ValueError(f"Файл с ID {content_file_id} не найден")

        existing_summary = await self.summary_repository.get_by_file_id(cf.id)
        if existing_summary:
            title = (cf.media_metadata or {}).get("title") or "Document"
            logger.info("[Summary] Cached summary for content_file_id=%d", content_file_id)
            return SummaryResponse(
                user_file_id=None,
                content_file_id=cf.id,
                summary=existing_summary.text,
                title=title,
                source_url=cf.source_url,
                is_cached=True,
                method=SummaryMethod.CACHED,
            )

        if cf.transcript_text and cf.transcript_text.strip():
            text = cf.transcript_text
        elif cf.file_path:
            logger.info("[Summary] Loading content from storage: %s", cf.file_path)
            text = await self.file_content_service.get_text_content(
                cf,
                filename=(cf.media_metadata or {}).get("title") or "document.md",
                strict=True,
            )
        else:
            raise ValueError("Путь к файлу не найден и transcript_text отсутствует")

        if not text or not text.strip():
            raise ValueError("Не удалось извлечь текст из файла")

        title = (cf.media_metadata or {}).get("title") or "Document"
        return ContentToSummarize(
            text=text,
            title=title,
            source_url=cf.source_url,
            content_file_id=cf.id,
            user_file_id=None,
        )

    async def get_summary_by_content_file_id(
        self,
        content_file_id: int,
    ) -> Optional[SummaryResponse]:
        """Получить суммаризацию по ID контент-файла (File.id) — работает и для непривязанных файлов."""
        cf = await self.file_repository.get_by_id(content_file_id)
        if not cf:
            return None
        summary = await self.summary_repository.get_by_file_id(cf.id)
        if not summary:
            return None
        title = (cf.media_metadata or {}).get("title") or "Document"
        return SummaryResponse(
            user_file_id=None,
            content_file_id=cf.id,
            summary=summary.text,
            title=title,
            source_url=cf.source_url,
            is_cached=True,
            method=SummaryMethod.CACHED,
        )

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
            if existing_uf:
                content_file = await self.file_repository.get_by_id(existing_uf.file_id)
                if content_file and content_file.file_path:
                    logger.info("[Ingest] URL already indexed: %s (user_file_id=%d)", url, existing_uf.id)
                    text = await self.file_content_service.get_text_content(
                        content_file,
                        filename=existing_uf.custom_title or (content_file.media_metadata or {}).get("title") or "document",
                        strict=False,
                    )
                    filename = existing_uf.custom_title or (content_file.media_metadata or {}).get("title") or "document"
                else:
                    text = ""
                    filename = existing_uf.custom_title or "document"
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
        )
        
        if dedup_result.is_duplicate and dedup_result.existing_file_id:
            existing_uf = await self.user_file_repository.get_by_id(dedup_result.existing_file_id)
            if existing_uf:
                logger.info("[Ingest] Content hash match: %s (user_file_id=%d)", parsed.content_hash[:16], existing_uf.id)
                content_file = await self.file_repository.get_by_id(existing_uf.file_id)
                filename = existing_uf.custom_title or (content_file.media_metadata or {}).get("title") if content_file else parsed.title
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
