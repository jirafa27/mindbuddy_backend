"""Парсер HTML страниц — извлечение чистого текста."""
import logging
from urllib.parse import urlparse

import trafilatura

from app.domain.entities import ParsedContent, ContentType
from app.domain.protocols import ContentParser
from app.core.exceptions import ContentExtractionError

logger = logging.getLogger(__name__)


class HTMLParser(ContentParser):
    """Извлекает чистый текст из HTML страниц через trafilatura."""
    
    def can_handle(self, url: str) -> bool:
        """Проверяет, является ли URL HTTP(S) ссылкой (не YouTube)."""
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return False
        # Исключаем YouTube — его обрабатывает YouTubeParser
        if 'youtube.com' in parsed.netloc or 'youtu.be' in parsed.netloc:
            return False
        return True
    
    async def parse(self, url: str) -> ParsedContent:
        """Извлекает текст из веб-страницы."""
        logger.info("[HTML] Extracting content from: %s", url)
        
        try:
            # trafilatura скачивает и парсит страницу
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                raise ContentExtractionError(f"Не удалось загрузить страницу: {url}")
            
            # Извлекаем текст
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            )
            
            if not text:
                raise ContentExtractionError(f"Не удалось извлечь текст из страницы: {url}")
            
            # Пробуем получить заголовок
            metadata = trafilatura.extract_metadata(downloaded)
            title = metadata.title if metadata and metadata.title else urlparse(url).netloc
            
        except ContentExtractionError:
            raise  # Пробрасываем наше исключение дальше
        except Exception as e:
            logger.exception("[HTML] Failed to parse %s", url)
            raise ContentExtractionError(f"Ошибка парсинга страницы: {e}")
        
        content_hash = ParsedContent.compute_hash(text)
        
        logger.info("[HTML] Extracted %d chars from %s", len(text), url)
        
        return ParsedContent(
            text=text,
            title=title,
            source_url=url,
            content_hash=content_hash,
            content_type=ContentType.HTML,
            file_size=len(text.encode("utf-8")),
        )
