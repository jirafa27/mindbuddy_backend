"""ContentExtractorService — унифицированный сервис извлечения контента."""
import logging
from typing import List, Optional

from app.domain.entities import ParsedContent
from app.domain.protocols import ContentParser
from app.infrastructure.parsers import YouTubeParser, HTMLParser
from app.core.exceptions import ContentExtractionError

logger = logging.getLogger(__name__)


class ContentExtractorService:
    """
    Сервис извлечения контента из различных источников.
    
    Поддерживает:
    - YouTube видео (субтитры или описание при недоступности субтитров)
    - HTML страницы (чистый текст)
    """
    
    def __init__(self, parsers: Optional[List[ContentParser]] = None):
        """Инициализирует сервис с набором парсеров.
        
        Args:
            parsers: Список парсеров. По умолчанию используются YouTube и HTML парсеры.
        """
        self.parsers: List[ContentParser] = parsers or [
            YouTubeParser(),
            HTMLParser(),
        ]
    
    def _find_parser(self, url: str) -> Optional[ContentParser]:
        """Находит подходящий парсер для URL."""
        for parser in self.parsers:
            if parser.can_handle(url):
                return parser
        return None
    
    async def extract(self, url: str) -> ParsedContent:
        """Извлекает контент из URL.
        
        Args:
            url: URL источника (YouTube, веб-страница).
            
        Returns:
            ParsedContent с текстом, хэшем и метаданными.
            
        Raises:
            ContentExtractionError: Если URL не поддерживается или произошла ошибка парсинга.
        """
        parser = self._find_parser(url)
        if not parser:
            raise ContentExtractionError(f"Неподдерживаемый тип URL: {url}")
        
        logger.info("[ContentExtractor] Using %s for URL: %s", parser.__class__.__name__, url)
        
        try:
            return await parser.parse(url)
        except ContentExtractionError:
            logger.exception("[ContentExtractor] ContentExtractionError for URL %s", url)
            raise
