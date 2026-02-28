"""Парсер YouTube: только транскрипт через youtube-transcript-api (без загрузки видео и OAuth). Поддержка прокси через YOUTUBE_PROXY. Название видео — через oEmbed (та же экосистема, без новых библиотек)."""
import asyncio
import logging
import os
import re
from typing import Optional
from urllib.parse import parse_qs, urlparse

import httpx
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)
from youtube_transcript_api.proxies import GenericProxyConfig

from app.domain.entities import ParsedContent, ContentType
from app.domain.protocols import ContentParser
from app.utils.transcript_cleaner import clean_transcript_to_markdown

logger = logging.getLogger(__name__)


def _get_video_id(url: str) -> Optional[str]:
    """Достаёт video_id из URL (параметр v или путь youtu.be/ID)."""
    parsed = urlparse(url)
    if parsed.hostname and "youtu.be" in parsed.hostname:
        return (parsed.path or "").strip("/").split("/")[0] or None
    if parsed.hostname and "youtube.com" in parsed.hostname:
        qs = parse_qs(parsed.query)
        v = qs.get("v", [])
        if v:
            return v[0]
        for part in (parsed.path or "").split("/"):
            if part in ("embed", "shorts") and parsed.path:
                idx = parsed.path.split("/").index(part) + 1
                parts = parsed.path.split("/")
                if idx < len(parts):
                    return parts[idx]
    patterns = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def _clean_youtube_url(video_url: str) -> str:
    """Чистый URL без &t=, &list= для source_url."""
    video_id = _get_video_id(video_url)
    if not video_id:
        return video_url
    return f"https://www.youtube.com/watch?v={video_id}"


def _title_safe_for_filename(title: str) -> str:
    """Убирает из названия только символы, недопустимые в имени файла. Название не заменяется на англоязычный плейсхолдер."""
    if not title or not title.strip():
        return title
    # Запрещённые в файловой системе: \ / : * ? " < > |
    unsafe = r'\/:*?"<>|'
    result = "".join(c if c not in unsafe else " " for c in title)
    return " ".join(result.split()).strip() or title.strip()


def _fallback_parsed_content(url: str, video_id: Optional[str], error_msg: str = "") -> ParsedContent:
    """Безопасный объект при ошибке (избегаем 422). Title безопасен для MinIO."""
    text = error_msg or "Не удалось получить транскрипцию. Попробуйте позже."
    title = f"YouTube_Video_{video_id}" if video_id else "YouTube_Video_unknown"
    return ParsedContent(
        title=title,
        text=text,
        source_url=_clean_youtube_url(url) if url else "",
        content_hash=ParsedContent.compute_hash(text),
        content_type=ContentType.YOUTUBE,
        file_size=len(text.encode("utf-8")),
        fallback_used=True,
    )


async def _fetch_video_title(video_id: str) -> Optional[str]:
    """Получает название видео через YouTube oEmbed (без API-ключа). Использует YOUTUBE_PROXY при необходимости."""
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    proxy_url = os.getenv("YOUTUBE_PROXY")
    try:
        async with httpx.AsyncClient(proxy=proxy_url, timeout=10.0) as client:
            r = await client.get(
                "https://www.youtube.com/oembed",
                params={"url": video_url, "format": "json"},
            )
            r.raise_for_status()
            data = r.json()
            title = data.get("title") if isinstance(data, dict) else None
            if title and isinstance(title, str) and title.strip():
                return _title_safe_for_filename(title.strip())
    except Exception as e:
        logger.debug("[YouTube] oEmbed title fetch failed for %s: %s", video_id, e)
    return None


def _fetch_transcript_sync(video_id: str) -> tuple[str, bool]:
    """
    Синхронно получает транскрипт: приоритет ru/en, затем a.ru/a.en, затем первый доступный.
    Прокси из YOUTUBE_PROXY (http://user:pass@host:port) для обхода блокировок по IP.
    Возвращает (текст, True если успех).
    """
    proxy_url = os.getenv("YOUTUBE_PROXY")
    if proxy_url:
        proxy_config = GenericProxyConfig(http_url=proxy_url, https_url=proxy_url)
        api = YouTubeTranscriptApi(proxy_config=proxy_config)
        logger.debug("[YouTube] Using proxy from YOUTUBE_PROXY")
    else:
        api = YouTubeTranscriptApi()
    try:
        transcript_list = api.list(video_id)
    except TranscriptsDisabled:
        logger.warning("[YouTube] Transcripts disabled for video %s", video_id)
        return "", False
    except NoTranscriptFound:
        logger.warning("[YouTube] No transcript found for video %s", video_id)
        return "", False
    except Exception as e:
        logger.warning("[YouTube] list() failed for %s: %s", video_id, e)
        return "", False

    transcript = None
    try:
        transcript = transcript_list.find_transcript(["ru", "en"])
    except NoTranscriptFound:
        try:
            transcript = transcript_list.find_transcript(["a.ru", "a.en"])
        except NoTranscriptFound:
            try:
                transcript = next(iter(transcript_list))
            except StopIteration:
                return "", False
    except Exception as e:
        logger.debug("[YouTube] find_transcript failed: %s", e)

    if transcript is None:
        try:
            transcript = next(iter(transcript_list))
        except StopIteration:
            return "", False

    try:
        result = transcript.fetch()
    except Exception as e:
        logger.warning("[YouTube] transcript.fetch() failed for %s: %s", video_id, e)
        return "", False

    if not result:
        return "", False
    parts = []
    for item in result:
        if hasattr(item, "text"):
            parts.append(getattr(item, "text", "") or "")
        elif isinstance(item, dict):
            parts.append(item.get("text", "") or "")
        else:
            parts.append(str(item))
    text = " ".join(parts)
    text = (text or "").strip()
    if text:
        text = clean_transcript_to_markdown(text)
    return text, bool(text)


class YouTubeParser(ContentParser):
    """Извлечение транскрипта YouTube через youtube-transcript-api. Без OAuth и загрузки видео."""

    YOUTUBE_PATTERNS = [
        r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
    ]

    def can_handle(self, url: str) -> bool:
        return any(re.search(p, url) for p in self.YOUTUBE_PATTERNS)

    def _get_video_id(self, url: str) -> Optional[str]:
        return _get_video_id(url)

    async def parse(self, url: str) -> ParsedContent:
        video_id = self._get_video_id(url)
        clean_url = _clean_youtube_url(url)
        if not video_id:
            return _fallback_parsed_content(url, None, error_msg="Не удалось извлечь video_id из URL.")

        try:
            text, used_captions = await asyncio.to_thread(_fetch_transcript_sync, video_id)
            if not text or len(text.strip()) < 10:
                return _fallback_parsed_content(
                    url, video_id, error_msg="Транскрипт недоступен или отключён для этого видео."
                )

            title = await _fetch_video_title(video_id)
            if not title:
                title = f"YouTube Video {video_id}"
            content_hash = ParsedContent.compute_hash(text)
            logger.info("[YouTube] youtube-transcript-api success for %s: text_len=%d", video_id, len(text))

            return ParsedContent(
                text=text,
                title=title,
                source_url=clean_url,
                content_hash=content_hash,
                content_type=ContentType.YOUTUBE,
                file_size=len(text.encode("utf-8")),
                fallback_used=not used_captions,
            )
        except TranscriptsDisabled as e:
            logger.warning("[YouTube] TranscriptsDisabled: %s", e)
            return _fallback_parsed_content(url, video_id, error_msg="Субтитры отключены для этого видео.")
        except NoTranscriptFound as e:
            logger.warning("[YouTube] NoTranscriptFound: %s", e)
            return _fallback_parsed_content(url, video_id, error_msg="Транскрипт для этого видео не найден.")
        except Exception as e:
            logger.exception("[YouTube] parse failed: %s", e)
            return _fallback_parsed_content(url, video_id, error_msg=f"Ошибка загрузки: {e}")
