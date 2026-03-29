"""Общий паттерн HTTP(S) URL для роутера, CRUD и других узлов графа /ask."""
import re
from typing import Optional

HTTP_URL_RE = re.compile(
    r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b"
    r"[-a-zA-Z0-9()@:%_\+.~#?&//=]*",
    re.IGNORECASE,
)


def extract_first_http_url(text: str) -> Optional[str]:
    match = HTTP_URL_RE.search(text)
    return match.group(0) if match else None


def is_http_url_only(text: str) -> bool:
    """Строка целиком (после strip) — один HTTP(S) URL, без другого текста."""
    return bool(HTTP_URL_RE.fullmatch(text.strip()))
