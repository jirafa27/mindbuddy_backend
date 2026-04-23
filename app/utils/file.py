"""Утилиты для работы с файлами и их именами."""
from typing import Optional
from urllib.parse import unquote, quote


TEXT_INLINE_EDITABLE_EXTENSIONS = frozenset(
    {
        "txt",
        "md",
        "json",
        "py",
        "js",
        "ts",
        "jsx",
        "tsx",
        "html",
        "css",
        "scss",
        "xml",
        "csv",
        "yml",
        "yaml",
        "ini",
        "toml",
        "sql",
    }
)


def normalize_file_ext(file_ext: Optional[str]) -> str:
    if not file_ext:
        return ""
    return str(file_ext).strip().lower().lstrip(".")


def can_inline_edit_file(file_ext: Optional[str]) -> bool:
    return normalize_file_ext(file_ext) in TEXT_INLINE_EDITABLE_EXTENSIONS


def should_include_text_content_in_sync(file_ext: Optional[str]) -> bool:
    return can_inline_edit_file(file_ext)


def decode_filename(filename: str) -> str:
    """
    Декодирует URL-encoded имя файла.
    
    Args:
        filename: Имя файла (может быть URL-encoded)
        
    Returns:
        Декодированное имя файла
    """
    if not filename:
        return filename
    
    try:
        decoded = unquote(filename, encoding='utf-8')
        # Используем декодированное имя, если оно отличается от исходного
        if decoded != filename:
            return decoded
    except Exception:
        # Если декодирование не удалось, возвращаем исходное имя
        pass
    
    return filename


def encode_filename_for_header(filename: str) -> str:
    """
    Кодирует имя файла для использования в HTTP заголовке Content-Disposition.
    Использует RFC 5987 формат для поддержки не-ASCII символов.
    
    Args:
        filename: Имя файла (может содержать не-ASCII символы)
        
    Returns:
        Строка заголовка Content-Disposition с правильно закодированным именем файла
Ы    """
    if not filename:
        return 'attachment; filename=""'
    
    # Проверяем, содержит ли имя файла не-ASCII символы
    try:
        filename.encode('ascii')
        # Если имя файла содержит только ASCII, используем простой формат
        return f'attachment; filename="{filename}"'
    except UnicodeEncodeError:
        # Если есть не-ASCII символы, используем RFC 5987 формат
        # Кодируем имя файла в UTF-8 и затем URL-encode
        encoded = quote(filename, safe='')
        # Используем оба формата: простой для совместимости и RFC 5987 для не-ASCII
        # Простой формат используем с ASCII-совместимым именем (fallback)
        ascii_fallback = filename.encode('ascii', 'ignore').decode('ascii') or 'file'
        # RFC 5987 формат: filename*=UTF-8''encoded_name (две одинарные кавычки)
        # Используем конкатенацию для правильного экранирования
        return f'attachment; filename="{ascii_fallback}"; filename*=UTF-8' + "''" + encoded
