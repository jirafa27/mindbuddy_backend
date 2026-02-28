"""Очистка текста транскрипта от таймкодов и разметки — на выходе чистый текст/Markdown для Obsidian."""
import re


def clean_transcript_to_markdown(raw: str) -> str:
    """
    Убирает таймкоды и лишнюю разметку из текста транскрипта (YouTube, SRT, VTT и т.п.),
    возвращает удобный для Obsidian текст (абзацы, без технических меток).
    """
    if not raw or not raw.strip():
        return ""
    text = raw.strip()
    # Удаление SRT/VTT таймкодов: 00:00:00,000 --> 00:00:01,000
    text = re.sub(r"\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?\s*-->\s*\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?", "", text)
    # Удаление меток в квадратных скобках [00:00:00] или [00:00]
    text = re.sub(r"\[\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?\]", "", text)
    # Удаление меток без скобок в начале строки 00:00:00 или 00:00
    text = re.sub(r"^(?:\d{1,2}:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?\s*", "", text, flags=re.MULTILINE)
    # Номера реплик (1, 2, 3...)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    # HTML-теги
    text = re.sub(r"<[^>]+>", "", text)
    # Лишние пробелы и пустые строки подряд — схлопнуть в максимум один перенос
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Строки из одной точки/тире
    text = re.sub(r"^[.\-\s]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
