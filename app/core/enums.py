"""Перечисления для приложения"""
from enum import Enum


class FileSource(str, Enum):
    """Источник загрузки файла"""
    TELEGRAM = "telegram"
    WATCHER = "watcher"
