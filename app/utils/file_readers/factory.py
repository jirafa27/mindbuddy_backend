from typing import Dict
from .base import BaseFileReader
from .txt_reader import TxtReader
from .md_reader import MarkdownReader
from .pdf_reader import PdfReader
from .docx_reader import DocxReader


class FileReaderFactory:
    """Фабрика для создания нужного reader'а в зависимости от типа файла"""

    def __init__(self):
        """Инициализирует фабрику и регистрирует все доступные reader'ы"""
        self._readers: Dict[str, BaseFileReader] = {}
        self._register_readers()

    def _register_readers(self):
        """Регистрирует все доступные reader'ы"""
        readers = [
            TxtReader(),
            MarkdownReader(),
            PdfReader(),
            DocxReader(),
        ]

        for reader in readers:
            for ext in reader.supported_extensions:
                self._readers[ext] = reader

    def get_reader(self, file_extension: str) -> BaseFileReader:
        """
        Возвращает reader для указанного расширения файла.

        Args:
            file_extension: Расширение файла (без точки, например 'pdf')

        Returns:
            Экземпляр Reader для данного типа файла

        Raises:
            ValueError: Если для данного типа файла нет Reader'а
        """
        file_extension = file_extension.lower()
        
        if file_extension not in self._readers:
            raise ValueError(f"No reader available for file extension: {file_extension}")
        
        return self._readers[file_extension]
