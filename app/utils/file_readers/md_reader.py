from .base import BaseFileReader


class MarkdownReader(BaseFileReader):
    """Reader для Markdown файлов (.md)"""

    def read(self, file_content: bytes) -> str:
        """
        Извлекает текст из .md файла.

        Args:
            file_content: Содержимое файла в виде bytes

        Returns:
            Текст файла в формате Markdown

        Raises:
            UnicodeDecodeError: При ошибке декодирования
        """
        return file_content.decode("utf-8")

    @property
    def supported_extensions(self) -> list[str]:
        return ["md"]
