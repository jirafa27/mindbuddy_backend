from .base import BaseFileReader


class TxtReader(BaseFileReader):
    """Reader для текстовых файлов (.txt)"""

    def read(self, file_content: bytes) -> str:
        """
        Извлекает текст из .txt файла.

        Args:
            file_content: Содержимое файла в виде bytes

        Returns:
            Текст файла

        Raises:
            UnicodeDecodeError: При ошибке декодирования
        """
        return file_content.decode("utf-8")

    @property
    def supported_extensions(self) -> list[str]:
        return ["txt"]
