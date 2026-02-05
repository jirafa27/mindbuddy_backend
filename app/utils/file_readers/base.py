from abc import ABC, abstractmethod


class BaseFileReader(ABC):
    """Базовый класс для чтения файлов разных типов"""

    @abstractmethod
    def read(self, file_content: bytes) -> str:
        """
        Извлекает текст из файла.

        Args:
            file_content: Содержимое файла в виде bytes

        Returns:
            Извлечённый текст

        Raises:
            Exception: При ошибке извлечения текста
        """
        pass

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Возвращает список поддерживаемых расширений файлов"""
        pass
