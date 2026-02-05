import io
import PyPDF2
from .base import BaseFileReader


class PdfReader(BaseFileReader):
    """Reader для PDF файлов (.pdf)"""

    def read(self, file_content: bytes) -> str:
        """
        Извлекает текст из .pdf файла.

        Args:
            file_content: Содержимое файла в виде bytes

        Returns:
            Извлечённый текст из всех страниц PDF

        Raises:
            Exception: При ошибке чтения PDF
        """
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        text_parts = []
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text_parts.append(extracted_text)

        return "\n\n".join(text_parts)

    @property
    def supported_extensions(self) -> list[str]:
        return ["pdf"]
