import io
from docx import Document
from .base import BaseFileReader


class DocxReader(BaseFileReader):
    """Reader для Word документов (.docx)"""

    def read(self, file_content: bytes) -> str:
        """
        Извлекает текст из .docx файла.

        Args:
            file_content: Содержимое файла в виде bytes

        Returns:
            Извлечённый текст из всех параграфов документа

        Raises:
            Exception: При ошибке чтения DOCX
        """
        docx_file = io.BytesIO(file_content)
        doc = Document(docx_file)

        text_parts = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)

        return "\n\n".join(text_parts)

    @property
    def supported_extensions(self) -> list[str]:
        return ["docx"]
