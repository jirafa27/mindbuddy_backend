import logging
from typing import Optional

from app.core.exceptions import ValidationError
from app.domain.protocols import FileStorage
from app.infrastructure.db.models import File
from app.utils.file_readers import FileReaderFactory

logger = logging.getLogger(__name__)


class FileContentService:
    """Низкоуровневый сервис чтения и извлечения текста из файлов."""

    def __init__(
        self,
        *,
        storage: FileStorage,
        file_reader_factory: Optional[FileReaderFactory] = None,
    ) -> None:
        self.storage = storage
        self.file_reader_factory = file_reader_factory

    @staticmethod
    def _file_ext(filename: str) -> str:
        return filename.rsplit(".", 1)[-1].lower() if "." in filename else "md"

    def extract_text(
        self,
        file_bytes: bytes,
        *,
        filename: Optional[str] = None,
        file_ext: Optional[str] = None,
        strict: bool = True,
    ) -> str:
        if not file_bytes:
            return ""

        resolved_ext = file_ext or self._file_ext(filename or "")
        if self.file_reader_factory is not None:
            try:
                reader = self.file_reader_factory.get_reader(resolved_ext)
                return reader.read(file_bytes)
            except UnicodeDecodeError:
                if strict:
                    raise ValidationError(
                        "File encoding not supported. Please use UTF-8 encoded files."
                    )
                return ""
            except ValueError as e:
                if strict:
                    raise ValidationError(str(e))
                return ""
            except Exception as e:
                if strict:
                    raise ValidationError(f"Failed to extract text from file: {str(e)}")
                logger.warning(
                    "[FileContentService] Failed to extract text for .%s",
                    resolved_ext,
                )
                return ""

        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            if strict:
                raise ValidationError(
                    "File encoding not supported. Please use UTF-8 encoded files."
                )
            return ""

    async def get_text_content(
        self,
        content_file: File,
        *,
        filename: Optional[str] = None,
        strict: bool = False,
    ) -> str:
        if content_file.transcript_text:
            return content_file.transcript_text
        if not content_file.file_path:
            return ""

        try:
            file_bytes = await self.storage.download_file(content_file.file_path)
        except Exception as e:
            if strict:
                raise ValidationError(f"Failed to download file from storage: {str(e)}")
            logger.warning(
                "[FileContentService] Failed to download file content for file=%s",
                content_file.id,
            )
            return ""

        resolved_filename = (
            filename
            or (content_file.media_metadata or {}).get("title")
            or content_file.file_path.rsplit("/", 1)[-1]
        )
        return self.extract_text(
            file_bytes,
            filename=resolved_filename,
            strict=strict,
        )
