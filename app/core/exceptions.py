class AppException(Exception):
    """Базовое исключение приложения"""
    
    default_message = "An error occurred"
    
    def __init__(self, message: str | None = None):
        self.message = message if message is not None else self.default_message
        super().__init__(self.message)


class EmbeddingGenerationError(AppException):
    """Ошибка при генерации эмбеддинга"""
    default_message = "Error generating embeddings"


class FileProcessingError(AppException):
    """Ошибка при обработке файла"""
    default_message = "Error processing file"


class ValidationError(AppException):
    """Ошибка валидации данных"""
    default_message = "Validation error"


class NotFoundError(AppException):
    """Ресурс не найден"""
    default_message = "Resource not found"


class UnauthorizedError(AppException):
    """Нет или неверные учётные данные"""
    default_message = "Unauthorized"


class ForbiddenError(AppException):
    """Доступ запрещен"""
    default_message = "Access forbidden"


class ConflictError(AppException):
    """Конфликт версий или синхронизации"""
    default_message = "Conflict"

    def __init__(self, message: str | None = None, payload: dict | None = None):
        self.payload = payload
        super().__init__(message)


class FileTooLargeError(AppException):
    """Файл слишком большой"""
    default_message = "File size exceeds maximum allowed size"


class ContentExtractionError(AppException):
    """Ошибка извлечения контента (YouTube, веб-страницы)"""
    default_message = "Failed to extract content from URL"

