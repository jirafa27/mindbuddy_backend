from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Project
    PROJECT_NAME: str = "MindBuddy"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@db:5432/mindbuddy"

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"

    # RabbitMQ
    RABBITMQ_URL: str = "amqp://guest:guest@rabbitmq:5672//"

    # Yandex Cloud API
    YANDEX_IAM_TOKEN: Optional[str] = None  # Можно задать напрямую или получить через OAuth
    YANDEX_OAUTH_TOKEN: Optional[str] = None  # OAuth-токен для автоматического получения IAM-токена
    YANDEX_FOLDER_ID: Optional[str] = None
    YANDEX_IAM_TOKEN_URL: str = "https://iam.api.cloud.yandex.net/iam/v1/tokens"
    YANDEX_IAM_TIMEOUT: float = 30.0
    YANDEX_EMBED_URL: str = "https://llm.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding"
    YANDEX_EMBED_TIMEOUT: float = 30.0

    # Text Chunking
    CHUNK_SIZE: int = 512  # tokens
    CHUNK_OVERLAP: int = 50  # tokens overlap between chunks

    # File Upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_FILE_TYPES: list[str] = ["md", "txt", "pdf", "docx"]

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

