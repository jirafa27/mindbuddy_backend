from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Project
    PROJECT_NAME: str = "MindBuddy"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@db:5432/mindbuddy"
    # Тесты: по умолчанию под docker-compose.test.yml (user/password/mindbuddy_test на 5433)
    DATABASE_TEST_URL: Optional[str] = "postgresql+asyncpg://user:password@localhost:5433/mindbuddy_test"

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"

    # RabbitMQ
    RABBITMQ_URL: str = "amqp://guest:guest@rabbitmq:5672//"

    # MinIO / S3
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "mindbuddy-files"
    MINIO_BUCKET_TEMP_BLOBS: str = "temp-blobs"
    MINIO_SECURE: bool = False

    # Yandex Cloud API
    YANDEX_IAM_TOKEN: Optional[str] = None  # Можно задать напрямую или получить через OAuth
    YANDEX_OAUTH_TOKEN: Optional[str] = None  # OAuth-токен для автоматического получения IAM-токена
    YANDEX_FOLDER_ID: Optional[str] = None
    YANDEX_IAM_TOKEN_URL: str = "https://iam.api.cloud.yandex.net/iam/v1/tokens"
    YANDEX_IAM_TIMEOUT: float = 30.0
    YANDEX_EMBED_URL: str = "https://llm.api.cloud.yandex.net:443/foundationModels/v1/textEmbedding"
    YANDEX_EMBED_TIMEOUT: float = 30.0
    YANDEX_COMPLETION_URL: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    YANDEX_COMPLETION_TIMEOUT: float = 60.0
    
    # Окно контекста модели (YandexGPT Lite, 8B, до 32k токенов)
    YANDEX_COMPLETION_CONTEXT_TOKENS: int = 32_000
    YANDEX_SUMMARY_CONTEXT_RESERVE: int = 2000

    # Ollama (локальный LLM + эмбеддинги)
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    OLLAMA_LLM_MODEL: str = "qwen2.5:14b"
    OLLAMA_EMBED_MODEL: str = "qwen2.5:7b"
    OLLAMA_EMBED_DIM: int = 3584
    OLLAMA_TIMEOUT: float = 300.0
    # Отдельная модель и таймаут для суммаризации (меньше/быстрее)
    OLLAMA_SUMMARY_MODEL: str = "qwen2.5:7b"
    OLLAMA_SUMMARY_TIMEOUT: float = 300.0

    # OpenRouter (внешний LLM API, OpenAI-совместимый)
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_LLM_MODEL: str = "openai/gpt-4o-mini"
    OPENROUTER_TIMEOUT: float = 60.0

    # Text Chunking
    CHUNK_SIZE: int = 512  # tokens
    CHUNK_OVERLAP: int = 50  # tokens overlap between chunks

    # File Upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_FILE_TYPES: list[str] = ["md", "txt", "pdf", "docx"]

    # User-Agent для yt_dlp (опционально)
    USER_AGENT: Optional[str] = None
    # IPv4-прокси для yt-dlp (YouTube), например http://user:pass@host:port
    YOUTUBE_PROXY: Optional[str] = None

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30 * 24 * 60

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

