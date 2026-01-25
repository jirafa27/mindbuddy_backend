import asyncio
import httpx
import time
from typing import Optional
from redis.asyncio import Redis
from redis.asyncio.connection import ConnectionPool
from app.core.config import settings
from app.core.exceptions import EmbeddingGenerationError


class YandexIAMService:
    """Сервис для получения и управления IAM-токенами Yandex Cloud"""

    def __init__(self):
        self.oauth_token: Optional[str] = settings.YANDEX_OAUTH_TOKEN
        self.iam_token: Optional[str] = settings.YANDEX_IAM_TOKEN
        self.token_url = settings.YANDEX_IAM_TOKEN_URL
        self.timeout = settings.YANDEX_IAM_TIMEOUT
        self._lock = asyncio.Lock()
        self._redis: Optional[Redis] = None
        self._redis_key = "yandex:iam_token"
        self._token_ttl = 11 * 60 * 60  # 11 часов (чуть меньше 12, чтобы обновить заранее)
        self._token_obtained_at: Optional[float] = None  # Время получения токена (timestamp)

    async def _get_redis(self) -> Optional[Redis]:
        """Получает или создает Redis клиент"""
        if self._redis is None:
            pool = ConnectionPool.from_url(settings.REDIS_URL)
            self._redis = Redis(connection_pool=pool, decode_responses=True)
        return self._redis

    async def _get_token_from_redis(self) -> Optional[str]:
        """Получает токен из Redis"""
        try:
            redis = await self._get_redis()
            return await redis.get(self._redis_key)
        except Exception:
            # Если Redis недоступен, возвращаем None
            return None

    async def _get_ttl_from_redis(self) -> Optional[int]:
        """Получает оставшееся время жизни токена в Redis (в секундах)"""
        try:
            redis = await self._get_redis()
            ttl = await redis.ttl(self._redis_key)
            return ttl if ttl > 0 else None
        except Exception:
            return None

    async def _save_token_to_redis(self, token: str) -> None:
        """Сохраняет токен в Redis"""
        try:
            redis = await self._get_redis()
            await redis.setex(self._redis_key, self._token_ttl, token)
        except Exception:
            # Если не удалось сохранить в Redis, продолжаем работу
            pass

    async def _delete_token_from_redis(self) -> None:
        """Удаляет токен из Redis"""
        try:
            redis = await self._get_redis()
            await redis.delete(self._redis_key)
        except Exception:
            # Если Redis недоступен, игнорируем
            pass

    async def close(self) -> None:
        """Закрывает соединение с Redis"""
        if self._redis:
            await self._redis.close()
            self._redis = None

    async def get_iam_token(self, force_refresh: bool = False) -> str:
        """
        Получает IAM-токен. Обновляется только при force_refresh=True (обычно при ошибке 401).

        Args:
            force_refresh: Принудительно обновить токен (используется при получении 401)

        Returns:
            IAM-токен

        Raises:
            EmbeddingGenerationError: При ошибке получения токена
        """
        # Если токен задан напрямую в конфиге и не требуется обновление, используем его
        if self.iam_token and not force_refresh and not self.oauth_token:
            return self.iam_token

        # Если есть OAuth-токен
        if self.oauth_token:
            async with self._lock:
                # Если требуется обновление, удаляем из Redis и памяти
                if force_refresh:
                    await self._delete_token_from_redis()
                    self.iam_token = None
                    self._token_obtained_at = None

                # Если токен уже есть в памяти и не требуется обновление, проверяем его срок действия
                if self.iam_token and not force_refresh:
                    # Проверяем, не истек ли токен (с учетом TTL)
                    if self._token_obtained_at is not None:
                        elapsed_time = time.time() - self._token_obtained_at
                        if elapsed_time < self._token_ttl:
                            return self.iam_token
                        # Токен истек, очищаем его
                        self.iam_token = None
                        self._token_obtained_at = None
                    else:
                        # Если время получения неизвестно, проверяем Redis TTL
                        # Если токен есть в Redis, значит он еще не истек
                        redis_ttl = await self._get_ttl_from_redis()
                        if redis_ttl is not None and redis_ttl > 0:
                            # Токен есть в Redis и еще не истек, используем его
                            # Устанавливаем время получения на основе оставшегося TTL
                            self._token_obtained_at = time.time() - (self._token_ttl - redis_ttl)
                            return self.iam_token
                        # Токена нет в Redis или он истек, очищаем память
                        self.iam_token = None

                # Пытаемся получить токен из Redis
                if not force_refresh:
                    cached_token = await self._get_token_from_redis()
                    if cached_token:
                        self.iam_token = cached_token
                        # Получаем оставшееся время жизни из Redis
                        redis_ttl = await self._get_ttl_from_redis()
                        if redis_ttl is not None and redis_ttl > 0:
                            # Устанавливаем время получения на основе оставшегося TTL
                            self._token_obtained_at = time.time() - (self._token_ttl - redis_ttl)
                        else:
                            # Если не удалось получить TTL, используем текущее время
                            self._token_obtained_at = time.time()
                        return self.iam_token

                # Получаем новый IAM-токен (при первом запросе или при force_refresh)
                self.iam_token = await self._exchange_oauth_for_iam()
                self._token_obtained_at = time.time()  # Запоминаем время получения
                # Сохраняем в Redis
                await self._save_token_to_redis(self.iam_token)
                return self.iam_token

        # Если нет ни IAM, ни OAuth токена
        if not self.iam_token:
            raise EmbeddingGenerationError(
                "Yandex IAM token or OAuth token must be configured"
            )

        return self.iam_token

    async def _exchange_oauth_for_iam(self) -> str:
        """
        Обменивает OAuth-токен на IAM-токен.

        Returns:
            IAM-токен

        Raises:
            EmbeddingGenerationError: При ошибке обмена токена
        """
        if not self.oauth_token:
            raise EmbeddingGenerationError("Yandex OAuth token not configured")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.token_url,
                json={"yandexPassportOauthToken": self.oauth_token},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()

            if "iamToken" in data:
                return data["iamToken"]
            
            raise EmbeddingGenerationError(
                "Invalid response format from Yandex IAM API"
            )
