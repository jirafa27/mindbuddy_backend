import asyncio
import httpx
import time
from typing import Optional
from redis.asyncio import Redis
from redis.asyncio.connection import ConnectionPool
from app.core.config import settings
from app.core.exceptions import EmbeddingGenerationError


class YandexIAMService:
    def __init__(self):
        self.oauth_token: Optional[str] = settings.YANDEX_OAUTH_TOKEN
        self.iam_token: Optional[str] = settings.YANDEX_IAM_TOKEN
        self.token_url = settings.YANDEX_IAM_TOKEN_URL
        self.timeout = settings.YANDEX_IAM_TIMEOUT
        self._lock = asyncio.Lock()
        self._redis: Optional[Redis] = None
        self._redis_key = "yandex:iam_token"
        self._token_ttl = 11 * 60 * 60
        self._token_obtained_at: Optional[float] = None

    async def _get_redis(self) -> Optional[Redis]:
        if self._redis is None:
            pool = ConnectionPool.from_url(settings.REDIS_URL)
            self._redis = Redis(connection_pool=pool, decode_responses=True)
        return self._redis

    async def _get_token_from_redis(self) -> Optional[str]:
        try:
            redis = await self._get_redis()
            return await redis.get(self._redis_key)
        except Exception:
            return None

    async def _get_ttl_from_redis(self) -> Optional[int]:
        try:
            redis = await self._get_redis()
            ttl = await redis.ttl(self._redis_key)
            return ttl if ttl > 0 else None
        except Exception:
            return None

    async def _save_token_to_redis(self, token: str) -> None:
        try:
            redis = await self._get_redis()
            await redis.setex(self._redis_key, self._token_ttl, token)
        except Exception:
            pass

    async def _delete_token_from_redis(self) -> None:
        try:
            redis = await self._get_redis()
            await redis.delete(self._redis_key)
        except Exception:
            pass

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()
            self._redis = None

    async def get_iam_token(self, force_refresh: bool = False) -> str:
        if self.iam_token and not force_refresh and not self.oauth_token:
            return self.iam_token

        if self.oauth_token:
            async with self._lock:
                if force_refresh:
                    await self._delete_token_from_redis()
                    self.iam_token = None
                    self._token_obtained_at = None

                if self.iam_token and not force_refresh:
                    if self._token_obtained_at is not None:
                        elapsed_time = time.time() - self._token_obtained_at
                        if elapsed_time < self._token_ttl:
                            return self.iam_token
                        self.iam_token = None
                        self._token_obtained_at = None
                    else:
                        redis_ttl = await self._get_ttl_from_redis()
                        if redis_ttl is not None and redis_ttl > 0:
                            self._token_obtained_at = time.time() - (self._token_ttl - redis_ttl)
                            return self.iam_token
                        self.iam_token = None

                if not force_refresh:
                    cached_token = await self._get_token_from_redis()
                    if cached_token:
                        self.iam_token = cached_token
                        redis_ttl = await self._get_ttl_from_redis()
                        if redis_ttl is not None and redis_ttl > 0:
                            self._token_obtained_at = time.time() - (self._token_ttl - redis_ttl)
                        else:
                            self._token_obtained_at = time.time()
                        return self.iam_token

                self.iam_token = await self._exchange_oauth_for_iam()
                self._token_obtained_at = time.time()
                await self._save_token_to_redis(self.iam_token)
                return self.iam_token

        if not self.iam_token:
            raise EmbeddingGenerationError(
                "Yandex IAM token or OAuth token must be configured"
            )
        return self.iam_token

    async def _exchange_oauth_for_iam(self) -> str:
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
