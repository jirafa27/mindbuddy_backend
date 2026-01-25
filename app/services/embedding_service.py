import asyncio
import httpx
from typing import List
from app.core.config import settings
from app.core.exceptions import EmbeddingGenerationError
from app.services.yandex_iam_service import YandexIAMService


class EmbeddingService:
    """Сервис для генерации эмбеддингов через Yandex Cloud API"""

    def __init__(self, iam_service: YandexIAMService):
        self.iam_service = iam_service
        self.folder_id = settings.YANDEX_FOLDER_ID
        self.embed_url = settings.YANDEX_EMBED_URL
        self.doc_uri = f"emb://{self.folder_id}/text-search-doc/latest"
        self.query_uri = f"emb://{self.folder_id}/text-search-query/latest"
        self.timeout = settings.YANDEX_EMBED_TIMEOUT

    async def _get_headers(self, force_refresh: bool = False) -> dict:
        """
        Возвращает заголовки для запроса к Yandex Cloud API

        Args:
            force_refresh: Принудительно обновить IAM-токен
        """
        iam_token = await self.iam_service.get_iam_token(force_refresh=force_refresh)
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {iam_token}",
            "x-folder-id": self.folder_id,
        }

    async def _get_embedding(self, text: str, text_type: str = "doc") -> List[float]:
        """
        Внутренний метод для генерации эмбеддинга одного текста.

        Args:
            text: Текст для векторизации
            text_type: Тип текста ("doc" для документов, "query" для запросов)

        Returns:
            Список чисел (вектор эмбеддинга)

        Raises:
            EmbeddingGenerationError: При ошибке генерации эмбеддинга
        """
        if not self.folder_id:
            raise EmbeddingGenerationError("Yandex folder ID not configured")

        if not text or not text.strip():
            raise EmbeddingGenerationError("Text cannot be empty")

        model_uri = self.doc_uri if text_type == "doc" else self.query_uri

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Первая попытка
            headers = await self._get_headers()
            response = await client.post(
                self.embed_url,
                headers=headers,
                json={
                    "modelUri": model_uri,
                    "text": text,
                },
            )
            
            # Если получили 401, обновляем токен и повторяем запрос
            if response.status_code == 401:
                headers = await self._get_headers(force_refresh=True)
                response = await client.post(
                    self.embed_url,
                    headers=headers,
                    json={
                        "modelUri": model_uri,
                        "text": text,
                    },
                )
            
            response.raise_for_status()
            data = response.json()

            if "embedding" in data:
                return data["embedding"]
            
            raise EmbeddingGenerationError("Invalid response format from Yandex Cloud API")

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Генерирует эмбеддинг для одного текста.

        Args:
            text: Текст для векторизации

        Returns:
            Список чисел (вектор эмбеддинга)

        Raises:
            EmbeddingGenerationError: При ошибке генерации эмбеддинга
        """
        return await self._get_embedding(text, text_type="doc")

    async def generate_embeddings_batch(
        self, texts: List[str]
    ) -> List[List[float]]:
        """
        Генерирует эмбеддинги для списка текстов (батч).

        Args:
            texts: Список текстов для векторизации

        Returns:
            Список эмбеддингов (каждый эмбеддинг - список чисел)
        """
        if not texts:
            return []

        # Yandex Cloud API не поддерживает батч-обработку в одном запросе,
        # поэтому делаем параллельные запросы
        embeddings = await asyncio.gather(
            *[self._get_embedding(text, text_type="doc") for text in texts]
        )

        return list(embeddings)
