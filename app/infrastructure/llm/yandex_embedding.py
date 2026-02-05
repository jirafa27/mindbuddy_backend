import asyncio
import httpx
from typing import List
from app.core.config import settings
from app.core.exceptions import EmbeddingGenerationError
from app.infrastructure.llm.yandex_iam import YandexIAMService


class YandexEmbeddingService:
    def __init__(self, iam_service: YandexIAMService):
        self.iam_service = iam_service
        self.folder_id = settings.YANDEX_FOLDER_ID
        self.embed_url = settings.YANDEX_EMBED_URL
        self.doc_uri = f"emb://{self.folder_id}/text-search-doc/latest"
        self.query_uri = f"emb://{self.folder_id}/text-search-query/latest"
        self.timeout = settings.YANDEX_EMBED_TIMEOUT

    async def _get_headers(self, force_refresh: bool = False) -> dict:
        iam_token = await self.iam_service.get_iam_token(force_refresh=force_refresh)
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {iam_token}",
            "x-folder-id": self.folder_id,
        }

    async def _get_embedding(self, text: str, text_type: str = "doc") -> List[float]:
        if not self.folder_id:
            raise EmbeddingGenerationError("Yandex folder ID not configured")
        if not text or not text.strip():
            raise EmbeddingGenerationError("Text cannot be empty")
        model_uri = self.doc_uri if text_type == "doc" else self.query_uri

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            headers = await self._get_headers()
            response = await client.post(
                self.embed_url,
                headers=headers,
                json={"modelUri": model_uri, "text": text},
            )
            if response.status_code == 401:
                headers = await self._get_headers(force_refresh=True)
                response = await client.post(
                    self.embed_url,
                    headers=headers,
                    json={"modelUri": model_uri, "text": text},
                )
            response.raise_for_status()
            data = response.json()
            if "embedding" in data:
                return data["embedding"]
            raise EmbeddingGenerationError("Invalid response format from Yandex Cloud API")

    async def generate_embedding(self, text: str) -> List[float]:
        return await self._get_embedding(text, text_type="doc")

    async def generate_query_embedding(self, text: str) -> List[float]:
        return await self._get_embedding(text, text_type="query")

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = await asyncio.gather(
            *[self._get_embedding(text, text_type="doc") for text in texts]
        )
        return list(embeddings)
