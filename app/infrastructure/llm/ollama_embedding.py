import asyncio
import httpx
from typing import List

from app.core.config import settings
from app.core.exceptions import EmbeddingGenerationError

EMBEDDING_CONCURRENCY = 5


class OllamaEmbeddingService:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_EMBED_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT
        self._semaphore = asyncio.Semaphore(EMBEDDING_CONCURRENCY)

    async def _get_embedding(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise EmbeddingGenerationError("Text cannot be empty")

        async with self._semaphore:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": text},
                )
                response.raise_for_status()
                data = response.json()

        embeddings = data.get("embeddings")
        if not embeddings or not embeddings[0]:
            raise EmbeddingGenerationError("Empty embedding response from Ollama")
        return embeddings[0]

    async def generate_embedding(self, text: str) -> List[float]:
        return await self._get_embedding(text)

    async def generate_query_embedding(self, text: str) -> List[float]:
        return await self._get_embedding(text)

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = await asyncio.gather(
            *[self._get_embedding(text) for text in texts]
        )
        return list(embeddings)
