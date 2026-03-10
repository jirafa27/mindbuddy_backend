import httpx
from typing import List, Dict

from app.core.config import settings


class LLMCompletionError(Exception):
    pass


class OllamaCompletionService:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_LLM_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT

    def _convert_messages(self, messages: List[Dict]) -> List[Dict]:
        """Конвертирует Яндекс-формат {role, text} в OpenAI-формат {role, content}."""
        return [
            {"role": m["role"], "content": m.get("text") or m.get("content", "")}
            for m in messages
        ]

    async def complete(
        self,
        messages: List[Dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        text = data.get("message", {}).get("content", "").strip()
        if not text:
            raise LLMCompletionError("Empty response from Ollama")
        return text
