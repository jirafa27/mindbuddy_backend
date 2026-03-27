import httpx
from typing import List, Dict, Optional

from app.core.config import settings
from app.infrastructure.llm.ollama_completion import LLMCompletionError


class OpenRouterCompletionService:
    """LLM-провайдер через OpenRouter (OpenAI-совместимый API)."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.model = model or settings.OPENROUTER_LLM_MODEL
        self.timeout = timeout if timeout is not None else settings.OPENROUTER_TIMEOUT
        self.api_key = settings.OPENROUTER_API_KEY

    def _convert_messages(self, messages: List[Dict]) -> List[Dict]:
        """Конвертирует {role, text} → {role, content} для OpenAI-формата."""
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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.BASE_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not text:
            raise LLMCompletionError("Empty response from OpenRouter")
        return text
