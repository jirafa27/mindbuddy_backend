import httpx
from typing import List, Dict

from app.core.config import settings
from app.infrastructure.llm.yandex_iam import YandexIAMService


class LLMCompletionError(Exception):
    pass


class YandexCompletionService:
    def __init__(self, iam_service: YandexIAMService):
        self.iam_service = iam_service
        self.folder_id = settings.YANDEX_FOLDER_ID
        self.completion_url = settings.YANDEX_COMPLETION_URL
        self.timeout = settings.YANDEX_COMPLETION_TIMEOUT
        self.model_uri = f"gpt://{self.folder_id}/yandexgpt-lite/latest" if self.folder_id else ""

    async def _get_headers(self, force_refresh: bool = False) -> dict:
        iam_token = await self.iam_service.get_iam_token(force_refresh=force_refresh)
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {iam_token}",
            "x-folder-id": self.folder_id or "",
        }

    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        if not self.folder_id:
            raise LLMCompletionError("Yandex folder ID not configured")

        payload = {
            "modelUri": self.model_uri,
            "messages": messages,
            "completionOptions": {
                "temperature": temperature,
                "maxTokens": str(max_tokens),
                "stream": False,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            headers = await self._get_headers()
            response = await client.post(
                self.completion_url,
                headers=headers,
                json=payload,
            )
            if response.status_code == 401:
                headers = await self._get_headers(force_refresh=True)
                response = await client.post(
                    self.completion_url,
                    headers=headers,
                    json=payload,
                )
            response.raise_for_status()
            data = response.json()

        result = data.get("result")
        if not result:
            raise LLMCompletionError("Empty result from Yandex GPT API")
        alternatives = result.get("alternatives", [])
        if not alternatives:
            raise LLMCompletionError("No alternatives in Yandex GPT response")
        message = alternatives[0].get("message")
        if not message:
            raise LLMCompletionError("No message in first alternative")
        return message.get("text", "").strip()
