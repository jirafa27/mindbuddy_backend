"""Интеграционные тесты для эндпоинта POST /api/v1/ask."""
import pytest
from fastapi import status
from io import BytesIO


@pytest.mark.asyncio
async def test_ask_success(client, test_user):
    """POST /ask с файлом и вопросом — 200, в ответе есть answer или data. Без привязки к namespace."""
    content = b"MindBuddy is a personal knowledge assistant. It helps organize documents."
    response = await client.post(
        "/api/v1/ask",
        params={
            "question": "What is MindBuddy?",
            "telegram_id": test_user.telegram_id,
        },
        files={"file": ("doc.txt", BytesIO(content), "text/plain")},
    )
    assert response.status_code == status.HTTP_200_OK, response.text
    body = response.json()
    assert "status" in body
    # answer может быть в data или в answer в зависимости от схемы ответа
    assert "answer" in body or "data" in body, body


@pytest.mark.asyncio
async def test_ask_without_file_200(client, test_user):
    """POST /ask только с вопросом (без файла) — 200."""
    response = await client.post(
        "/api/v1/ask",
        params={"question": "What is MindBuddy?", "telegram_id": test_user.telegram_id},
    )
    assert response.status_code == status.HTTP_200_OK, response.text
    body = response.json()
    assert "answer" in body or "data" in body, body


@pytest.mark.asyncio
async def test_ask_without_question_422(client, test_user):
    """POST /ask без question — 422."""
    response = await client.post(
        "/api/v1/ask",
        params={"telegram_id": test_user.telegram_id},
        files={"file": ("f.txt", BytesIO(b"content"), "text/plain")},
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY, response.json()


@pytest.mark.asyncio
async def test_ask_unknown_telegram_id_404(client):
    """POST /ask с несуществующим telegram_id — 404."""
    response = await client.post(
        "/api/v1/ask",
        params={"question": "What?", "telegram_id": 999999},
        files={"file": ("f.txt", BytesIO(b"content"), "text/plain")},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.json()
