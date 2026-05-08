"""Интеграционные тесты для эндпоинта POST /api/v1/ask."""
import pytest
from fastapi import status
from io import BytesIO


@pytest.mark.asyncio
async def test_ask_text_question_200(client, test_user, auth_headers):
    """POST /ask с текстовым вопросом (без файла) — 200, answer присутствует."""
    response = await client.post(
        "/api/v1/ask",
        data={"question": "Привет, как дела?"},
        headers=auth_headers,
    )
    assert response.status_code == status.HTTP_200_OK, response.text
    body = response.json()
    assert "data" in body
    assert "answer" in body["data"]


@pytest.mark.asyncio
async def test_ask_with_file_200(client, test_user, auth_headers):
    """POST /ask с прикреплённым файлом и вопросом — 200."""
    content = b"MindBuddy is a personal knowledge assistant."
    response = await client.post(
        "/api/v1/ask",
        data={"question": "Сохрани этот файл"},
        files={"files": ("doc.txt", BytesIO(content), "text/plain")},
        headers=auth_headers,
    )
    assert response.status_code == status.HTTP_200_OK, response.text
    body = response.json()
    assert "data" in body


@pytest.mark.asyncio
async def test_ask_without_question_422(client, test_user, auth_headers):
    """POST /ask без поля question (обязательное Form-поле) — 422."""
    response = await client.post(
        "/api/v1/ask",
        files={"files": ("f.txt", BytesIO(b"content"), "text/plain")},
        headers=auth_headers,
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY, response.json()


@pytest.mark.asyncio
async def test_ask_no_auth_403(client):
    """POST /ask без JWT-токена — 401."""
    response = await client.post(
        "/api/v1/ask",
        data={"question": "Привет"},
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED, response.json()


@pytest.mark.asyncio
async def test_ask_creates_new_chat(client, test_user, auth_headers):
    """POST /ask без chat_id должен создавать новый чат и возвращать chat_id."""
    response = await client.post(
        "/api/v1/ask",
        data={"question": "Тестовый вопрос"},
        headers=auth_headers,
    )
    assert response.status_code == status.HTTP_200_OK, response.text
    data = response.json()["data"]
    assert data["chat_id"] is not None


@pytest.mark.asyncio
async def test_ask_continues_existing_chat(client, test_user, auth_headers):
    """POST /ask с chat_id продолжает существующий чат."""
    first_resp = await client.post(
        "/api/v1/ask",
        data={"question": "Первый вопрос"},
        headers=auth_headers,
    )
    assert first_resp.status_code == status.HTTP_200_OK
    chat_id = first_resp.json()["data"]["chat_id"]
    assert chat_id is not None

    second_resp = await client.post(
        "/api/v1/ask",
        data={"question": "Второй вопрос"},
        params={"chat_id": chat_id},
        headers=auth_headers,
    )
    assert second_resp.status_code == status.HTTP_200_OK, second_resp.text
    assert second_resp.json()["data"]["chat_id"] == chat_id
