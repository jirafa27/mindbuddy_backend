import pytest
from fastapi import status


@pytest.mark.asyncio
async def test_create_user(client):
    """POST /users/ — создание пользователя, проверка формата и 201."""
    response = await client.post(
        "/api/v1/users/",
        json={
            "telegram_id": 999,
            "username": "newuser",
            "full_name": "New User",
        },
    )
    assert response.status_code == status.HTTP_201_CREATED, response.json()
    data = response.json()["data"]
    assert data["telegram_id"] == 999
    assert data["username"] == "newuser"
    assert data["full_name"] == "New User"
    assert data["id"] is not None
    assert data["created_at"] is not None
    assert data["updated_at"] is not None
    assert "watcher_token" in data


@pytest.mark.asyncio
async def test_get_user_by_id(client, test_user):
    """GET /users/{id} — получение пользователя по id, 200."""
    response = await client.get(f"/api/v1/users/{test_user.id}")
    assert response.status_code == status.HTTP_200_OK, response.json()
    data = response.json()["data"]
    assert data["id"] == test_user.id
    assert data["telegram_id"] == test_user.telegram_id
    assert data["username"] == test_user.username
    assert data["full_name"] == test_user.full_name


@pytest.mark.asyncio
async def test_get_user_by_telegram_id(client, test_user):
    """GET /users/telegram/{telegram_id} — получение по Telegram ID, 200."""
    response = await client.get(f"/api/v1/users/telegram/{test_user.telegram_id}")
    assert response.status_code == status.HTTP_200_OK, response.json()
    data = response.json()["data"]
    assert data["telegram_id"] == test_user.telegram_id
    assert data["id"] == test_user.id


@pytest.mark.asyncio
async def test_get_user_not_found(client):
    """GET /users/{id} по несуществующему user_id → 404."""
    response = await client.get("/api/v1/users/99999")
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.json()
    assert "detail" in response.json()
