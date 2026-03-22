"""Интеграционные тесты для регистрации и работы с пользователями."""
import pytest
from fastapi import status


@pytest.mark.asyncio
async def test_register_user(client):
    """POST /api/v1/users/ — регистрация по email/паролю: 201, токен в ответе."""
    response = await client.post(
        "/api/v1/users/",
        json={
            "email": "newuser@example.com",
            "password": "securepassword123",
            "full_name": "New User",
        },
    )
    assert response.status_code == status.HTTP_201_CREATED, response.json()
    data = response.json()["data"]
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "newuser@example.com"
    assert data["user"]["full_name"] == "New User"
    assert data["user"]["id"] is not None


@pytest.mark.asyncio
async def test_register_duplicate_email(client):
    """POST /api/v1/users/ с уже существующим email → 400."""
    payload = {"email": "dup@example.com", "password": "pass123", "full_name": "Dup"}
    await client.post("/api/v1/users/", json=payload)
    response = await client.post("/api/v1/users/", json=payload)
    assert response.status_code == status.HTTP_400_BAD_REQUEST, response.json()


@pytest.mark.asyncio
async def test_get_me(client, test_user, auth_headers):
    """GET /api/v1/users/me — текущий пользователь по JWT → 200."""
    response = await client.get("/api/v1/users/me", headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK, response.json()
    data = response.json()["data"]
    assert data["id"] == test_user.id
    assert data["email"] == test_user.email


@pytest.mark.asyncio
async def test_get_me_no_token(client):
    """GET /api/v1/users/me без токена → 403."""
    response = await client.get("/api/v1/users/me")
    assert response.status_code == status.HTTP_403_FORBIDDEN, response.json()


@pytest.mark.asyncio
async def test_get_user_by_id(client, test_user):
    """GET /api/v1/users/{id} — получение по ID, без авторизации → 200."""
    response = await client.get(f"/api/v1/users/{test_user.id}")
    assert response.status_code == status.HTTP_200_OK, response.json()
    data = response.json()["data"]
    assert data["id"] == test_user.id
    assert data["email"] == test_user.email


@pytest.mark.asyncio
async def test_get_user_not_found(client):
    """GET /api/v1/users/99999 — несуществующий ID → 404."""
    response = await client.get("/api/v1/users/99999")
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.json()
    assert "detail" in response.json()
