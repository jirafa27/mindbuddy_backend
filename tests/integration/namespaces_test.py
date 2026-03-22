"""Интеграционные тесты для управления пространствами знаний (namespaces)."""
import pytest
from fastapi import status


# ---------------------------------------------------------------------------
# Базовые CRUD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_and_get_namespace(client, test_user, auth_headers):
    """Создаём namespace через API и проверяем его содержимое."""
    response = await client.post(
        "/api/v1/namespaces/",
        json={"name": "test_namespace", "description": "test_description"},
        headers=auth_headers,
    )
    assert response.status_code == status.HTTP_201_CREATED, response.json()
    ns_id = response.json()["data"]["id"]
    assert response.json()["data"]["name"] == "test_namespace"

    response = await client.get(
        f"/api/v1/namespaces/{ns_id}",
        headers=auth_headers,
    )
    assert response.status_code == status.HTTP_200_OK, response.json()
    data = response.json()["data"]
    assert data["id"] == ns_id
    assert data["name"] == "test_namespace"
    assert data["description"] == "test_description"
    assert data["created_at"] is not None
    assert data["files"] == []


@pytest.mark.asyncio
async def test_list_namespaces(client, test_user, auth_headers):
    """Создаём несколько namespace и проверяем список."""
    for name in ("ns_alpha", "ns_beta"):
        r = await client.post(
            "/api/v1/namespaces/",
            json={"name": name, "description": f"desc_{name}"},
            headers=auth_headers,
        )
        assert r.status_code == status.HTTP_201_CREATED

    response = await client.get("/api/v1/namespaces/", headers=auth_headers)
    assert response.status_code == status.HTTP_200_OK, response.json()
    items = response.json()["data"]["items"]
    names = [item["name"] for item in items]
    assert "ns_alpha" in names
    assert "ns_beta" in names


@pytest.mark.asyncio
async def test_delete_namespace(client, test_user, auth_headers):
    """Создаём namespace и удаляем его — повторный GET → 404."""
    r = await client.post(
        "/api/v1/namespaces/",
        json={"name": "to_delete", "description": ""},
        headers=auth_headers,
    )
    assert r.status_code == status.HTTP_201_CREATED
    ns_id = r.json()["data"]["id"]

    del_r = await client.delete(f"/api/v1/namespaces/{ns_id}", headers=auth_headers)
    assert del_r.status_code == status.HTTP_204_NO_CONTENT

    get_r = await client.get(f"/api/v1/namespaces/{ns_id}", headers=auth_headers)
    assert get_r.status_code == status.HTTP_404_NOT_FOUND, get_r.json()


# ---------------------------------------------------------------------------
# Ошибки и права доступа
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_namespace_duplicate_name_same_user(client, test_user, auth_headers):
    """Дубликат имени namespace у одного пользователя → 400."""
    payload = {"name": "duplicate_name", "description": "first"}
    await client.post("/api/v1/namespaces/", json=payload, headers=auth_headers)
    response = await client.post("/api/v1/namespaces/", json=payload, headers=auth_headers)
    assert response.status_code == status.HTTP_400_BAD_REQUEST, response.json()
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_get_namespace_not_found(client, test_user, auth_headers):
    """GET несуществующего namespace → 404."""
    response = await client.get("/api/v1/namespaces/99999", headers=auth_headers)
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.json()


@pytest.mark.asyncio
async def test_get_namespace_forbidden_other_user(client, test_user_2, auth_headers_2, test_namespace):
    """GET чужого namespace другим пользователем → 403."""
    response = await client.get(
        f"/api/v1/namespaces/{test_namespace.id}",
        headers=auth_headers_2,
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN, response.json()


@pytest.mark.asyncio
async def test_patch_namespace_not_found(client, test_user, auth_headers):
    """PATCH несуществующего namespace → 404."""
    response = await client.patch(
        "/api/v1/namespaces/99999",
        json={"name": "new_name"},
        headers=auth_headers,
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.json()


@pytest.mark.asyncio
async def test_patch_namespace_forbidden_other_user(client, test_user_2, auth_headers_2, test_namespace):
    """PATCH чужого namespace → 403."""
    response = await client.patch(
        f"/api/v1/namespaces/{test_namespace.id}",
        json={"name": "hacked"},
        headers=auth_headers_2,
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN, response.json()


@pytest.mark.asyncio
async def test_delete_namespace_not_found(client, test_user, auth_headers):
    """DELETE несуществующего namespace → 404."""
    response = await client.delete("/api/v1/namespaces/99999", headers=auth_headers)
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.json()


@pytest.mark.asyncio
async def test_delete_namespace_forbidden_other_user(client, test_user_2, auth_headers_2, test_namespace):
    """DELETE чужого namespace → 403."""
    response = await client.delete(
        f"/api/v1/namespaces/{test_namespace.id}",
        headers=auth_headers_2,
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN, response.json()


# ---------------------------------------------------------------------------
# Валидация
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_namespace_empty_name_422(client, test_user, auth_headers):
    """Пустое название → 422."""
    response = await client.post(
        "/api/v1/namespaces/",
        json={"name": "", "description": "desc"},
        headers=auth_headers,
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY, response.json()


# ---------------------------------------------------------------------------
# Пагинация
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_namespaces_pagination(client, test_user, auth_headers):
    """GET списка namespaces с page/page_size: проверка пагинации."""
    for i in range(5):
        await client.post(
            "/api/v1/namespaces/",
            json={"name": f"ns_{i}", "description": f"desc_{i}"},
            headers=auth_headers,
        )
    response = await client.get(
        "/api/v1/namespaces/",
        params={"page": 1, "page_size": 2},
        headers=auth_headers,
    )
    assert response.status_code == status.HTTP_200_OK, response.json()
    data = response.json()["data"]
    assert len(data["items"]) == 2, data
    assert data["pagination"]["total"] == 5, data["pagination"]
    assert data["pagination"]["has_next"] is True
    assert data["pagination"]["has_previous"] is False

    response2 = await client.get(
        "/api/v1/namespaces/",
        params={"page": 2, "page_size": 2},
        headers=auth_headers,
    )
    assert response2.status_code == status.HTTP_200_OK
    assert len(response2.json()["data"]["items"]) == 2
    assert response2.json()["data"]["pagination"]["has_previous"] is True
