import pytest
from fastapi import status


@pytest.mark.asyncio
async def test_create_and_get_namespace(client, test_user):
    """Создаём пользователя в тестовой БД, затем namespace через API, проверяем список."""
    response = await client.post(
        "/api/v1/namespaces/",
        params={"user_id": test_user.id},
        json={"name": "test_namespace", "description": "test_description"},
    )
    assert response.status_code == status.HTTP_201_CREATED, response.json()
    assert response.json()["data"]["name"] == "test_namespace", response.json()

    response = await client.get(f"/api/v1/namespaces/{response.json()['data']['id']}", params={"user_id": test_user.id})
    assert response.status_code == status.HTTP_200_OK, response.json()
    assert response.json()["data"]["id"] == response.json()["data"]["id"], response.text
    assert response.json()["data"]["name"] == "test_namespace", response.json()
    assert response.json()["data"]["description"] == "test_description", response.json()
    assert response.json()["data"]["created_at"] is not None, response.json()
    assert response.json()["data"]["files"] == [], response.json()


@pytest.mark.asyncio
async def test_list_namespaces(client, test_user):
    """Создаём несколько namespace и проверяем список."""
    response = await client.post(
        "/api/v1/namespaces/",
        params={"user_id": test_user.id},
        json={"name": "test_namespace_1", "description": "test_description_1"},
    )
    assert response.status_code == status.HTTP_201_CREATED, response.json()
    response = await client.post(
        "/api/v1/namespaces/",
        params={"user_id": test_user.id},
        json={"name": "test_namespace_2", "description": "test_description_2"},
    )
    assert response.status_code == status.HTTP_201_CREATED, response.json()

    response = await client.get(
        "/api/v1/namespaces/",
        params={"user_id": test_user.id},
    )
    assert response.status_code == status.HTTP_200_OK, response.json()
    assert len(response.json()["data"]["items"]) == 2, response.json()
    assert response.json()["data"]["items"][0]["name"] == "test_namespace_2", response.json()
    assert response.json()["data"]["items"][0]["description"] == "test_description_2", response.json()
    assert response.json()["data"]["items"][0]["created_at"] is not None, response.json()
    assert response.json()["data"]["items"][0]["files_count"] == 0, response.json()
    assert response.json()["data"]["items"][1]["name"] == "test_namespace_1", response.json()
    assert response.json()["data"]["items"][1]["description"] == "test_description_1", response.json()
    assert response.json()["data"]["items"][1]["created_at"] is not None, response.json()
    assert response.json()["data"]["items"][1]["files_count"] == 0, response.json()






@pytest.mark.asyncio
async def test_delete_namespace(client, test_user):
    """Создаём namespace и удаляем его."""
    response = await client.post(
        "/api/v1/namespaces/",
        params={"user_id": test_user.id},
        json={"name": "test_namespace", "description": "test_description"},
    )
    assert response.status_code == status.HTTP_201_CREATED, response.json()
    namespace_id = response.json()['data']['id']
    response = await client.delete(f"/api/v1/namespaces/{namespace_id}", params={"user_id": test_user.id})
    assert response.status_code == status.HTTP_204_NO_CONTENT
    response = await client.get(f"/api/v1/namespaces/{namespace_id}", params={"user_id": test_user.id})
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.json()


# --- Ошибки и граничные случаи ---


@pytest.mark.asyncio
async def test_create_namespace_duplicate_name_same_user(client, test_user):
    """Создание namespace с дубликатом имени у того же user → 400."""
    await client.post(
        "/api/v1/namespaces/",
        params={"user_id": test_user.id},
        json={"name": "duplicate_name", "description": "first"},
    )
    response = await client.post(
        "/api/v1/namespaces/",
        params={"user_id": test_user.id},
        json={"name": "duplicate_name", "description": "second"},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST, response.json()
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_get_namespace_not_found(client, test_user):
    """GET несуществующего namespace → 404."""
    response = await client.get(
        "/api/v1/namespaces/99999",
        params={"user_id": test_user.id},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.json()


@pytest.mark.asyncio
async def test_get_namespace_forbidden_other_user(client, test_user, test_user_2, test_namespace):
    """GET чужого namespace (user_id другого пользователя) → 403."""
    response = await client.get(
        f"/api/v1/namespaces/{test_namespace.id}",
        params={"user_id": test_user_2.id},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN, response.json()


@pytest.mark.asyncio
async def test_patch_namespace_not_found(client, test_user):
    """PATCH несуществующего namespace → 404."""
    response = await client.patch(
        "/api/v1/namespaces/99999",
        params={"user_id": test_user.id},
        json={"name": "new_name"},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.json()


@pytest.mark.asyncio
async def test_patch_namespace_forbidden_other_user(client, test_user, test_user_2, test_namespace):
    """PATCH чужого namespace → 403."""
    response = await client.patch(
        f"/api/v1/namespaces/{test_namespace.id}",
        params={"user_id": test_user_2.id},
        json={"name": "hacked"},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN, response.json()


@pytest.mark.asyncio
async def test_delete_namespace_not_found(client, test_user):
    """DELETE несуществующего namespace → 404."""
    response = await client.delete(
        "/api/v1/namespaces/99999",
        params={"user_id": test_user.id},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.json()


@pytest.mark.asyncio
async def test_delete_namespace_forbidden_other_user(client, test_user, test_user_2, test_namespace):
    """DELETE чужого namespace → 403."""
    response = await client.delete(
        f"/api/v1/namespaces/{test_namespace.id}",
        params={"user_id": test_user_2.id},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN, response.json()


# --- Валидация ---


@pytest.mark.asyncio
async def test_create_namespace_empty_name_422(client, test_user):
    """POST namespace с пустым именем → 422."""
    response = await client.post(
        "/api/v1/namespaces/",
        params={"user_id": test_user.id},
        json={"name": "", "description": "desc"},
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY, response.json()


@pytest.mark.asyncio
async def test_create_namespace_invalid_types_422(client, test_user):
    """POST namespace с неверными типами (name не строка) → 422."""
    response = await client.post(
        "/api/v1/namespaces/",
        params={"user_id": test_user.id},
        json={"name": 123, "description": "desc"},
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY, response.json()


# --- Пагинация ---


@pytest.mark.asyncio
async def test_list_namespaces_pagination(client, test_user):
    """GET списка namespaces с page/page_size: проверка пагинации."""
    for i in range(5):
        await client.post(
            "/api/v1/namespaces/",
            params={"user_id": test_user.id},
            json={"name": f"ns_{i}", "description": f"desc_{i}"},
        )
    # Страница 1, размер 2
    response = await client.get(
        "/api/v1/namespaces/",
        params={"user_id": test_user.id, "page": 1, "page_size": 2},
    )
    assert response.status_code == status.HTTP_200_OK, response.json()
    data = response.json()["data"]
    assert len(data["items"]) == 2, data
    assert data["pagination"]["total"] == 5, data["pagination"]
    assert data["pagination"]["page"] == 1, data["pagination"]
    assert data["pagination"]["page_size"] == 2, data["pagination"]
    assert data["pagination"]["total_pages"] == 3, data["pagination"]
    assert data["pagination"]["has_next"] is True, data["pagination"]
    assert data["pagination"]["has_previous"] is False, data["pagination"]
    # Страница 2
    response2 = await client.get(
        "/api/v1/namespaces/",
        params={"user_id": test_user.id, "page": 2, "page_size": 2},
    )
    assert response2.status_code == status.HTTP_200_OK, response2.json()
    assert len(response2.json()["data"]["items"]) == 2, response2.json()
    assert response2.json()["data"]["pagination"]["has_previous"] is True, response2.json()
