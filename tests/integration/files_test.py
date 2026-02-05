import pytest
from fastapi import status
from io import BytesIO


@pytest.mark.asyncio
async def test_upload_file(client, test_user, test_namespace):
    """Создаём файл и проверяем его список (GET namespace возвращает data.files)."""
    content = b"test file content"
    response = await client.post(
        "/api/v1/files/upload",
        params={"user_id": test_user.id, "namespace_id": test_namespace.id},
        files={"file": ("test_file.txt", BytesIO(content), "text/plain")},
    )
    assert response.status_code == status.HTTP_201_CREATED
    response = await client.get(
        f"/api/v1/namespaces/{test_namespace.id}",
        params={"user_id": test_user.id},
    )
    assert response.status_code == status.HTTP_200_OK
    files_list = response.json()["data"]["files"]
    assert len(files_list) == 1
    assert files_list[0]["filename"] == "test_file.txt"
    assert files_list[0]["file_type"] == "txt"
    assert files_list[0]["file_size"] == len(content)
    assert files_list[0]["created_at"] is not None


# --- Ошибки и граничные случаи ---


@pytest.mark.asyncio
async def test_upload_file_namespace_not_found(client, test_user):
    """Upload в несуществующий namespace_id → 404."""
    response = await client.post(
        "/api/v1/files/upload",
        params={"user_id": test_user.id, "namespace_id": 99999},
        files={"file": ("test.txt", BytesIO(b"content"), "text/plain")},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.json()


@pytest.mark.asyncio
async def test_download_file_not_found(client, test_user):
    """GET /files/download/{file_id} несуществующего файла → 404."""
    response = await client.get(
        "/api/v1/files/download/99999",
        params={"telegram_id": test_user.telegram_id},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.text


@pytest.mark.asyncio
async def test_delete_file_not_found(client, test_user):
    """DELETE несуществующего file_id → 404."""
    response = await client.delete(
        "/api/v1/files/99999",
        params={"telegram_id": test_user.telegram_id},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.text


@pytest.mark.asyncio
async def test_download_file_forbidden_other_user(client, test_user, test_user_2, test_namespace):
    """GET /download/{file_id} чужого файла (другой telegram_id) → 403."""
    content = b"secret"
    upload_resp = await client.post(
        "/api/v1/files/upload",
        params={"user_id": test_user.id, "namespace_id": test_namespace.id},
        files={"file": ("f.txt", BytesIO(content), "text/plain")},
    )
    assert upload_resp.status_code == status.HTTP_201_CREATED
    file_id = upload_resp.json()["data"]["file_id"]
    response = await client.get(
        f"/api/v1/files/download/{file_id}",
        params={"telegram_id": test_user_2.telegram_id},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN, response.text


@pytest.mark.asyncio
async def test_delete_file_forbidden_other_user(client, test_user, test_user_2, test_namespace):
    """DELETE чужого файла → 403."""
    content = b"content"
    upload_resp = await client.post(
        "/api/v1/files/upload",
        params={"user_id": test_user.id, "namespace_id": test_namespace.id},
        files={"file": ("f.txt", BytesIO(content), "text/plain")},
    )
    assert upload_resp.status_code == status.HTTP_201_CREATED
    file_id = upload_resp.json()["data"]["file_id"]
    response = await client.delete(
        f"/api/v1/files/{file_id}",
        params={"telegram_id": test_user_2.telegram_id},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN, response.text


# --- GET /structure, DELETE file, GET /download ---


@pytest.mark.asyncio
async def test_get_structure(client, test_user, test_namespace):
    """GET /watcher/structure — после user + namespace + файлов проверяем структуру."""
    content = b"structure test"
    await client.post(
        "/api/v1/files/upload",
        params={"user_id": test_user.id, "namespace_id": test_namespace.id},
        files={"file": ("struct.txt", BytesIO(content), "text/plain")},
    )
    response = await client.get(
            "/api/v1/watcher/structure",
            params={"token": test_user.watcher_token},
        )
    assert response.status_code == status.HTTP_200_OK, response.json()
    data = response.json()["data"]
    assert "namespaces" in data
    assert len(data["namespaces"]) >= 1
    ns = next((n for n in data["namespaces"] if n["id"] == test_namespace.id), None)
    assert ns is not None
    assert len(ns["files"]) == 1
    assert ns["files"][0]["filename"] == "struct.txt"


@pytest.mark.asyncio
async def test_delete_file_then_404(client, test_user, test_namespace):
    """После upload → delete → GET по file_id или список: файла нет (404 / пустой список)."""
    content = b"to delete"
    upload_resp = await client.post(
        "/api/v1/files/upload",
        params={"user_id": test_user.id, "namespace_id": test_namespace.id},
        files={"file": ("del.txt", BytesIO(content), "text/plain")},
    )
    assert upload_resp.status_code == status.HTTP_201_CREATED
    file_id = upload_resp.json()["data"]["file_id"]
    del_resp = await client.delete(
        f"/api/v1/files/{file_id}",
        params={"telegram_id": test_user.telegram_id},
    )
    assert del_resp.status_code == status.HTTP_204_NO_CONTENT
    get_resp = await client.get(
        f"/api/v1/files/download/{file_id}",
        params={"telegram_id": test_user.telegram_id},
    )
    assert get_resp.status_code == status.HTTP_404_NOT_FOUND
    ns_resp = await client.get(
        f"/api/v1/namespaces/{test_namespace.id}",
        params={"user_id": test_user.id},
    )
    assert ns_resp.status_code == status.HTTP_200_OK
    files_in_ns = ns_resp.json()["data"]["files"]
    assert len(files_in_ns) == 0


@pytest.mark.asyncio
async def test_download_file_200_headers(client, test_user, test_namespace):
    """После upload: GET /download/{file_id} — 200, ожидаемые заголовки и содержимое."""
    content = b"download test content"
    upload_resp = await client.post(
        "/api/v1/files/upload",
        params={"user_id": test_user.id, "namespace_id": test_namespace.id},
        files={"file": ("download_test.txt", BytesIO(content), "text/plain")},
    )
    assert upload_resp.status_code == status.HTTP_201_CREATED
    file_id = upload_resp.json()["data"]["file_id"]
    response = await client.get(
        f"/api/v1/files/download/{file_id}",
        params={"telegram_id": test_user.telegram_id},
    )
    assert response.status_code == status.HTTP_200_OK, response.text
    assert "content-type" in [h.lower() for h in response.headers.keys()] or "content-type" in response.headers
    assert response.content == content


# --- Валидация: upload без файла ---


@pytest.mark.asyncio
async def test_upload_without_file_422(client, test_user, test_namespace):
    """Upload без файла (обязательное поле file) → 422."""
    response = await client.post(
        "/api/v1/files/upload",
        params={"user_id": test_user.id, "namespace_id": test_namespace.id},
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY, response.json()