"""Интеграционные тесты для работы с файлами."""
import pytest
from fastapi import status
from io import BytesIO


# ---------------------------------------------------------------------------
# Загрузка файла
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_file(client, test_user, test_namespace, auth_headers):
    """Создаём файл и проверяем, что он появился в namespace."""
    content = b"test file content"
    response = await client.post(
        "/api/v1/files/upload",
        params={"namespace_id": test_namespace.id},
        files={"files": ("test_file.txt", BytesIO(content), "text/plain")},
        headers=auth_headers,
    )
    assert response.status_code == status.HTTP_201_CREATED, response.json()

    ns_resp = await client.get(
        f"/api/v1/namespaces/{test_namespace.id}",
        headers=auth_headers,
    )
    assert ns_resp.status_code == status.HTTP_200_OK
    files_list = ns_resp.json()["data"]["files"]
    assert len(files_list) == 1
    assert files_list[0]["filename"] == "test_file.txt"
    assert files_list[0]["created_at"] is not None


@pytest.mark.asyncio
async def test_upload_file_namespace_not_found(client, test_user, auth_headers):
    """Upload в несуществующий namespace_id → 404."""
    response = await client.post(
        "/api/v1/files/upload",
        params={"namespace_id": 99999},
        files={"files": ("test.txt", BytesIO(b"content"), "text/plain")},
        headers=auth_headers,
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.json()


@pytest.mark.asyncio
async def test_upload_without_file_422(client, test_user, test_namespace, auth_headers):
    """Upload без файла (ни file, ни url, ни text) → 400 или 422."""
    response = await client.post(
        "/api/v1/files/upload",
        params={"namespace_id": test_namespace.id},
        headers=auth_headers,
    )
    assert response.status_code in (
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_422_UNPROCESSABLE_ENTITY,
    ), response.json()


# ---------------------------------------------------------------------------
# Скачивание файла
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_file_not_found(client, test_user, auth_headers):
    """GET /files/download/99999 — несуществующий файл → 404."""
    response = await client.get(
        "/api/v1/files/download/99999",
        headers=auth_headers,
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.text


@pytest.mark.asyncio
async def test_download_file_200(client, test_user, test_namespace, auth_headers):
    """Загружаем файл → скачиваем → проверяем содержимое."""
    content = b"download test content"
    upload_resp = await client.post(
        "/api/v1/files/upload",
        params={"namespace_id": test_namespace.id},
        files={"files": ("download_test.txt", BytesIO(content), "text/plain")},
        headers=auth_headers,
    )
    assert upload_resp.status_code == status.HTTP_201_CREATED
    file_id = upload_resp.json()["data"][0]["file_id"]

    response = await client.get(
        f"/api/v1/files/download/{file_id}",
        headers=auth_headers,
    )
    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.content == content


# ---------------------------------------------------------------------------
# Удаление файла
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_file_not_found(client, test_user, auth_headers):
    """DELETE несуществующего file_id → 404."""
    response = await client.delete(
        "/api/v1/files/99999",
        headers=auth_headers,
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND, response.text


@pytest.mark.asyncio
async def test_delete_file_then_404(client, test_user, test_namespace, auth_headers):
    """Upload → delete → повторный GET по file_id → 404."""
    content = b"to delete"
    upload_resp = await client.post(
        "/api/v1/files/upload",
        params={"namespace_id": test_namespace.id},
        files={"files": ("del.txt", BytesIO(content), "text/plain")},
        headers=auth_headers,
    )
    assert upload_resp.status_code == status.HTTP_201_CREATED
    file_id = upload_resp.json()["data"][0]["file_id"]

    del_resp = await client.delete(f"/api/v1/files/{file_id}", headers=auth_headers)
    assert del_resp.status_code == status.HTTP_204_NO_CONTENT

    get_resp = await client.get(f"/api/v1/files/download/{file_id}", headers=auth_headers)
    assert get_resp.status_code == status.HTTP_404_NOT_FOUND

    ns_resp = await client.get(f"/api/v1/namespaces/{test_namespace.id}", headers=auth_headers)
    assert ns_resp.status_code == status.HTTP_200_OK
    assert len(ns_resp.json()["data"]["files"]) == 0


# ---------------------------------------------------------------------------
# Права доступа (другой пользователь)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_file_forbidden_other_user(
    client, test_user, test_user_2, test_namespace, auth_headers, auth_headers_2
):
    """Скачать чужой файл → 403."""
    upload_resp = await client.post(
        "/api/v1/files/upload",
        params={"namespace_id": test_namespace.id},
        files={"files": ("f.txt", BytesIO(b"secret"), "text/plain")},
        headers=auth_headers,
    )
    assert upload_resp.status_code == status.HTTP_201_CREATED
    file_id = upload_resp.json()["data"][0]["file_id"]

    response = await client.get(
        f"/api/v1/files/download/{file_id}",
        headers=auth_headers_2,
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN, response.text


@pytest.mark.asyncio
async def test_delete_file_forbidden_other_user(
    client, test_user, test_user_2, test_namespace, auth_headers, auth_headers_2
):
    """Удалить чужой файл → 403."""
    upload_resp = await client.post(
        "/api/v1/files/upload",
        params={"namespace_id": test_namespace.id},
        files={"files": ("f.txt", BytesIO(b"content"), "text/plain")},
        headers=auth_headers,
    )
    assert upload_resp.status_code == status.HTTP_201_CREATED
    file_id = upload_resp.json()["data"][0]["file_id"]

    response = await client.delete(f"/api/v1/files/{file_id}", headers=auth_headers_2)
    assert response.status_code == status.HTTP_403_FORBIDDEN, response.text

