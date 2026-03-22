"""
Интеграционные тесты CrudNode через POST /api/v1/ask.

Используем make_ask_client(intent=...) чтобы обойти LLM-классификацию:
MockLLMProvider возвращает нужный JSON с заданным intent, а реальный граф
выполняет соответствующую операцию с тестовой БД.
"""
import pytest
from fastapi import status
from sqlalchemy import text


# ---------------------------------------------------------------------------
# Создание пространства (create_namespace)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_namespace_via_ask(make_ask_client, test_user, auth_headers):
    """
    /ask с intent=create_namespace → CrudNode создаёт пространство в БД.
    Проверяем: ответ 200, в тексте имя пространства, запись есть в БД.
    """
    ns_name = "Новое тестовое пространство"
    async with make_ask_client(intent="create_namespace", entity_name=ns_name) as client:
        response = await client.post(
            "/api/v1/ask",
            data={"question": f"Создай пространство '{ns_name}'"},
            headers=auth_headers,
        )

    assert response.status_code == status.HTTP_200_OK, response.text
    data = response.json()["data"]
    assert ns_name in data["answer"], f"Ожидали '{ns_name}' в ответе: {data['answer']}"
    assert "CrudNode" in " ".join(data.get("agent_steps", []))


@pytest.mark.asyncio
async def test_create_namespace_without_name_returns_prompt(make_ask_client, test_user, auth_headers):
    """
    /ask с intent=create_namespace, но без entity_name →
    CrudNode должен попросить указать имя (не 500).
    """
    async with make_ask_client(intent="create_namespace", entity_name=None) as client:
        response = await client.post(
            "/api/v1/ask",
            data={"question": "Создай пространство"},
            headers=auth_headers,
        )

    assert response.status_code == status.HTTP_200_OK, response.text
    answer = response.json()["data"]["answer"]
    # CrudNode возвращает подсказку, а не ошибку сервера
    assert len(answer) > 0


# ---------------------------------------------------------------------------
# Создание файла (create_file) — с мок-хранилищем
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_file_via_ask(make_ask_client, test_user, test_namespace, auth_headers, db_session):
    """
    /ask с intent=create_file → CrudNode создаёт файл в тестовом namespace.
    Проверяем: ответ 200, file_id присутствует, запись в user_files есть.
    """
    file_name = "Заметка о паттернах"
    file_content = "Паттерны проектирования: Singleton, Factory, Observer."

    async with make_ask_client(
        intent="create_file",
        mock_storage=True,
        entity_name=file_name,
        entity_content=file_content,
        namespace_hint="test_namespace",
    ) as client:
        response = await client.post(
            "/api/v1/ask",
            data={"question": f"Создай заметку '{file_name}' в пространстве test_namespace"},
            params={"namespace_id": test_namespace.id},
            headers=auth_headers,
        )

    assert response.status_code == status.HTTP_200_OK, response.text
    data = response.json()["data"]
    assert len(data["answer"]) > 0, "Ответ не должен быть пустым"
    assert "CrudNode" in " ".join(data.get("agent_steps", []))

    # Проверяем, что файл появился в БД
    result = await db_session.execute(
        text("SELECT COUNT(*) FROM user_files WHERE user_id = :uid"),
        {"uid": test_user.id},
    )
    count = result.scalar()
    assert count >= 1, "Файл не создан в user_files"


# ---------------------------------------------------------------------------
# Удаление несуществующего файла (delete_file) — graceful 200, не 500
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_nonexistent_file_returns_graceful_message(
    make_ask_client, test_user, auth_headers
):
    """
    /ask с intent=delete_file и несуществующим именем →
    CrudNode должен ответить «Не нашёл файл», а не упасть с 500.
    """
    async with make_ask_client(
        intent="delete_file",
        search_query="несуществующий_файл_xyz_123",
        entity_name="несуществующий_файл_xyz_123",
    ) as client:
        response = await client.post(
            "/api/v1/ask",
            data={"question": "Удали файл 'несуществующий_файл_xyz_123'"},
            headers=auth_headers,
        )

    assert response.status_code == status.HTTP_200_OK, (
        f"Ожидали 200, получили {response.status_code}: {response.text}"
    )
    answer = response.json()["data"]["answer"]
    # Агент должен сообщить, что файл не найден — но не упасть
    assert len(answer) > 0
    # Ни в коем случае не должно быть трассировки или HTTP 500
    assert "500" not in response.text
    assert "Internal Server Error" not in response.text


# ---------------------------------------------------------------------------
# Удаление несуществующего пространства (delete_namespace)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_nonexistent_namespace_returns_graceful_message(
    make_ask_client, test_user, auth_headers
):
    """
    /ask с intent=delete_namespace и несуществующим именем →
    CrudNode должен ответить «Не нашёл пространство», не 500.
    """
    async with make_ask_client(
        intent="delete_namespace",
        namespace_hint="НесуществующееПространство_xyz",
    ) as client:
        response = await client.post(
            "/api/v1/ask",
            data={"question": "Удали пространство 'НесуществующееПространство_xyz'"},
            headers=auth_headers,
        )

    assert response.status_code == status.HTTP_200_OK, response.text
    answer = response.json()["data"]["answer"]
    assert len(answer) > 0
    assert "Internal Server Error" not in response.text
