"""
Интеграционные тесты механизма pending_action (двухшаговое подтверждение удаления).

Сценарий:
1. Создаём тестовые данные напрямую в БД (файл, чат с pending_action).
2. Отправляем сообщение «да» через POST /ask с chat_id.
3. ChatService.ask() видит pending_action и выполняет или отменяет удаление —
   не доходя до LangGraph.

Тесты не требуют работающего MinIO: delete_file удаляет только user_files-запись.
"""
import pytest
from fastapi import status
from sqlalchemy import text, select

from app.infrastructure.db.models import File, UserFile, Chat, ChatMessage
from app.core.security import create_access_token


# ---------------------------------------------------------------------------
# Вспомогательные фикстуры
# ---------------------------------------------------------------------------


@pytest.fixture
async def file_with_chat(db_session, test_user, test_namespace):
    """
    Создаёт File + UserFile и Chat с pending_action=delete_file.
    Возвращает (user_file_id, chat_id).
    """
    # Минимальная запись файла без реального MinIO-пути
    content_file = File(
        content_hash="deadbeef" + "0" * 56,
        file_path="test/mock/test_file.txt",
        processing_status="done",
        media_metadata={"title": "Тестовый файл", "file_type": "txt", "file_size": 42},
    )
    db_session.add(content_file)
    await db_session.flush()  # получаем content_file.id

    user_file = UserFile(
        user_id=test_user.id,
        file_id=content_file.id,
        namespace_id=test_namespace.id,
        custom_title="Тестовый файл",
    )
    db_session.add(user_file)
    await db_session.flush()

    chat = Chat(
        user_id=test_user.id,
        name="Тестовый чат",
        pending_action={
            "type": "delete_file",
            "params": {"file_id": user_file.id},
            "target": "файл «Тестовый файл»",
        },
    )
    db_session.add(chat)
    await db_session.commit()
    await db_session.refresh(user_file)
    await db_session.refresh(chat)

    return user_file.id, chat.id


# ---------------------------------------------------------------------------
# Подтверждение удаления
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pending_delete_file_confirmed(client, test_user, auth_headers, file_with_chat, db_session):
    """
    Сценарий: файл существует → чат ожидает подтверждения удаления →
    пользователь пишет «да» → файл удаляется из user_files.
    """
    user_file_id, chat_id = file_with_chat

    response = await client.post(
        "/api/v1/ask",
        data={"question": "да"},
        params={"chat_id": chat_id},
        headers=auth_headers,
    )

    assert response.status_code == status.HTTP_200_OK, response.text
    answer = response.json()["data"]["answer"]
    # ChatService возвращает «удалён» или «удалено»
    assert "удал" in answer.lower(), f"Ожидали подтверждение удаления, получили: {answer}"

    # Файл должен исчезнуть из user_files
    result = await db_session.execute(
        text("SELECT COUNT(*) FROM user_files WHERE id = :id"),
        {"id": user_file_id},
    )
    count = result.scalar()
    assert count == 0, "Запись user_files должна быть удалена после подтверждения"

    # pending_action в чате должен быть сброшен
    result2 = await db_session.execute(
        text("SELECT pending_action FROM chats WHERE id = :cid"),
        {"cid": chat_id},
    )
    pending = result2.scalar()
    assert pending is None, "pending_action должен быть очищен после подтверждения"


@pytest.mark.asyncio
async def test_pending_delete_file_declined(client, test_user, auth_headers, file_with_chat, db_session):
    """
    Сценарий: чат ожидает подтверждения →
    пользователь пишет «нет» → файл остаётся нетронутым.
    """
    user_file_id, chat_id = file_with_chat

    response = await client.post(
        "/api/v1/ask",
        data={"question": "нет, не удаляй"},
        params={"chat_id": chat_id},
        headers=auth_headers,
    )

    assert response.status_code == status.HTTP_200_OK, response.text
    answer = response.json()["data"]["answer"]
    # ChatService отменяет действие
    assert len(answer) > 0

    # Файл должен остаться
    result = await db_session.execute(
        text("SELECT COUNT(*) FROM user_files WHERE id = :id"),
        {"id": user_file_id},
    )
    count = result.scalar()
    assert count == 1, "Файл не должен быть удалён при отказе"

    # pending_action очищается и при отказе
    result2 = await db_session.execute(
        text("SELECT pending_action FROM chats WHERE id = :cid"),
        {"cid": chat_id},
    )
    pending = result2.scalar()
    assert pending is None, "pending_action должен быть очищен даже при отказе"


# ---------------------------------------------------------------------------
# Чат без pending_action работает как обычно
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_pending_action_goes_to_graph(client, test_user, auth_headers, db_session):
    """
    Если у чата нет pending_action — «да» обрабатывается как обычный вопрос через граф,
    а не как подтверждение.
    """
    # Создаём чат без pending_action
    chat = Chat(user_id=test_user.id, name="Обычный чат", pending_action=None)
    db_session.add(chat)
    await db_session.commit()
    await db_session.refresh(chat)

    response = await client.post(
        "/api/v1/ask",
        data={"question": "да, всё хорошо"},
        params={"chat_id": chat.id},
        headers=auth_headers,
    )

    assert response.status_code == status.HTTP_200_OK, response.text
    # Ответ должен прийти от MindBuddyAgent («да, всё хорошо» → general_chat)
    data = response.json()["data"]
    assert data.get("answer") is not None
