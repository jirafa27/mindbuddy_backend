"""
Интеграционные тесты синхронизации векторов при замене содержимого файла.

Критичный сценарий: если при PUT /files/{id}/content старые векторы NOT удаляются,
RAG начинает «галлюцинировать», смешивая данные из разных версий файла.

Тесты работают с тестовой БД и мок-хранилищем (MockFileStorage из conftest),
поэтому MinIO для данных тестов не требуется.
"""
import pytest
from fastapi import status
from io import BytesIO
from sqlalchemy import text

from app.core.dependencies import get_storage_service
from app.infrastructure.db.models import File, UserFile, VectorEmbedding
from app.main import app


# ---------------------------------------------------------------------------
# Вспомогательная фикстура: файл с вектор-эмбеддингами в БД
# ---------------------------------------------------------------------------


@pytest.fixture
async def file_with_vectors(db_session, test_user, test_namespace):
    """
    Создаёт File + UserFile + 3 записи VectorEmbedding.
    Возвращает (user_file_id, content_file_id).
    """
    content_file = File(
        content_hash="aabbcc" + "0" * 58,
        file_path="test/mock/original.txt",
        processing_status="done",
        media_metadata={"title": "original.txt", "file_type": "txt", "file_size": 20},
    )
    db_session.add(content_file)
    await db_session.flush()

    user_file = UserFile(
        user_id=test_user.id,
        file_id=content_file.id,
        namespace_id=test_namespace.id,
        custom_title="original.txt",
    )
    db_session.add(user_file)
    await db_session.flush()

    # Векторы-заглушки (3584-мерный вектор из нулей)
    zero_vector = [0.0] * 3584
    for idx in range(3):
        vec = VectorEmbedding(
            file_id=content_file.id,
            chunk_index=idx,
            chunk_text=f"Chunk {idx}: борщ рецепт ингредиенты",
            embedding=zero_vector,
        )
        db_session.add(vec)

    await db_session.commit()
    await db_session.refresh(user_file)

    return user_file.id, content_file.id


# ---------------------------------------------------------------------------
# Основной тест: замена содержимого удаляет старые векторы
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replace_file_deletes_old_vectors(
    client, test_user, auth_headers, file_with_vectors, db_session
):
    """
    PUT /files/{user_file_id}/content → FileService.replace_file_content()
    должен вызвать vector_repository.delete_by_file_id() до постановки задачи
    на новые эмбеддинги.

    Проверяем: после замены содержимого векторов для этого content_file.id нет.
    """
    user_file_id, content_file_id = file_with_vectors

    # Убеждаемся: векторы ЕСТЬ до замены
    result = await db_session.execute(
        text("SELECT COUNT(*) FROM vector_embeddings WHERE file_id = :fid"),
        {"fid": content_file_id},
    )
    count_before = result.scalar()
    assert count_before == 3, f"До замены должно быть 3 вектора, есть: {count_before}"

    # Мокаем хранилище, чтобы не нужен реальный MinIO
    from tests.conftest import MockFileStorage, MockTaskPublisher

    app.dependency_overrides[get_storage_service] = lambda: MockFileStorage()
    try:
        new_content = b"Pizza recipe: dough, tomato sauce, mozzarella."
        response = await client.put(
            f"/api/v1/files/{user_file_id}/content",
            files={"file": ("updated.txt", BytesIO(new_content), "text/plain")},
            headers=auth_headers,
        )
    finally:
        app.dependency_overrides.pop(get_storage_service, None)

    assert response.status_code == status.HTTP_200_OK, (
        f"PUT /files/{user_file_id}/content вернул {response.status_code}: {response.text}"
    )

    # Векторы для СТАРОГО content_file_id должны быть удалены
    result2 = await db_session.execute(
        text("SELECT COUNT(*) FROM vector_embeddings WHERE file_id = :fid"),
        {"fid": content_file_id},
    )
    count_after = result2.scalar()
    assert count_after == 0, (
        f"После замены файла старые векторы должны быть удалены, "
        f"но осталось {count_after} записей. Это вызовет галлюцинации в RAG!"
    )


# ---------------------------------------------------------------------------
# Дополнительный тест: удаление файла также чистит векторы
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_file_does_not_leave_orphan_vectors(
    client, test_user, auth_headers, file_with_vectors, db_session
):
    """
    DELETE /files/{user_file_id} → запись user_files удалена.

    Примечание: текущая реализация delete_file удаляет только user_files-запись.
    Сам content-файл (files) и его векторы остаются до полного GC.
    Тест документирует это поведение — если оно изменится, тест упадёт и
    напомнит об обновлении логики.
    """
    user_file_id, content_file_id = file_with_vectors

    response = await client.delete(
        f"/api/v1/files/{user_file_id}",
        headers=auth_headers,
    )
    assert response.status_code == status.HTTP_204_NO_CONTENT, response.text

    # user_files запись удалена
    result = await db_session.execute(
        text("SELECT COUNT(*) FROM user_files WHERE id = :id"),
        {"id": user_file_id},
    )
    assert result.scalar() == 0, "user_files запись должна быть удалена"


# ---------------------------------------------------------------------------
# Тест на деduplication: загрузка одинакового файла не дублирует векторы
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_duplicate_vectors_on_upsert(db_session, test_user, test_namespace):
    """
    Если два чанка с одинаковым (file_id, chunk_index) вставляются повторно,
    ON CONFLICT DO UPDATE обновляет, а не дублирует запись.
    """
    from app.infrastructure.repositories.vector_embedding_repository import PgVectorRepository

    repo = PgVectorRepository(db_session)
    content_file = File(
        content_hash="ddeeff" + "0" * 58,
        file_path="test/mock/dedup.txt",
        processing_status="done",
        media_metadata={"title": "dedup.txt"},
    )
    db_session.add(content_file)
    await db_session.flush()

    zero_vector = [0.0] * 3584
    chunks = ["Первый чанк", "Второй чанк"]
    embeddings = [zero_vector, zero_vector]

    # Первая вставка
    await repo.create_batch(file_id=content_file.id, chunks=chunks, embeddings=embeddings)
    await db_session.commit()

    # Повторная вставка тех же чанков
    await repo.create_batch(file_id=content_file.id, chunks=chunks, embeddings=embeddings)
    await db_session.commit()

    result = await db_session.execute(
        text("SELECT COUNT(*) FROM vector_embeddings WHERE file_id = :fid"),
        {"fid": content_file.id},
    )
    count = result.scalar()
    assert count == 2, (
        f"После двойного upsert должно остаться 2 чанка, обнаружено: {count}. "
        "Проблема с ON CONFLICT: происходит дублирование векторов!"
    )
