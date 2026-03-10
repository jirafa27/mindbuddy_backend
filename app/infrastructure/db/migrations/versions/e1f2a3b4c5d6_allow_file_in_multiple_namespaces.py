"""allow same file in multiple namespaces

Revision ID: e1f2a3b4c5d6
Revises: d1e2f3a4b5c6
Create Date: 2026-03-06

"""
from typing import Sequence, Union

from alembic import op


revision: str = "e1f2a3b4c5d6"
down_revision: Union[str, None] = "d1e2f3a4b5c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Удаляем старое ограничение (user_id, file_id) без учёта namespace
    op.execute("ALTER TABLE user_files DROP CONSTRAINT IF EXISTS uq_user_file")

    # Уникальность для файлов без пространства (namespace_id IS NULL)
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_user_file_no_ns
        ON user_files (user_id, file_id)
        WHERE namespace_id IS NULL
        """
    )

    # Уникальность для файлов в конкретном пространстве (namespace_id IS NOT NULL)
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_user_file_with_ns
        ON user_files (user_id, file_id, namespace_id)
        WHERE namespace_id IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_user_file_no_ns")
    op.execute("DROP INDEX IF EXISTS uq_user_file_with_ns")

    # Восстанавливаем старое ограничение, предварительно удалив возможные дубли
    # (оставляем запись с наименьшим id для каждой пары user_id+file_id)
    op.execute(
        """
        DELETE FROM user_files uf1
        USING user_files uf2
        WHERE uf1.user_id = uf2.user_id
          AND uf1.file_id = uf2.file_id
          AND uf1.id > uf2.id
        """
    )
    op.execute(
        "ALTER TABLE user_files ADD CONSTRAINT uq_user_file UNIQUE (user_id, file_id)"
    )
