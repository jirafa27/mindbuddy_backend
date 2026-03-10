"""drop namespace_id from chats, add name to chats

Revision ID: b3c4d5e6f7a8
Revises: 65b70780956d
Create Date: 2026-02-28

"""
from typing import Sequence, Union

from alembic import op


revision: str = "b3c4d5e6f7a8"
down_revision: Union[str, None] = "65b70780956d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Удаление namespace_id (если есть)
    op.execute("DROP INDEX IF EXISTS ix_chats_namespace_id")
    op.execute("ALTER TABLE chats DROP COLUMN IF EXISTS namespace_id")
    # Добавление имени чата (IF NOT EXISTS для идемпотентности)
    op.execute(
        "ALTER TABLE chats ADD COLUMN IF NOT EXISTS name VARCHAR(255) NULL"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE chats DROP COLUMN IF EXISTS name")
    op.execute(
        "ALTER TABLE chats ADD COLUMN IF NOT EXISTS namespace_id INTEGER NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_chats_namespace_id ON chats (namespace_id)"
    )
