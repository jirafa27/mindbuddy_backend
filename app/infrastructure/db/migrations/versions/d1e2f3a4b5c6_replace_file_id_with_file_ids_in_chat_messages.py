"""replace file_id with file_ids in chat_messages

Revision ID: d1e2f3a4b5c6
Revises: c1d2e3f4a5b6
Create Date: 2026-03-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d1e2f3a4b5c6"
down_revision: Union[str, None] = "c1d2e3f4a5b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chat_messages",
        sa.Column("file_ids", sa.JSON(), nullable=True),
    )
    op.execute(
        """
        UPDATE chat_messages
        SET file_ids = CASE
            WHEN file_id IS NOT NULL THEN json_build_array(file_id)
            ELSE '[]'::json
        END
        """
    )
    op.alter_column("chat_messages", "file_ids", nullable=False)

    op.execute(
        "ALTER TABLE chat_messages DROP CONSTRAINT IF EXISTS chat_messages_file_id_fkey"
    )
    op.drop_index("ix_chat_messages_file_id", table_name="chat_messages")
    op.drop_column("chat_messages", "file_id")


def downgrade() -> None:
    op.add_column(
        "chat_messages",
        sa.Column("file_id", sa.Integer(), nullable=True),
    )
    op.execute(
        """
        UPDATE chat_messages
        SET file_id = CASE
            WHEN json_array_length(file_ids) > 0
            THEN (file_ids->0)::int
            ELSE NULL
        END
        """
    )
    op.create_index("ix_chat_messages_file_id", "chat_messages", ["file_id"])
    op.execute(
        "ALTER TABLE chat_messages "
        "ADD CONSTRAINT chat_messages_file_id_fkey "
        "FOREIGN KEY (file_id) REFERENCES user_files(id) ON DELETE SET NULL"
    )
    op.drop_column("chat_messages", "file_ids")
