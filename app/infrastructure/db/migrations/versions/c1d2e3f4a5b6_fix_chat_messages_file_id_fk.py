"""fix chat_messages file_id fk to user_files

Revision ID: c1d2e3f4a5b6
Revises: b3c4d5e6f7a8
Create Date: 2026-03-05

"""
from typing import Sequence, Union

from alembic import op


revision: str = "c1d2e3f4a5b6"
down_revision: Union[str, None] = "b3c4d5e6f7a8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE chat_messages DROP CONSTRAINT IF EXISTS chat_messages_file_id_fkey"
    )
    op.execute(
        "ALTER TABLE chat_messages "
        "ADD CONSTRAINT chat_messages_file_id_fkey "
        "FOREIGN KEY (file_id) REFERENCES user_files(id) ON DELETE SET NULL"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE chat_messages DROP CONSTRAINT IF EXISTS chat_messages_file_id_fkey"
    )
    op.execute(
        "ALTER TABLE chat_messages "
        "ADD CONSTRAINT chat_messages_file_id_fkey "
        "FOREIGN KEY (file_id) REFERENCES files(id)"
    )
