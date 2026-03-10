"""add pending_action to chats

Revision ID: a1b2c3d4e5f6
Revises: d1e2f3a4b5c6
Create Date: 2026-03-06

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE chats ADD COLUMN IF NOT EXISTS pending_action JSON NULL"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE chats DROP COLUMN IF EXISTS pending_action")
