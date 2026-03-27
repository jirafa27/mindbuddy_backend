"""add namespace_id to chat_messages

Revision ID: g1h2i3j4k5l6
Revises: a2b3c4d5e6f7
Create Date: 2026-03-25

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "g1h2i3j4k5l6"
down_revision: Union[str, None] = "a2b3c4d5e6f7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chat_messages",
        sa.Column(
            "namespace_id",
            sa.Integer(),
            sa.ForeignKey("namespaces.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_chat_messages_namespace_id", "chat_messages", ["namespace_id"])


def downgrade() -> None:
    op.drop_index("ix_chat_messages_namespace_id", table_name="chat_messages")
    op.drop_column("chat_messages", "namespace_id")
