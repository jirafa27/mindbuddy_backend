"""add trash namespace for existing users

Revision ID: l1m2n3o4p5q6
Revises: k2l3m4n5o6p7
Create Date: 2026-04-05
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "l1m2n3o4p5q6"
down_revision: Union[str, None] = "k2l3m4n5o6p7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(inspector: sa.Inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_column(inspector: sa.Inspector, table_name: str, column_name: str) -> bool:
    return any(col["name"] == column_name for col in inspector.get_columns(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "users") or not _has_table(inspector, "namespaces"):
        return
    if not _has_column(inspector, "namespaces", "kind"):
        return

    bind.execute(
        sa.text(
            """
            UPDATE namespaces
            SET kind = 'trash'
            WHERE parent_id IS NULL
              AND name = 'Trash'
              AND kind <> 'trash'
            """
        )
    )

    bind.execute(
        sa.text(
            """
            INSERT INTO namespaces (user_id, parent_id, name, kind, description, created_at)
            SELECT u.id, NULL, 'Trash', 'trash', NULL, NOW()
            FROM users u
            WHERE NOT EXISTS (
                SELECT 1
                FROM namespaces n
                WHERE n.user_id = u.id
                  AND n.parent_id IS NULL
                  AND (n.kind = 'trash' OR n.name = 'Trash')
            )
            """
        )
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "namespaces"):
        return

    bind.execute(
        sa.text(
            """
            DELETE FROM namespaces
            WHERE parent_id IS NULL
              AND kind = 'trash'
              AND name = 'Trash'
            """
        )
    )
