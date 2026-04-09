"""add namespace tree support

Revision ID: j1k2l3m4n5o6
Revises: i2j3k4l5m6n7
Create Date: 2026-03-30
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "j1k2l3m4n5o6"
down_revision: Union[str, None] = "i2j3k4l5m6n7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(inspector: sa.Inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_column(inspector: sa.Inspector, table_name: str, column_name: str) -> bool:
    return any(col["name"] == column_name for col in inspector.get_columns(table_name))


def _has_index(inspector: sa.Inspector, table_name: str, index_name: str) -> bool:
    return any(idx["name"] == index_name for idx in inspector.get_indexes(table_name))


def _has_fk(inspector: sa.Inspector, table_name: str, fk_name: str) -> bool:
    return any(fk["name"] == fk_name for fk in inspector.get_foreign_keys(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "namespaces"):
        return

    if not _has_column(inspector, "namespaces", "parent_id"):
        op.add_column("namespaces", sa.Column("parent_id", sa.Integer(), nullable=True))

    if not _has_column(inspector, "namespaces", "kind"):
        op.add_column(
            "namespaces",
            sa.Column("kind", sa.String(length=32), nullable=False, server_default="regular"),
        )

    bind.execute(sa.text("UPDATE namespaces SET kind = 'regular' WHERE kind IS NULL"))
    bind.execute(
        sa.text(
            "UPDATE namespaces SET kind = 'inbox' "
            "WHERE parent_id IS NULL AND kind = 'regular' AND name = 'Inbox'"
        )
    )

    inspector = sa.inspect(bind)
    if not _has_fk(inspector, "namespaces", "fk_namespaces_parent_id"):
        op.create_foreign_key(
            "fk_namespaces_parent_id",
            "namespaces",
            "namespaces",
            ["parent_id"],
            ["id"],
        )

    inspector = sa.inspect(bind)
    if not _has_index(inspector, "namespaces", "ix_namespaces_parent_id"):
        op.create_index("ix_namespaces_parent_id", "namespaces", ["parent_id"], unique=False)
    if not _has_index(inspector, "namespaces", "ix_namespaces_kind"):
        op.create_index("ix_namespaces_kind", "namespaces", ["kind"], unique=False)

    if not _has_index(inspector, "namespaces", "uq_namespaces_root_name"):
        op.execute(
            """
            CREATE UNIQUE INDEX uq_namespaces_root_name
            ON namespaces (user_id, name)
            WHERE parent_id IS NULL
            """
        )

    if not _has_index(inspector, "namespaces", "uq_namespaces_child_name"):
        op.execute(
            """
            CREATE UNIQUE INDEX uq_namespaces_child_name
            ON namespaces (user_id, parent_id, name)
            WHERE parent_id IS NOT NULL
            """
        )

    op.alter_column("namespaces", "kind", server_default=None)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "namespaces"):
        return

    if _has_index(inspector, "namespaces", "uq_namespaces_child_name"):
        op.drop_index("uq_namespaces_child_name", table_name="namespaces")
    if _has_index(inspector, "namespaces", "uq_namespaces_root_name"):
        op.drop_index("uq_namespaces_root_name", table_name="namespaces")
    if _has_index(inspector, "namespaces", "ix_namespaces_kind"):
        op.drop_index("ix_namespaces_kind", table_name="namespaces")
    if _has_index(inspector, "namespaces", "ix_namespaces_parent_id"):
        op.drop_index("ix_namespaces_parent_id", table_name="namespaces")

    inspector = sa.inspect(bind)
    if _has_fk(inspector, "namespaces", "fk_namespaces_parent_id"):
        op.drop_constraint("fk_namespaces_parent_id", "namespaces", type_="foreignkey")

    inspector = sa.inspect(bind)
    if _has_column(inspector, "namespaces", "kind"):
        op.drop_column("namespaces", "kind")
    if _has_column(inspector, "namespaces", "parent_id"):
        op.drop_column("namespaces", "parent_id")
