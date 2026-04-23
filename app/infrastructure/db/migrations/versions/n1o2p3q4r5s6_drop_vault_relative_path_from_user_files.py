"""drop vault_relative_path from user_files

Revision ID: n1o2p3q4r5s6
Revises: m1n2o3p4q5r6
Create Date: 2026-04-13

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "n1o2p3q4r5s6"
down_revision: Union[str, None] = "m1n2o3p4q5r6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(inspector: sa.Inspector, table_name: str, column_name: str) -> bool:
    return any(col["name"] == column_name for col in inspector.get_columns(table_name))


def _has_index(inspector: sa.Inspector, table_name: str, index_name: str) -> bool:
    return any(idx["name"] == index_name for idx in inspector.get_indexes(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_index(inspector, "user_files", "ix_user_files_vault_relative_path"):
        op.drop_index("ix_user_files_vault_relative_path", table_name="user_files")

    inspector = sa.inspect(bind)
    if _has_column(inspector, "user_files", "vault_relative_path"):
        op.drop_column("user_files", "vault_relative_path")


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_column(inspector, "user_files", "vault_relative_path"):
        op.add_column("user_files", sa.Column("vault_relative_path", sa.String(length=1024), nullable=True))

    inspector = sa.inspect(bind)
    if not _has_index(inspector, "user_files", "ix_user_files_vault_relative_path"):
        op.create_index("ix_user_files_vault_relative_path", "user_files", ["vault_relative_path"], unique=False)
