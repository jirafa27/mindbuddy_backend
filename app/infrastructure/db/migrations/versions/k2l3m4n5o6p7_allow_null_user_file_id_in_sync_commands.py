"""allow null user_file_id in sync commands

Revision ID: k2l3m4n5o6p7
Revises: i2j3k4l5m6n7
Create Date: 2026-03-30
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "k2l3m4n5o6p7"
down_revision: Union[str, None] = "j1k2l3m4n5o6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(inspector: sa.Inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_column(inspector: sa.Inspector, table_name: str, column_name: str) -> bool:
    return any(col["name"] == column_name for col in inspector.get_columns(table_name))


def _is_nullable(inspector: sa.Inspector, table_name: str, column_name: str) -> bool:
    for col in inspector.get_columns(table_name):
        if col["name"] == column_name:
            return bool(col.get("nullable"))
    return False


def _find_fk_name(inspector: sa.Inspector, table_name: str, constrained_columns: list[str]) -> str | None:
    expected = list(constrained_columns)
    for fk in inspector.get_foreign_keys(table_name):
        if list(fk.get("constrained_columns") or []) == expected:
            return fk.get("name")
    return None


def _fk_ondelete(inspector: sa.Inspector, table_name: str, constrained_columns: list[str]) -> str | None:
    expected = list(constrained_columns)
    for fk in inspector.get_foreign_keys(table_name):
        if list(fk.get("constrained_columns") or []) == expected:
            options = fk.get("options") or {}
            return options.get("ondelete")
    return None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "sync_commands") or not _has_column(inspector, "sync_commands", "user_file_id"):
        return

    if not _is_nullable(inspector, "sync_commands", "user_file_id"):
        op.alter_column("sync_commands", "user_file_id", existing_type=sa.Integer(), nullable=True)

    inspector = sa.inspect(bind)
    fk_name = _find_fk_name(inspector, "sync_commands", ["user_file_id"])
    if fk_name and _fk_ondelete(inspector, "sync_commands", ["user_file_id"]) != "SET NULL":
        op.drop_constraint(fk_name, "sync_commands", type_="foreignkey")
        fk_name = None

    if fk_name is None:
        op.create_foreign_key(
            "sync_commands_user_file_id_fkey",
            "sync_commands",
            "user_files",
            ["user_file_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _has_table(inspector, "sync_commands") or not _has_column(inspector, "sync_commands", "user_file_id"):
        return

    bind.execute(sa.text("DELETE FROM sync_commands WHERE user_file_id IS NULL"))

    inspector = sa.inspect(bind)
    fk_name = _find_fk_name(inspector, "sync_commands", ["user_file_id"])
    if fk_name:
        op.drop_constraint(fk_name, "sync_commands", type_="foreignkey")

    op.create_foreign_key(
        "sync_commands_user_file_id_fkey",
        "sync_commands",
        "user_files",
        ["user_file_id"],
        ["id"],
    )
    op.alter_column("sync_commands", "user_file_id", existing_type=sa.Integer(), nullable=False)
