"""add sync state to user_files

Revision ID: i2j3k4l5m6n7
Revises: h1i2j3k4l5m6
Create Date: 2026-03-30

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "i2j3k4l5m6n7"
down_revision: Union[str, None] = "h1i2j3k4l5m6"
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

    if not _has_column(inspector, "user_files", "vault_relative_path"):
        op.add_column("user_files", sa.Column("vault_relative_path", sa.String(length=1024), nullable=True))
    if not _has_column(inspector, "user_files", "updated_at"):
        op.add_column("user_files", sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()))
    if not _has_column(inspector, "user_files", "desktop_updated_at"):
        op.add_column("user_files", sa.Column("desktop_updated_at", sa.DateTime(), nullable=True))
    if not _has_column(inspector, "user_files", "app_updated_at"):
        op.add_column("user_files", sa.Column("app_updated_at", sa.DateTime(), nullable=True))
    if not _has_column(inspector, "user_files", "content_revision"):
        op.add_column("user_files", sa.Column("content_revision", sa.Integer(), nullable=False, server_default="1"))
    if not _has_column(inspector, "user_files", "last_update_source"):
        op.add_column("user_files", sa.Column("last_update_source", sa.String(length=32), nullable=True))
    if not _has_column(inspector, "user_files", "is_conflict_copy"):
        op.add_column("user_files", sa.Column("is_conflict_copy", sa.Boolean(), nullable=False, server_default=sa.false()))
    if not _has_column(inspector, "user_files", "conflict_origin_user_file_id"):
        op.add_column("user_files", sa.Column("conflict_origin_user_file_id", sa.Integer(), nullable=True))

    inspector = sa.inspect(bind)
    if not _has_index(inspector, "user_files", "ix_user_files_vault_relative_path"):
        op.create_index("ix_user_files_vault_relative_path", "user_files", ["vault_relative_path"], unique=False)
    if not _has_index(inspector, "user_files", "ix_user_files_updated_at"):
        op.create_index("ix_user_files_updated_at", "user_files", ["updated_at"], unique=False)
    if not _has_index(inspector, "user_files", "ix_user_files_conflict_origin_user_file_id"):
        op.create_index(
            "ix_user_files_conflict_origin_user_file_id",
            "user_files",
            ["conflict_origin_user_file_id"],
            unique=False,
        )
    if not _has_fk(inspector, "user_files", "fk_user_files_conflict_origin_user_file_id"):
        op.create_foreign_key(
            "fk_user_files_conflict_origin_user_file_id",
            "user_files",
            "user_files",
            ["conflict_origin_user_file_id"],
            ["id"],
        )

    inspector = sa.inspect(bind)
    if not _has_table(inspector, "sync_commands"):
        op.create_table(
            "sync_commands",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("user_file_id", sa.Integer(), nullable=False),
            sa.Column("command_type", sa.String(length=32), nullable=False),
            sa.Column("payload_json", sa.JSON(), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("acked_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["user_file_id"], ["user_files.id"]),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.PrimaryKeyConstraint("id"),
        )

    inspector = sa.inspect(bind)
    if not _has_index(inspector, "sync_commands", op.f("ix_sync_commands_id")):
        op.create_index(op.f("ix_sync_commands_id"), "sync_commands", ["id"], unique=False)
    if not _has_index(inspector, "sync_commands", op.f("ix_sync_commands_user_id")):
        op.create_index(op.f("ix_sync_commands_user_id"), "sync_commands", ["user_id"], unique=False)
    if not _has_index(inspector, "sync_commands", op.f("ix_sync_commands_user_file_id")):
        op.create_index(op.f("ix_sync_commands_user_file_id"), "sync_commands", ["user_file_id"], unique=False)
    if not _has_index(inspector, "sync_commands", op.f("ix_sync_commands_command_type")):
        op.create_index(op.f("ix_sync_commands_command_type"), "sync_commands", ["command_type"], unique=False)
    if not _has_index(inspector, "sync_commands", op.f("ix_sync_commands_status")):
        op.create_index(op.f("ix_sync_commands_status"), "sync_commands", ["status"], unique=False)
    if not _has_index(inspector, "sync_commands", op.f("ix_sync_commands_created_at")):
        op.create_index(op.f("ix_sync_commands_created_at"), "sync_commands", ["created_at"], unique=False)

    if _has_column(inspector, "user_files", "updated_at"):
        op.alter_column("user_files", "updated_at", server_default=None)
    if _has_column(inspector, "user_files", "content_revision"):
        op.alter_column("user_files", "content_revision", server_default=None)
    if _has_column(inspector, "user_files", "is_conflict_copy"):
        op.alter_column("user_files", "is_conflict_copy", server_default=None)
    if _has_table(inspector, "sync_commands"):
        op.alter_column("sync_commands", "status", server_default=None)
        op.alter_column("sync_commands", "created_at", server_default=None)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_table(inspector, "sync_commands"):
        if _has_index(inspector, "sync_commands", op.f("ix_sync_commands_created_at")):
            op.drop_index(op.f("ix_sync_commands_created_at"), table_name="sync_commands")
        if _has_index(inspector, "sync_commands", op.f("ix_sync_commands_status")):
            op.drop_index(op.f("ix_sync_commands_status"), table_name="sync_commands")
        if _has_index(inspector, "sync_commands", op.f("ix_sync_commands_command_type")):
            op.drop_index(op.f("ix_sync_commands_command_type"), table_name="sync_commands")
        if _has_index(inspector, "sync_commands", op.f("ix_sync_commands_user_file_id")):
            op.drop_index(op.f("ix_sync_commands_user_file_id"), table_name="sync_commands")
        if _has_index(inspector, "sync_commands", op.f("ix_sync_commands_user_id")):
            op.drop_index(op.f("ix_sync_commands_user_id"), table_name="sync_commands")
        if _has_index(inspector, "sync_commands", op.f("ix_sync_commands_id")):
            op.drop_index(op.f("ix_sync_commands_id"), table_name="sync_commands")
        op.drop_table("sync_commands")

    inspector = sa.inspect(bind)
    if _has_fk(inspector, "user_files", "fk_user_files_conflict_origin_user_file_id"):
        op.drop_constraint("fk_user_files_conflict_origin_user_file_id", "user_files", type_="foreignkey")
    if _has_index(inspector, "user_files", "ix_user_files_conflict_origin_user_file_id"):
        op.drop_index("ix_user_files_conflict_origin_user_file_id", table_name="user_files")
    if _has_index(inspector, "user_files", "ix_user_files_updated_at"):
        op.drop_index("ix_user_files_updated_at", table_name="user_files")
    if _has_index(inspector, "user_files", "ix_user_files_vault_relative_path"):
        op.drop_index("ix_user_files_vault_relative_path", table_name="user_files")
    if _has_column(inspector, "user_files", "conflict_origin_user_file_id"):
        op.drop_column("user_files", "conflict_origin_user_file_id")
    if _has_column(inspector, "user_files", "is_conflict_copy"):
        op.drop_column("user_files", "is_conflict_copy")
    if _has_column(inspector, "user_files", "last_update_source"):
        op.drop_column("user_files", "last_update_source")
    if _has_column(inspector, "user_files", "content_revision"):
        op.drop_column("user_files", "content_revision")
    if _has_column(inspector, "user_files", "app_updated_at"):
        op.drop_column("user_files", "app_updated_at")
    if _has_column(inspector, "user_files", "desktop_updated_at"):
        op.drop_column("user_files", "desktop_updated_at")
    if _has_column(inspector, "user_files", "updated_at"):
        op.drop_column("user_files", "updated_at")
    if _has_column(inspector, "user_files", "vault_relative_path"):
        op.drop_column("user_files", "vault_relative_path")
