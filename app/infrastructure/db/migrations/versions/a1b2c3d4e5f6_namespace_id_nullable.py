"""namespace_id nullable in files and vector_embeddings

Revision ID: a1b2c3d4e5f6
Revises: 9f0a80d107d2
Create Date: 2026-02-01

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "9f0a80d107d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "files",
        "namespace_id",
        existing_type=sa.Integer(),
        nullable=True,
    )
    op.alter_column(
        "vector_embeddings",
        "namespace_id",
        existing_type=sa.Integer(),
        nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "vector_embeddings",
        "namespace_id",
        existing_type=sa.Integer(),
        nullable=False,
    )
    op.alter_column(
        "files",
        "namespace_id",
        existing_type=sa.Integer(),
        nullable=False,
    )
