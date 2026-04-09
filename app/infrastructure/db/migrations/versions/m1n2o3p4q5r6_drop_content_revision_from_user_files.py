"""drop content_revision from user_files

Revision ID: m1n2o3p4q5r6
Revises: l1m2n3o4p5q6
Create Date: 2026-04-06

"""
from alembic import op
import sqlalchemy as sa

revision = "m1n2o3p4q5r6"
down_revision = "l1m2n3o4p5q6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("user_files", "content_revision")


def downgrade() -> None:
    op.add_column(
        "user_files",
        sa.Column(
            "content_revision",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),
    )
