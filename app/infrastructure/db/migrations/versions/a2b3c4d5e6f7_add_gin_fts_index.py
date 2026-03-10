"""Add GIN index for full-text search (Russian) on vector_embeddings.chunk_text

Revision ID: a2b3c4d5e6f7
Revises: f1a2b3c4d5e6
Create Date: 2026-03-10

"""
from alembic import op

revision = 'a2b3c4d5e6f7'
down_revision = ('f1a2b3c4d5e6', 'b1c2d3e4f5a6')
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_ve_fts_russian
        ON vector_embeddings USING GIN (to_tsvector('russian', chunk_text));
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_ve_fts_russian;")
