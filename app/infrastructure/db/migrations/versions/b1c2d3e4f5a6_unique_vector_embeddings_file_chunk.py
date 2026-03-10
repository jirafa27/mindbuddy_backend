"""unique constraint on vector_embeddings(file_id, chunk_index)

Revision ID: b1c2d3e4f5a6
Revises: a1b2c3d4e5f6
Create Date: 2026-03-09

"""
from alembic import op

revision = 'b1c2d3e4f5a6'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Удаляем дубли перед созданием constraint:
    # оставляем строку с наименьшим id для каждой пары (file_id, chunk_index)
    op.execute("""
        DELETE FROM vector_embeddings
        WHERE id IN (
            SELECT id FROM (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY file_id, chunk_index
                           ORDER BY id
                       ) AS rn
                FROM vector_embeddings
            ) t
            WHERE rn > 1
        )
    """)
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'uq_vector_embeddings_file_chunk'
            ) THEN
                ALTER TABLE vector_embeddings
                ADD CONSTRAINT uq_vector_embeddings_file_chunk
                UNIQUE (file_id, chunk_index);
            END IF;
        END
        $$;
    """)


def downgrade() -> None:
    op.drop_constraint(
        'uq_vector_embeddings_file_chunk',
        'vector_embeddings',
        type_='unique',
    )
