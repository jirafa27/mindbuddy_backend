"""change embedding dim 256 -> 3584 (switch to Ollama/Qwen)

Revision ID: f1a2b3c4d5e6
Revises: e1f2a3b4c5d6
Create Date: 2026-03-06

"""
from typing import Sequence, Union

import pgvector.sqlalchemy
from alembic import op
import sqlalchemy as sa

revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, None] = 'e1f2a3b4c5d6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Старые векторы (dim=256 от Yandex) несовместимы с новой моделью — удаляем их.
    # Файлы и метаданные остаются нетронутыми; переиндексация произойдёт при следующей загрузке.
    op.execute("DELETE FROM vector_embeddings")
    op.execute(
        "ALTER TABLE vector_embeddings "
        "ALTER COLUMN embedding TYPE vector(3584) "
        "USING embedding::text::vector(3584)"
    )


def downgrade() -> None:
    op.execute("DELETE FROM vector_embeddings")
    op.execute(
        "ALTER TABLE vector_embeddings "
        "ALTER COLUMN embedding TYPE vector(256) "
        "USING embedding::text::vector(256)"
    )
