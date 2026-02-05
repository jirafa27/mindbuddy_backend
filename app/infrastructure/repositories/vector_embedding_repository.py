from typing import List, Optional

import tiktoken
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.domain.entities import SearchResultRow
from app.infrastructure.db.models import VectorEmbedding
from app.infrastructure.repositories.vector_queries import VECTOR_SEARCH_SQL


class PgVectorRepository:
    def __init__(self, db: Session):
        self.db = db
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def create_batch(
        self,
        file_id: int,
        chunks: List[str],
        embeddings: List[List[float]],
        namespace_id: Optional[int] = None,
    ) -> List[VectorEmbedding]:
        vector_embeddings = []
        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            vector_embedding = VectorEmbedding(
                file_id=file_id,
                namespace_id=namespace_id,
                chunk_index=idx,
                chunk_text=chunk_text,
                embedding=embedding,
            )
            vector_embeddings.append(vector_embedding)
        self.db.add_all(vector_embeddings)
        return vector_embeddings

    def search_by_embedding(
        self,
        query_embedding: List[float],
        limit: int = 5,
        namespace_id: Optional[int] = None,
    ) -> List[SearchResultRow]:
        vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        stmt = text(VECTOR_SEARCH_SQL).bindparams(
            namespace_id=namespace_id,
            query_embedding=vec_str,
            limit=limit,
        )
        result = self.db.execute(stmt)
        rows = result.mappings().all()
        return [dict(row) for row in rows]
