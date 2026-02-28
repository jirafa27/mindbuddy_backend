"""Общие SQL-запросы для векторного поиска (pgvector).
   Search uses user_files (user's library) and content_files (global content).
"""

# Search over user's content: join embeddings -> content_files -> user_files (filter by user_id, optional namespace)
VECTOR_SEARCH_SQL = """
SELECT ve.chunk_text,
       COALESCE(uf.custom_title, cf.media_metadata->>'title', cf.source_url, 'Document') AS filename,
       1 - (ve.embedding <=> CAST(:query_embedding AS vector)) AS relevance
FROM vector_embeddings ve
JOIN content_files cf ON cf.id = ve.file_id
JOIN user_files uf ON uf.file_id = ve.file_id
WHERE uf.user_id = :user_id
  AND (:namespace_id::integer IS NULL OR uf.namespace_id = :namespace_id)
ORDER BY ve.embedding <=> CAST(:query_embedding AS vector)
LIMIT :limit
"""
