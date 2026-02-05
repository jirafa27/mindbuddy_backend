"""Общие SQL-запросы для векторного поиска (pgvector)."""

# Fallback: поиск по ВСЕМ файлам пользователя (без фильтра по namespace)
# SQLAgent сам добавит фильтр по namespace если пользователь его указал
VECTOR_SEARCH_SQL = """
SELECT ve.chunk_text, f.filename,
       1 - (ve.embedding <=> CAST(:query_embedding AS vector)) AS relevance
FROM vector_embeddings ve
JOIN files f ON f.id = ve.file_id
WHERE f.user_id = :user_id
ORDER BY ve.embedding <=> CAST(:query_embedding AS vector)
LIMIT :limit
"""
