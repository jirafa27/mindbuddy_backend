"""Общие SQL-запросы для векторного поиска (pgvector).
   Search: vector_embeddings -> files -> user_files (фильтр по user_id, опционально namespace).
"""

VECTOR_SEARCH_SQL = """
SELECT ve.chunk_text,
       COALESCE(uf.custom_title, f.media_metadata->>'title', f.source_url, f.file_path, 'Document') AS filename,
       1 - (ve.embedding <=> CAST(:query_embedding AS vector)) AS relevance
FROM vector_embeddings ve
JOIN files f ON f.id = ve.file_id
JOIN user_files uf ON uf.file_id = ve.file_id
WHERE uf.user_id = :user_id
  AND (:namespace_id::integer IS NULL OR uf.namespace_id = :namespace_id)
ORDER BY ve.embedding <=> CAST(:query_embedding AS vector)
LIMIT :limit
"""

VECTOR_SEARCH_BY_FILE_SQL = """
SELECT ve.chunk_text,
       COALESCE(uf.custom_title, f.media_metadata->>'title', f.source_url, f.file_path, 'Document') AS filename,
       1 - (ve.embedding <=> CAST(:query_embedding AS vector)) AS relevance
FROM vector_embeddings ve
JOIN files f ON f.id = ve.file_id
JOIN user_files uf ON uf.file_id = ve.file_id
WHERE uf.user_id = :user_id
  AND uf.id = :file_id
ORDER BY ve.embedding <=> CAST(:query_embedding AS vector)
LIMIT :limit
"""

# Поиск по нескольким user_files.id. __FILE_IDS__ заменяется на список id во время выполнения.
VECTOR_SEARCH_BY_FILES_SQL = """
SELECT ve.chunk_text,
       COALESCE(uf.custom_title, f.media_metadata->>'title', f.source_url, f.file_path, 'Document') AS filename,
       1 - (ve.embedding <=> CAST(:query_embedding AS vector)) AS relevance
FROM vector_embeddings ve
JOIN files f ON f.id = ve.file_id
JOIN user_files uf ON uf.file_id = ve.file_id
WHERE uf.user_id = :user_id
  AND uf.id IN (__FILE_IDS__)
ORDER BY ve.embedding <=> CAST(:query_embedding AS vector)
LIMIT :limit
"""

LIST_FILES_SQL = """
SELECT COALESCE(uf.custom_title, f.media_metadata->>'title', f.source_url, f.file_path, 'Document') AS filename,
       n.name AS namespace_name,
       TO_CHAR(uf.created_at, 'DD.MM.YYYY') AS created_at
FROM user_files uf
JOIN files f ON f.id = uf.file_id
LEFT JOIN namespaces n ON n.id = uf.namespace_id
WHERE uf.user_id = :user_id
  AND (CAST(:namespace_id AS integer) IS NULL OR uf.namespace_id = :namespace_id)
ORDER BY uf.created_at DESC
LIMIT :limit
"""

FIND_FILE_SQL = """
SELECT uf.id AS file_id,
       COALESCE(uf.custom_title, f.media_metadata->>'title', f.source_url, f.file_path, 'Document') AS filename
FROM user_files uf
JOIN files f ON f.id = uf.file_id
WHERE uf.user_id = :user_id
  AND (CAST(:namespace_id AS integer) IS NULL OR uf.namespace_id = :namespace_id)
  AND LOWER(COALESCE(uf.custom_title, f.media_metadata->>'title', f.source_url, f.file_path, '')) ILIKE LOWER(:filename_pattern)
ORDER BY uf.created_at DESC
LIMIT 1
"""

# Поиск файла по смысловому содержанию через векторные эмбеддинги.
#
# Особенности:
# - one_user_file: берём ONE user_files.id на физический файл — это устраняет
#   дублирование чанков когда файл лежит в нескольких неймспейсах.
# - rn_dist: ранжирование чанков по косинусному расстоянию (для relevance).
# - rn_snippet: ранжирование для сниппета — чанки содержащие ключевое слово
#   (ILIKE :kw_pattern) идут первыми, остальные сортируются по расстоянию.
#   Так LLM-реранкинг видит тематически релевантный текст, а не только заголовки.
FIND_FILE_BY_TOPIC_SQL = """
WITH one_user_file AS (
  SELECT DISTINCT ON (file_id)
         id                                                           AS user_file_id,
         file_id,
         COALESCE(custom_title,
                  (SELECT media_metadata->>'title' FROM files WHERE id = user_files.file_id),
                  (SELECT source_url               FROM files WHERE id = user_files.file_id),
                  (SELECT file_path                FROM files WHERE id = user_files.file_id),
                  'Document')                                         AS filename
  FROM user_files
  WHERE user_id = :user_id
    AND (CAST(:namespace_id AS integer) IS NULL OR namespace_id = :namespace_id)
  ORDER BY file_id, id
),
chunks AS (
  SELECT ouf.file_id   AS raw_file_id,
         ouf.user_file_id AS file_id,
         ouf.filename,
         ve.chunk_text,
         ve.embedding <=> CAST(:query_embedding AS vector) AS distance
  FROM vector_embeddings ve
  JOIN one_user_file ouf ON ouf.file_id = ve.file_id
),
ranked AS (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY raw_file_id ORDER BY distance) AS rn_dist,
         ROW_NUMBER() OVER (
           PARTITION BY raw_file_id
           ORDER BY
             CASE WHEN LOWER(chunk_text) LIKE LOWER(:kw_pattern) THEN 0 ELSE 1 END,
             distance
         ) AS rn_snippet
  FROM chunks
),
best_per_file AS (
  SELECT
    raw_file_id,
    MAX(file_id)    FILTER (WHERE rn_dist = 1)   AS file_id,
    MAX(filename)   FILTER (WHERE rn_dist = 1)   AS filename,
    MIN(distance)                                AS best_distance,
    STRING_AGG(LEFT(chunk_text, 300), ' ... ' ORDER BY rn_snippet)
      FILTER (WHERE rn_snippet <= 3)             AS snippet
  FROM ranked
  GROUP BY raw_file_id
)
SELECT file_id,
       filename,
       1 - best_distance  AS relevance,
       snippet
FROM best_per_file
WHERE 1 - best_distance >= :min_relevance
ORDER BY best_distance
LIMIT :limit
"""

# Гибридный поиск: векторный (top-30) + полнотекстовый (FTS, русский язык).
# Объединяет результаты по ve_id, ранжирует по relevance + fts_score.
# Параметры: :query_embedding, :fts_query, :user_id, :limit.
# __FILE_IDS__ заменяется на список user_files.id перед выполнением.
HYBRID_SEARCH_BY_FILES_SQL = """
WITH fts_matches AS (
    SELECT ve.id AS ve_id,
           ve.chunk_text,
           COALESCE(uf.custom_title, f.media_metadata->>'title', f.source_url, f.file_path, 'Document') AS filename,
           1 - (ve.embedding <=> CAST(:query_embedding AS vector)) AS relevance,
           ts_rank(to_tsvector('russian', ve.chunk_text),
                   plainto_tsquery('russian', :fts_query))          AS fts_score
    FROM vector_embeddings ve
    JOIN files f ON f.id = ve.file_id
    JOIN user_files uf ON uf.file_id = ve.file_id
    WHERE uf.user_id = :user_id
      AND uf.id IN (__FILE_IDS__)
      AND to_tsvector('russian', ve.chunk_text) @@ plainto_tsquery('russian', :fts_query)
),
vector_top AS (
    SELECT ve.id AS ve_id,
           ve.chunk_text,
           COALESCE(uf.custom_title, f.media_metadata->>'title', f.source_url, f.file_path, 'Document') AS filename,
           1 - (ve.embedding <=> CAST(:query_embedding AS vector)) AS relevance,
           0::float                                                  AS fts_score
    FROM vector_embeddings ve
    JOIN files f ON f.id = ve.file_id
    JOIN user_files uf ON uf.file_id = ve.file_id
    WHERE uf.user_id = :user_id
      AND uf.id IN (__FILE_IDS__)
    ORDER BY ve.embedding <=> CAST(:query_embedding AS vector)
    LIMIT 30
),
combined AS (
    SELECT ve_id,
           MAX(chunk_text)  AS chunk_text,
           MAX(filename)    AS filename,
           MAX(relevance)   AS relevance,
           MAX(fts_score)   AS fts_score
    FROM (
        SELECT ve_id, chunk_text, filename, relevance, fts_score FROM fts_matches
        UNION ALL
        SELECT ve_id, chunk_text, filename, relevance, fts_score FROM vector_top
    ) sub
    GROUP BY ve_id
)
SELECT chunk_text, filename, relevance
FROM combined
ORDER BY relevance + fts_score DESC
LIMIT :limit
"""

# Гибридный поиск по всем файлам пользователя (без фильтра по user_files.id).
# Параметры: :query_embedding, :fts_query, :user_id, :namespace_id, :limit.
HYBRID_SEARCH_SQL = """
WITH fts_matches AS (
    SELECT ve.id AS ve_id,
           ve.chunk_text,
           COALESCE(uf.custom_title, f.media_metadata->>'title', f.source_url, f.file_path, 'Document') AS filename,
           1 - (ve.embedding <=> CAST(:query_embedding AS vector)) AS relevance,
           ts_rank(to_tsvector('russian', ve.chunk_text),
                   plainto_tsquery('russian', :fts_query))          AS fts_score
    FROM vector_embeddings ve
    JOIN files f ON f.id = ve.file_id
    JOIN user_files uf ON uf.file_id = ve.file_id
    WHERE uf.user_id = :user_id
      AND (:namespace_id::integer IS NULL OR uf.namespace_id = :namespace_id)
      AND to_tsvector('russian', ve.chunk_text) @@ plainto_tsquery('russian', :fts_query)
),
vector_top AS (
    SELECT ve.id AS ve_id,
           ve.chunk_text,
           COALESCE(uf.custom_title, f.media_metadata->>'title', f.source_url, f.file_path, 'Document') AS filename,
           1 - (ve.embedding <=> CAST(:query_embedding AS vector)) AS relevance,
           0::float                                                  AS fts_score
    FROM vector_embeddings ve
    JOIN files f ON f.id = ve.file_id
    JOIN user_files uf ON uf.file_id = ve.file_id
    WHERE uf.user_id = :user_id
      AND (:namespace_id::integer IS NULL OR uf.namespace_id = :namespace_id)
    ORDER BY ve.embedding <=> CAST(:query_embedding AS vector)
    LIMIT 30
),
combined AS (
    SELECT ve_id,
           MAX(chunk_text)  AS chunk_text,
           MAX(filename)    AS filename,
           MAX(relevance)   AS relevance,
           MAX(fts_score)   AS fts_score
    FROM (
        SELECT ve_id, chunk_text, filename, relevance, fts_score FROM fts_matches
        UNION ALL
        SELECT ve_id, chunk_text, filename, relevance, fts_score FROM vector_top
    ) sub
    GROUP BY ve_id
)
SELECT chunk_text, filename, relevance
FROM combined
ORDER BY relevance + fts_score DESC
LIMIT :limit
"""

# Поиск файла по буквальному вхождению текста в чанки (ILIKE).
# Возвращает файл с наибольшим количеством совпадающих чанков.
FIND_FILE_BY_CONTENT_SQL = """
SELECT uf.id AS file_id,
       COALESCE(uf.custom_title, f.media_metadata->>'title', f.source_url, f.file_path, 'Document') AS filename,
       COUNT(*) AS match_count
FROM vector_embeddings ve
JOIN files f ON f.id = ve.file_id
JOIN user_files uf ON uf.file_id = ve.file_id
WHERE uf.user_id = :user_id
  AND (CAST(:namespace_id AS integer) IS NULL OR uf.namespace_id = :namespace_id)
  AND LOWER(ve.chunk_text) LIKE LOWER(:content_pattern)
GROUP BY uf.id, filename
ORDER BY match_count DESC
LIMIT :limit
"""
