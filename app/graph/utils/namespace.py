"""Вспомогательные функции для работы с пространствами (namespaces) через БД."""
import logging
from typing import Any, List, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

_NAMESPACE_TREE_SQL = """
WITH RECURSIVE namespace_paths AS (
    SELECT
        id,
        user_id,
        parent_id,
        name,
        kind,
        name::text AS full_path,
        0 AS depth
    FROM namespaces
    WHERE user_id = :user_id AND parent_id IS NULL
    UNION ALL
    SELECT
        child.id,
        child.user_id,
        child.parent_id,
        child.name,
        child.kind,
        namespace_paths.full_path || '/' || child.name AS full_path,
        namespace_paths.depth + 1 AS depth
    FROM namespaces child
    JOIN namespace_paths ON child.parent_id = namespace_paths.id
    WHERE child.user_id = :user_id
)
SELECT id, name, kind, full_path, depth
FROM namespace_paths
"""

_NAMESPACE_NAME_BY_ID_SQL = """
WITH RECURSIVE namespace_lineage AS (
    SELECT id, parent_id, name, 0 AS depth
    FROM namespaces
    WHERE id = :ns_id
    UNION ALL
    SELECT parent.id, parent.parent_id, parent.name, lineage.depth + 1 AS depth
    FROM namespaces parent
    JOIN namespace_lineage lineage ON lineage.parent_id = parent.id
)
SELECT string_agg(name, '/' ORDER BY depth DESC) AS full_path
FROM namespace_lineage
"""


def _normalize_namespace_query(name: str) -> str:
    return "/".join(part.strip() for part in (name or "").replace("\\", "/").split("/") if part.strip())


async def resolve_namespace_id(db: Any, user_id: int, name: str) -> Optional[int]:
    """
    Ищет namespace по имени для пользователя.

    Сначала — точное совпадение (case-insensitive).
    Затем — fuzzy matching по префиксу (для опечаток и сокращений).
    Возвращает id или None.
    """
    try:
        normalized_name = _normalize_namespace_query(name)
        result = await db.execute(text(_NAMESPACE_TREE_SQL), {"user_id": user_id})
        rows = result.mappings().all()

        if not rows or not normalized_name:
            return None

        normalized_lower = normalized_name.lower()
        path_matches = [row for row in rows if str(row["full_path"]).lower() == normalized_lower]
        if path_matches:
            path_matches.sort(key=lambda row: int(row["depth"]))
            return path_matches[0]["id"]

        name_matches = [row for row in rows if str(row["name"]).lower() == normalized_lower]
        if len(name_matches) == 1:
            return name_matches[0]["id"]
        if len(name_matches) > 1:
            name_matches.sort(key=lambda row: (int(row["depth"]), str(row["full_path"]).lower()))
            logger.info(
                "[namespace] Ambiguous exact match '%s', selected '%s' (id=%d)",
                name,
                name_matches[0]["full_path"],
                name_matches[0]["id"],
            )
            return name_matches[0]["id"]

        if len(normalized_name) > 4:
            search_prefix = normalized_lower[: max(4, len(normalized_lower) - 2)]
            for ns_row in rows:
                stored_name = str(ns_row["name"]).lower()
                stored_path = str(ns_row["full_path"]).lower()
                stored_prefix = stored_name[: max(4, len(stored_name) - 2)]
                if (
                    stored_name.startswith(search_prefix)
                    or stored_path.startswith(search_prefix)
                    or normalized_lower.startswith(stored_prefix)
                ):
                    logger.info(
                        "[namespace] Fuzzy match: '%s' → '%s' (id=%d)",
                        name, ns_row["full_path"], ns_row["id"],
                    )
                    return ns_row["id"]

        return None
    except Exception as exc:
        logger.warning("[namespace] Failed to resolve '%s': %s", name, exc)
        return None


async def list_namespace_names(db: Any, user_id: int) -> List[str]:
    """Возвращает список полных путей пространств пользователя."""
    try:
        result = await db.execute(
            text(_NAMESPACE_TREE_SQL + " ORDER BY full_path"),
            {"user_id": user_id},
        )
        return [row["full_path"] for row in result.mappings().all()]
    except Exception as exc:
        logger.warning("[namespace] Failed to list namespaces for user %s: %s", user_id, exc)
        return []


async def resolve_namespace_name(db: Any, namespace_id: int) -> Optional[str]:
    """Возвращает полный путь пространства по id или None."""
    try:
        result = await db.execute(
            text(_NAMESPACE_NAME_BY_ID_SQL),
            {"ns_id": namespace_id},
        )
        row = result.mappings().first()
        if row and row["full_path"]:
            return row["full_path"]

        fallback = await db.execute(
            text("SELECT name FROM namespaces WHERE id = :ns_id LIMIT 1"),
            {"ns_id": namespace_id},
        )
        fallback_row = fallback.mappings().first()
        return fallback_row["name"] if fallback_row else None
    except Exception as exc:
        logger.warning("[namespace] Failed to resolve name for id=%s: %s", namespace_id, exc)
        return None
