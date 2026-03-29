"""Вспомогательные функции для работы с пространствами (namespaces) через БД."""
import logging
from typing import Any, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)


async def resolve_namespace_id(db: Any, user_id: int, name: str) -> Optional[int]:
    """
    Ищет namespace по имени для пользователя.

    Сначала — точное совпадение (case-insensitive).
    Затем — fuzzy matching по префиксу (для опечаток и сокращений).
    Возвращает id или None.
    """
    try:
        result = await db.execute(
            text(
                "SELECT id FROM namespaces "
                "WHERE user_id = :user_id AND LOWER(name) = LOWER(:name) LIMIT 1"
            ),
            {"user_id": user_id, "name": name},
        )
        row = result.mappings().first()
        if row:
            return row["id"]

        if len(name) > 4:
            result2 = await db.execute(
                text("SELECT id, name FROM namespaces WHERE user_id = :user_id"),
                {"user_id": user_id},
            )
            rows = result2.mappings().all()
            name_lower = name.lower()
            search_prefix = name_lower[: max(4, len(name_lower) - 2)]
            for ns_row in rows:
                stored = ns_row["name"].lower()
                stored_prefix = stored[: max(4, len(stored) - 2)]
                if stored.startswith(search_prefix) or name_lower.startswith(stored_prefix):
                    logger.info(
                        "[namespace] Fuzzy match: '%s' → '%s' (id=%d)",
                        name, ns_row["name"], ns_row["id"],
                    )
                    return ns_row["id"]

        return None
    except Exception as exc:
        logger.warning("[namespace] Failed to resolve '%s': %s", name, exc)
        return None


async def resolve_namespace_name(db: Any, namespace_id: int) -> Optional[str]:
    """Возвращает название пространства по id или None."""
    try:
        result = await db.execute(
            text("SELECT name FROM namespaces WHERE id = :ns_id LIMIT 1"),
            {"ns_id": namespace_id},
        )
        row = result.mappings().first()
        return row["name"] if row else None
    except Exception as exc:
        logger.warning("[namespace] Failed to resolve name for id=%s: %s", namespace_id, exc)
        return None
