"""CrudNode — выполнение CRUD-операций над пространствами и файлами.

Интенты:
- create_namespace: создать пространство
- delete_namespace: запросить подтверждение → set_pending_action
- move_file: переместить файл в пространство
- create_file: создать файл из текста
- delete_file: запросить подтверждение → set_pending_action
- edit_file: заменить содержимое файла

Для удаления CrudNode НЕ выполняет действие сразу — возвращает вопрос подтверждения
и просит ChatService сохранить pending_action. Фактическое удаление происходит в
ChatService при следующем сообщении с подтверждением (до вызова графа).
"""
import logging
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from sqlalchemy import text

from app.graph.state import AskState
from app.core.enums import IntentType
from app.core.exceptions import NotFoundError, ForbiddenError, ValidationError
from app.services.file_service import FileService
from app.services.namespace_service import NamespaceService

logger = logging.getLogger(__name__)


class CrudNode:
    """Выполняет CRUD-операции над пространствами и файлами."""

    def __init__(
        self,
        *,
        file_service: FileService,
        namespace_service: Optional[NamespaceService] = None,
    ) -> None:
        self.file_service = file_service
        self.namespace_service = namespace_service

    async def run(self, state: AskState, config: RunnableConfig) -> dict[str, Any]:
        intent = state.get("intent")
        user_id = state.get("user_id")
        agent_steps = list(state.get("agent_steps") or []) + [f"CrudNode({intent})"]

        if not user_id:
            return {"answer": "Ошибка: не указан пользователь.", "agent_steps": agent_steps}

        try:
            if intent == IntentType.CREATE_NAMESPACE:
                return await self._create_namespace(state, user_id, agent_steps)
            elif intent == IntentType.DELETE_NAMESPACE:
                return await self._request_delete_namespace(state, user_id, agent_steps, config)
            elif intent == IntentType.EDIT_NAMESPACE:
                return await self._edit_namespace(state, user_id, agent_steps)
            elif intent == IntentType.MOVE_FILE:
                return await self._move_file(state, user_id, agent_steps, config)
            elif intent == IntentType.CREATE_FILE:
                return await self._create_file(state, user_id, agent_steps, config)
            elif intent == IntentType.DELETE_FILE:
                return await self._request_delete_file(state, user_id, agent_steps, config)
            elif intent == IntentType.EDIT_FILE:
                return await self._edit_file(state, user_id, agent_steps, config)
            else:
                return {
                    "answer": f"Неизвестная операция: {intent}",
                    "agent_steps": agent_steps,
                }
        except (NotFoundError, ForbiddenError, ValidationError) as exc:
            return {"answer": str(exc), "agent_steps": agent_steps}
        except Exception:
            logger.exception("[CrudNode] Unexpected error intent=%s user_id=%s", intent, user_id)
            return {"answer": "Произошла ошибка при выполнении операции.", "agent_steps": agent_steps}

    # ------------------------------------------------------------------
    # Пространства
    # ------------------------------------------------------------------

    async def _create_namespace(
        self, state: AskState, user_id: int, agent_steps: list
    ) -> dict[str, Any]:
        name = state.get("entity_name")
        description = state.get("entity_description")

        if not name:
            return {
                "answer": "Укажите название пространства. Например: «Создай пространство Работа».",
                "agent_steps": agent_steps,
            }

        if not self.namespace_service:
            return {"answer": "Сервис пространств недоступен.", "agent_steps": agent_steps}

        namespace = await self.namespace_service.create_namespace(
            name=name, user_id=user_id, description=description
        )
        desc_part = f"\nОписание: {description}" if description else ""
        return {
            "answer": f"Пространство «{namespace.name}» создано.{desc_part}",
            "agent_steps": agent_steps,
        }

    async def _edit_namespace(
        self, state: AskState, user_id: int, agent_steps: list
    ) -> dict[str, Any]:
        namespace_id = state.get("namespace_id")
        namespace_name = state.get("namespace_name_hint") or state.get("entity_name")
        new_description = state.get("entity_description")
        new_name = state.get("entity_content")

        if not namespace_id:
            hint = f" «{namespace_name}»" if namespace_name else ""
            return {
                "answer": f"Не нашёл пространство{hint}. Уточните название.",
                "agent_steps": agent_steps,
            }

        if not self.namespace_service:
            return {"answer": "Сервис пространств недоступен.", "agent_steps": agent_steps}

        if not new_name and new_description is None:
            return {
                "answer": "Укажите, что именно изменить: новое название или описание пространства.",
                "agent_steps": agent_steps,
            }

        namespace = await self.namespace_service.update_namespace(
            namespace_id=namespace_id,
            user_id=user_id,
            name=new_name or None,
            description=new_description,
        )

        parts = []
        if new_name:
            parts.append(f"переименовано в «{namespace.name}»")
        if new_description is not None:
            parts.append(f"описание обновлено")
        summary = ", ".join(parts)
        return {
            "answer": f"Пространство {summary}.",
            "agent_steps": agent_steps,
        }

    async def _request_delete_namespace(
        self,
        state: AskState,
        user_id: int,
        agent_steps: list,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """Находит пространство, возвращает вопрос подтверждения и сохраняет pending_action."""
        namespace_name = state.get("namespace_name_hint")
        namespace_id = state.get("namespace_id")

        if not namespace_id and namespace_name:
            namespace_id = await self._resolve_namespace_id(config, user_id, namespace_name)

        if not namespace_id:
            hint = f" «{namespace_name}»" if namespace_name else ""
            return {
                "answer": f"Не нашёл пространство{hint}. Уточните название.",
                "agent_steps": agent_steps,
            }

        if not self.namespace_service:
            return {"answer": "Сервис пространств недоступен.", "agent_steps": agent_steps}

        try:
            namespace = await self.namespace_service.get_namespace(
                namespace_id=namespace_id, user_id=user_id
            )
        except (NotFoundError, ForbiddenError) as exc:
            return {"answer": str(exc), "agent_steps": agent_steps}

        pending = {
            "type": "delete_namespace",
            "params": {"namespace_id": namespace_id},
            "target": f"пространство «{namespace.name}»",
        }
        return {
            "answer": (
                f"Вы уверены, что хотите удалить пространство «{namespace.name}»? "
                "Это действие нельзя отменить. Напишите «да» для подтверждения."
            ),
            "pending_action": pending,
            "agent_steps": agent_steps,
        }

    # ------------------------------------------------------------------
    # Файлы
    # ------------------------------------------------------------------

    async def _move_file(
        self,
        state: AskState,
        user_id: int,
        agent_steps: list,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        namespace_name = state.get("namespace_name_hint")
        namespace_id = state.get("namespace_id")
        file_id = state.get("history_file_id") or (
            state.get("search_file_ids") or [None]
        )[0]
        search_query = state.get("search_query") or state.get("entity_name")

        if not namespace_id and namespace_name:
            namespace_id = await self._resolve_namespace_id(config, user_id, namespace_name)

        if not namespace_id:
            hint = f" «{namespace_name}»" if namespace_name else ""
            return {
                "answer": f"Не нашёл пространство назначения{hint}. Укажите корректное название.",
                "agent_steps": agent_steps,
            }

        if not file_id and search_query:
            file_id = await self._find_file_id_by_name(config, user_id, search_query)

        if not file_id:
            hint = f" «{search_query}»" if search_query else ""
            return {
                "answer": f"Не нашёл файл{hint}. Уточните название.",
                "agent_steps": agent_steps,
            }

        file_info = await self.file_service.move_to_namespace(
            file_id=file_id,
            namespace_id=namespace_id,
            user_id=user_id,
        )
        ns_display = f"«{namespace_name}»" if namespace_name else f"(id={namespace_id})"
        return {
            "answer": f"Файл «{file_info.filename}» перемещён в пространство {ns_display}.",
            "agent_steps": agent_steps,
        }

    async def _create_file(
        self, state: AskState, user_id: int, agent_steps: list, config: RunnableConfig = None
    ) -> dict[str, Any]:
        content = state.get("entity_content")
        title = state.get("entity_name")
        namespace_id = state.get("namespace_id")

        if not content:
            return {
                "answer": "Укажите содержимое файла. Например: «Создай заметку Идеи: текст заметки».",
                "agent_steps": agent_steps,
            }

        # Если пространство не указано — пытаемся найти Inbox как дефолтное
        namespace_name = state.get("namespace_name_hint")
        if namespace_id is None and config is not None:
            inbox_id = await self._resolve_namespace_id(config, user_id, "Inbox")
            if inbox_id is not None:
                namespace_id = inbox_id
                namespace_name = "Inbox"
                logger.info("[CrudNode] create_file: no namespace, using Inbox (id=%d)", inbox_id)

        file_created = await self.file_service.create_file_from_text(
            text=content,
            user_id=user_id,
            namespace_id=namespace_id,
            title=title,
        )
        ns_part = f" в пространство «{namespace_name or namespace_id}»" if namespace_id else ""
        return {
            "answer": f"Файл «{file_created.filename}» создан и сохранён{ns_part}.",
            "file_id": file_created.file_id,
            "agent_steps": agent_steps,
        }

    async def _request_delete_file(
        self,
        state: AskState,
        user_id: int,
        agent_steps: list,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """Находит файл, возвращает вопрос подтверждения и сохраняет pending_action."""
        file_id = state.get("history_file_id") or (
            state.get("search_file_ids") or [None]
        )[0]
        search_query = state.get("search_query") or state.get("entity_name")

        if not file_id and search_query:
            file_id = await self._find_file_id_by_name(config, user_id, search_query)

        if not file_id:
            hint = f" «{search_query}»" if search_query else ""
            return {
                "answer": f"Не нашёл файл{hint}. Уточните название.",
                "agent_steps": agent_steps,
            }

        filename = await self._get_filename(config, file_id, user_id)
        display = f"«{filename}»" if filename else f"(id={file_id})"

        pending = {
            "type": "delete_file",
            "params": {"file_id": file_id},
            "target": f"файл {display}",
        }
        return {
            "answer": (
                f"Вы уверены, что хотите удалить файл {display}? "
                "Это действие нельзя отменить. Напишите «да» для подтверждения."
            ),
            "pending_action": pending,
            "agent_steps": agent_steps,
        }

    async def _edit_file(
        self,
        state: AskState,
        user_id: int,
        agent_steps: list,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        file_id = state.get("history_file_id") or (
            state.get("search_file_ids") or [None]
        )[0]
        search_query = state.get("search_query") or state.get("entity_name")
        new_content = state.get("entity_content")

        if not new_content:
            return {
                "answer": "Укажите новое содержимое файла. Например: «Измени заметку Идеи: новый текст».",
                "agent_steps": agent_steps,
            }

        if not file_id and search_query:
            file_id = await self._find_file_id_by_name(config, user_id, search_query)

        if not file_id:
            hint = f" «{search_query}»" if search_query else ""
            return {
                "answer": f"Не нашёл файл{hint}. Уточните название.",
                "agent_steps": agent_steps,
            }

        title = search_query or "document"
        if not title.endswith(".md"):
            title = f"{title}.md"

        file_info = await self.file_service.replace_file_content(
            file_id=file_id,
            user_id=user_id,
            file_content=new_content.encode("utf-8"),
            filename=title,
        )
        return {
            "answer": f"Файл «{file_info.filename}» был отредактирован. Содержимое обновлено.",
            "agent_steps": agent_steps,
        }

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    async def _resolve_namespace_id(
        self, config: RunnableConfig, user_id: int, name: str
    ) -> Optional[int]:
        """Ищет namespace по имени (case-insensitive)."""
        try:
            db = (config.get("configurable") or {}).get("async_db")
            if not db:
                return None
            result = await db.execute(
                text(
                    "SELECT id FROM namespaces "
                    "WHERE user_id = :user_id AND LOWER(name) = LOWER(:name) LIMIT 1"
                ),
                {"user_id": user_id, "name": name},
            )
            row = result.mappings().first()
            return row["id"] if row else None
        except Exception as exc:
            logger.warning("[CrudNode] Failed to resolve namespace '%s': %s", name, exc)
            return None

    async def _find_file_id_by_name(
        self, config: RunnableConfig, user_id: int, name: str
    ) -> Optional[int]:
        """Ищет user_files.id по имени файла (ILIKE)."""
        try:
            db = (config.get("configurable") or {}).get("async_db")
            if not db:
                return None
            result = await db.execute(
                text(
                    "SELECT uf.id FROM user_files uf "
                    "JOIN files f ON f.id = uf.file_id "
                    "WHERE uf.user_id = :user_id "
                    "AND LOWER(COALESCE(uf.custom_title, f.media_metadata->>'title', '')) "
                    "ILIKE LOWER(:pattern) "
                    "ORDER BY uf.created_at DESC LIMIT 1"
                ),
                {"user_id": user_id, "pattern": f"%{name}%"},
            )
            row = result.mappings().first()
            return row["id"] if row else None
        except Exception as exc:
            logger.warning("[CrudNode] Failed to find file '%s': %s", name, exc)
            return None

    async def _get_filename(
        self, config: RunnableConfig, file_id: int, user_id: int
    ) -> Optional[str]:
        """Возвращает отображаемое имя файла."""
        try:
            db = (config.get("configurable") or {}).get("async_db")
            if not db:
                return None
            result = await db.execute(
                text(
                    "SELECT COALESCE(uf.custom_title, f.media_metadata->>'title', 'document') AS filename "
                    "FROM user_files uf JOIN files f ON f.id = uf.file_id "
                    "WHERE uf.id = :file_id AND uf.user_id = :user_id LIMIT 1"
                ),
                {"file_id": file_id, "user_id": user_id},
            )
            row = result.mappings().first()
            return row["filename"] if row else None
        except Exception as exc:
            logger.warning("[CrudNode] Failed to get filename for file_id=%s: %s", file_id, exc)
            return None
