"""CrudNode — выполнение CRUD-операций над пространствами и файлами.

Интенты:
- create_namespace: создать пространство
- delete_namespace: запросить подтверждение → set_pending_action
- move_file: переместить файл в пространство
- create_file: создать файл из текста
- delete_file: запросить подтверждение → set_pending_action
- edit_file: читает текущий текст файла, применяет инструкцию через LLM, сохраняет результат
- rename_file: меняет отображаемое имя (custom_title) у user_file

Для удаления CrudNode НЕ выполняет действие сразу — возвращает вопрос подтверждения
и просит ChatService сохранить pending_action. Фактическое удаление происходит в
ChatService при следующем сообщении с подтверждением (до вызова графа).
"""
import logging
import re
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from sqlalchemy import text

from app.graph.state import AskState
from app.core.enums import IntentType
from app.core.exceptions import NotFoundError, ForbiddenError, ValidationError
from app.domain.protocols import LLMProvider, FileStorage, TaskPublisher
from app.services.file_service import FileService
from app.services.namespace_service import NamespaceService
from app.services.content_extractor import ContentExtractorService
from app.utils.url import is_http_url_only
from app.graph.utils.namespace import resolve_namespace_id

logger = logging.getLogger(__name__)


class CrudNode:
    """Выполняет CRUD-операции над пространствами и файлами."""

    def __init__(
        self,
        *,
        file_service: FileService,
        namespace_service: Optional[NamespaceService] = None,
        llm_service: Optional[LLMProvider] = None,
        storage: Optional[FileStorage] = None,
        task_publisher: Optional[TaskPublisher] = None,
        content_extractor: Optional[ContentExtractorService] = None,
    ) -> None:
        self.file_service = file_service
        self.namespace_service = namespace_service
        self.llm_service = llm_service
        self.storage = storage
        self.task_publisher = task_publisher
        self.content_extractor = content_extractor

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
            elif intent == IntentType.EDIT_NAMESPACE_NAME:
                return await self._rename_namespace(state, user_id, agent_steps)
            elif intent == IntentType.EDIT_NAMESPACE_DESCRIPTION:
                return await self._update_namespace_description(state, user_id, agent_steps)
            elif intent == IntentType.MOVE_FILE:
                return await self._move_file(state, user_id, agent_steps, config)
            elif intent == IntentType.CREATE_FILE:
                return await self._create_file(state, user_id, agent_steps, config)
            elif intent == IntentType.SAVE_SUMMARY:
                return await self._save_summary_as_file(state, user_id, agent_steps, config)
            elif intent == IntentType.DELETE_FILE:
                return await self._request_delete_file(state, user_id, agent_steps, config)
            elif intent == IntentType.EDIT_FILE:
                return await self._edit_file(state, user_id, agent_steps, config)
            elif intent == IntentType.RENAME_FILE:
                return await self._rename_file(state, user_id, agent_steps, config)
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
            "created_namespace_id": namespace.id,
            "created_namespace_name": namespace.name,
            "agent_steps": agent_steps,
        }

    async def _edit_namespace(
        self, state: AskState, user_id: int, agent_steps: list
    ) -> dict[str, Any]:
        namespace_id = state.get("namespace_id")
        namespace_name = state.get("namespace_name_hint") or state.get("entity_name")
        new_description = state.get("entity_description")
        # Новое имя может прийти как entity_name (от Planner) или entity_content (legacy)
        new_name = state.get("entity_name") or state.get("entity_content")

        # Если LLM передал текущее имя в entity_name (при операции "добавь описание"),
        # не считаем это переименованием
        current_name = state.get("namespace_name_hint")
        if new_name and current_name and new_name.strip().lower() == current_name.strip().lower():
            new_name = None

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
            parts.append("описание обновлено")
        summary = ", ".join(parts)
        return {
            "answer": f"Пространство {summary}.",
            "agent_steps": agent_steps,
        }

    async def _rename_namespace(
        self, state: AskState, user_id: int, agent_steps: list
    ) -> dict[str, Any]:
        """Переименовать пространство (edit_namespace_name)."""
        namespace_id = state.get("namespace_id")
        current_name = state.get("namespace_name_hint")
        new_name = state.get("entity_name") or state.get("entity_content")

        if not namespace_id:
            hint = f" «{current_name}»" if current_name else ""
            return {"answer": f"Не нашёл пространство{hint}. Уточните название.", "agent_steps": agent_steps}
        if not self.namespace_service:
            return {"answer": "Сервис пространств недоступен.", "agent_steps": agent_steps}
        if not new_name:
            return {"answer": "Укажите новое название пространства.", "agent_steps": agent_steps}
        if new_name.strip().lower() == (current_name or "").strip().lower():
            return {"answer": f"Пространство уже называется «{current_name}».", "agent_steps": agent_steps}

        namespace = await self.namespace_service.update_namespace(
            namespace_id=namespace_id,
            user_id=user_id,
            name=new_name,
            description=None,
        )
        return {"answer": f"Пространство переименовано в «{namespace.name}».", "agent_steps": agent_steps}

    async def _update_namespace_description(
        self, state: AskState, user_id: int, agent_steps: list
    ) -> dict[str, Any]:
        """Обновить описание пространства (edit_namespace_description)."""
        namespace_id = state.get("namespace_id")
        current_name = state.get("namespace_name_hint")
        new_description = state.get("entity_description") or state.get("entity_content")

        if not namespace_id:
            hint = f" «{current_name}»" if current_name else ""
            return {"answer": f"Не нашёл пространство{hint}. Уточните название.", "agent_steps": agent_steps}
        if not self.namespace_service:
            return {"answer": "Сервис пространств недоступен.", "agent_steps": agent_steps}
        if new_description is None:
            return {"answer": "Укажите новое описание пространства.", "agent_steps": agent_steps}

        await self.namespace_service.update_namespace(
            namespace_id=namespace_id,
            user_id=user_id,
            name=None,
            description=new_description,
        )
        return {"answer": "Описание пространства обновлено.", "agent_steps": agent_steps}

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
        search_query = state.get("search_query")
        source_ns_name = state.get("entity_name") if not search_query else None

        # Bulk-перемещение: есть исходное пространство, нет конкретного файла — игнорируем history_file_id
        if source_ns_name and not search_query:
            file_id = None
        else:
            file_id = state.get("history_file_id") or (
                state.get("search_file_ids") or [None]
            )[0]

        if not namespace_id and namespace_name:
            namespace_id = await self._resolve_namespace_id(config, user_id, namespace_name)

        if not namespace_id:
            hint = f" «{namespace_name}»" if namespace_name else ""
            return {
                "answer": f"Не нашёл пространство назначения{hint}. Укажите корректное название.",
                "agent_steps": agent_steps,
            }

        if not file_id and not search_query and source_ns_name:
            source_ns_id = await self._resolve_namespace_id(config, user_id, source_ns_name)
            if not source_ns_id:
                return {
                    "answer": f"Не нашёл исходное пространство «{source_ns_name}». Уточните название.",
                    "agent_steps": agent_steps,
                }
            file_ids = await self._find_all_file_ids_in_namespace(user_id, source_ns_id)
            if not file_ids:
                return {
                    "answer": f"В пространстве «{source_ns_name}» нет файлов.",
                    "agent_steps": agent_steps,
                }
            moved = []
            errors = []
            for fid in file_ids:
                try:
                    info = await self.file_service.move_to_namespace(
                        file_id=fid, namespace_id=namespace_id, user_id=user_id
                    )
                    moved.append(info.filename)
                except Exception as exc:
                    logger.warning("[CrudNode] Failed to move file_id=%d: %s", fid, exc)
                    errors.append(str(fid))
            ns_display = f"«{namespace_name}»" if namespace_name else f"(id={namespace_id})"
            result_msg = f"Перемещено {len(moved)} файл(ов) из «{source_ns_name}» в {ns_display}."
            if errors:
                result_msg += f" Не удалось переместить: {', '.join(errors)}."
            return {"answer": result_msg, "agent_steps": agent_steps}

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

    async def _save_summary_as_file(
        self, state: AskState, user_id: int, agent_steps: list, config: RunnableConfig = None
    ) -> dict[str, Any]:
        """Сохраняет последнее саммари из истории или pipeline_context как файл."""
        # 1. Приоритет: entity_content, выставленный multi_action pipeline
        content = state.get("entity_content")
        title = state.get("entity_name")

        # 2. Если нет — ищем в истории последнее сообщение ассистента с достаточным текстом
        if not content:
            history = state.get("history") or []
            for msg in reversed(history):
                if msg.get("role") == "assistant":
                    text = (msg.get("text") or "").strip()
                    # Берём первое непустое сообщение ассистента длиннее 100 символов
                    if len(text) > 100:
                        content = text
                        break

        if not content:
            return {
                "answer": (
                    "Не нашёл саммари для сохранения. "
                    "Сначала попросите «суммаризируй файл», затем — «сохрани это»."
                ),
                "agent_steps": agent_steps,
            }

        # Делегируем в _create_file, подставив content
        patched_state = {**state, "entity_content": content, "entity_name": title}
        return await self._create_file(patched_state, user_id, agent_steps, config)

    async def _create_file(
        self, state: AskState, user_id: int, agent_steps: list, config: RunnableConfig = None
    ) -> dict[str, Any]:
        content = state.get("entity_content")
        title = state.get("entity_name")
        namespace_id = state.get("namespace_id")

        # Если content — это URL, загружаем содержимое страницы
        url_to_fetch: Optional[str] = None
        if content and is_http_url_only(content):
            url_to_fetch = content.strip()
        elif not content:
            question = state.get("question") or ""
            detected = state.get("detected_url")
            candidate = detected or (question.strip() if is_http_url_only(question) else None)
            if candidate:
                url_to_fetch = candidate

        if url_to_fetch:
            if not self.content_extractor:
                return {
                    "answer": "Не могу загрузить страницу: сервис извлечения контента недоступен.",
                    "agent_steps": agent_steps,
                }
            try:
                logger.info("[CrudNode] create_file: fetching URL %s", url_to_fetch)
                parsed = await self.content_extractor.extract(url_to_fetch)
                content = parsed.text
                if not title:
                    title = parsed.title
                logger.info(
                    "[CrudNode] create_file: fetched URL %s → title=%s len=%d",
                    url_to_fetch, parsed.title, len(content or ""),
                )
            except Exception as exc:
                logger.warning("[CrudNode] create_file: failed to fetch URL %s: %s", url_to_fetch, exc)
                return {
                    "answer": f"Не удалось загрузить страницу: {exc}",
                    "agent_steps": agent_steps,
                }

        if not content:
            return {
                "answer": "Укажите содержимое файла. Например: «Создай заметку Идеи: текст заметки».",
                "agent_steps": agent_steps,
            }

        # Если пространство не указано — пытаемся найти Inbox как дефолтное
        # Но если пользователь явно указал пространство (namespace_name_hint) и оно не нашлось — сообщаем
        namespace_name = state.get("namespace_name_hint")
        if namespace_id is None and namespace_name:
            return {
                "answer": (
                    f"Пространство «{namespace_name}» не найдено. "
                    "Проверьте название или создайте его командой «Создай пространство ...»."
                ),
                "agent_steps": agent_steps,
            }
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

        if self.task_publisher and file_created.is_new_file and file_created.text:
            self.task_publisher.send_embeddings_task(
                content_file_id=file_created.content_file_id,
                text=file_created.text,
                namespace_id=namespace_id,
                filename=file_created.filename,
                user_file_id=file_created.file_id,
            )
            logger.info(
                "[CrudNode] create_file: sent embeddings task for file_id=%d '%s'",
                file_created.file_id, file_created.filename,
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
        """Находит файл(ы), возвращает вопрос подтверждения и сохраняет pending_action."""
        search_query = state.get("search_query") or state.get("entity_name")
        namespace_id = state.get("namespace_id")
        namespace_name = state.get("namespace_name_hint")

        # Если задан namespace без конкретного файла — режим namespace, игнорируем history_file_id
        if namespace_id and not search_query:
            file_id = None
        else:
            file_id = state.get("history_file_id") or (
                state.get("search_file_ids") or [None]
            )[0]

        # Случай «удали файлы из пространства X» — нет конкретного файла, но есть namespace
        if not file_id and not search_query and namespace_id:
            all_ids = await self._find_all_file_ids_in_namespace(user_id, namespace_id)
            if not all_ids:
                ns_label = f"«{namespace_name}»" if namespace_name else f"(id={namespace_id})"
                return {
                    "answer": f"В пространстве {ns_label} нет файлов.",
                    "agent_steps": agent_steps,
                }

            search_limit = state.get("search_limit")
            ns_label = f"«{namespace_name}»" if namespace_name else f"(id={namespace_id})"

            # "удали любой/один файл" — берём только первые N
            if search_limit and search_limit < len(all_ids):
                target_ids = all_ids[:search_limit]
                filenames = []
                for fid in target_ids:
                    fn = await self._get_filename(config, fid, user_id)
                    filenames.append(fn or f"id={fid}")
                names_str = ", ".join(f"«{n}»" for n in filenames)
                pending = {
                    "type": "delete_all_in_namespace",
                    "params": {"namespace_id": namespace_id, "file_ids": target_ids},
                    "target": f"файл(ы) {names_str} из пространства {ns_label}",
                }
                return {
                    "answer": (
                        f"Вы уверены, что хотите удалить {names_str} из пространства {ns_label}? "
                        "Это действие нельзя отменить. Напишите «да» для подтверждения."
                    ),
                    "pending_action": pending,
                    "agent_steps": agent_steps,
                }

            count = len(all_ids)
            pending = {
                "type": "delete_all_in_namespace",
                "params": {"namespace_id": namespace_id, "file_ids": all_ids},
                "target": f"{count} файл(ов) из пространства {ns_label}",
            }
            return {
                "answer": (
                    f"Вы уверены, что хотите удалить все {count} файл(ов) из пространства {ns_label}? "
                    "Это действие нельзя отменить. Напишите «да» для подтверждения."
                ),
                "pending_action": pending,
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

    @staticmethod
    def _parse_bulk_new_names(entity_content: Optional[str]) -> list[str]:
        if not entity_content or not str(entity_content).strip():
            return []
        t = str(entity_content).strip().replace(" и ", ",")
        parts = re.split(r"[,;]", t)
        return [p.strip() for p in parts if p.strip()]

    async def _rename_file(
        self,
        state: AskState,
        user_id: int,
        agent_steps: list,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """Переименование отображаемого имени файла (custom_title)."""
        search_query = state.get("search_query")
        new_name_single = (state.get("entity_name") or "").strip() or None
        entity_content_raw = state.get("entity_content")
        namespace_id = state.get("namespace_id")

        configurable = config.get("configurable") or {}
        user_file_repository = configurable.get("user_file_repository")
        if not user_file_repository:
            return {"answer": "Ошибка: репозиторий файлов недоступен.", "agent_steps": agent_steps}

        bulk_names = self._parse_bulk_new_names(
            entity_content_raw if isinstance(entity_content_raw, str) else None
        )

        # Пакетное переименование: список имён в entity_content, без одного нового имени в entity_name
        if bulk_names and not new_name_single:
            if not namespace_id:
                return {
                    "answer": (
                        "Чтобы переименовать несколько файлов, укажите пространство в сообщении "
                        "или откройте его в интерфейсе."
                    ),
                    "agent_steps": agent_steps,
                }
            file_ids = await self._find_all_file_ids_in_namespace(user_id, namespace_id)
            if not file_ids:
                return {"answer": "В этом пространстве нет файлов.", "agent_steps": agent_steps}
            if len(bulk_names) != len(file_ids):
                return {
                    "answer": (
                        f"В пространстве {len(file_ids)} файл(ов), в сообщении — {len(bulk_names)} имён. "
                        "Перечислите новые имена через запятую в том же порядке, что и файлы в пространстве."
                    ),
                    "agent_steps": agent_steps,
                }
            done: list[str] = []
            for fid, raw_title in zip(file_ids, bulk_names):
                safe = FileService.sanitize_filename(raw_title)
                if not safe:
                    return {
                        "answer": f"Недопустимое имя: «{raw_title}».",
                        "agent_steps": agent_steps,
                    }
                await user_file_repository.update_custom_title(fid, safe)
                done.append(safe)
            return {
                "answer": "Переименовано: " + ", ".join(f"«{n}»" for n in done) + ".",
                "agent_steps": agent_steps,
            }

        if not new_name_single:
            return {
                "answer": "Укажите новое имя файла или список имён через запятую.",
                "agent_steps": agent_steps,
            }

        safe_new = FileService.sanitize_filename(new_name_single)
        if not safe_new:
            return {"answer": "Недопустимое новое имя файла.", "agent_steps": agent_steps}

        file_id: Optional[int] = state.get("history_file_id") or (
            (state.get("search_file_ids") or [None])[0]
        )
        if search_query:
            file_id = await self._find_file_id_by_name(config, user_id, search_query)
        elif not file_id and namespace_id:
            all_ids = await self._find_all_file_ids_in_namespace(user_id, namespace_id)
            if len(all_ids) == 1:
                file_id = all_ids[0]
            elif len(all_ids) > 1:
                return {
                    "answer": (
                        "В пространстве несколько файлов — укажите текущее имя файла, "
                        "который нужно переименовать."
                    ),
                    "agent_steps": agent_steps,
                }

        if not file_id:
            hint = f" «{search_query}»" if search_query else ""
            return {
                "answer": f"Не нашёл файл{hint}. Уточните название.",
                "agent_steps": agent_steps,
            }

        uf = await user_file_repository.get_by_id(file_id)
        if not uf or uf.user_id != user_id:
            return {"answer": "Файл не найден или нет доступа.", "agent_steps": agent_steps}

        await user_file_repository.update_custom_title(file_id, safe_new)
        return {
            "answer": f"Файл переименован в «{safe_new}».",
            "agent_steps": agent_steps,
        }

    # Промпт для LLM, применяющего правку к тексту файла
    _EDIT_FILE_PROMPT = """\
ВАЖНО: Ты редактор текстовых файлов. Ты ДОЛЖЕН вернуть ПОЛНЫЙ отредактированный текст файла.
Возвращай ТОЛЬКО текст файла после изменений — никаких пояснений, комментариев, приветствий.

Правила выполнения инструкции:
- "удали последнюю строку" — убери последнюю непустую строку текста, остальное оставь
- "удали первую строку" — убери первую строку текста, остальное оставь
- "удали строку с X" — найди строку содержащую X и убери её
- "добавь в конец: Y" — добавь Y как новую строку в конце текста
- "добавь в начало: Y" — добавь Y как новую строку в начале текста
- "замени X на Y" — замени все вхождения X на Y
Если инструкция говорит "удали" — соответствующий фрагмент ДОЛЖЕН ОТСУТСТВОВАТЬ в результате.

===== ТЕКУЩИЙ ТЕКСТ ФАЙЛА =====
{current_text}
===== КОНЕЦ ТЕКУЩЕГО ТЕКСТА =====

===== ИНСТРУКЦИЯ ПО РЕДАКТИРОВАНИЮ =====
{edit_instruction}
===== КОНЕЦ ИНСТРУКЦИИ =====

Выполни инструкцию и верни ПОЛНЫЙ отредактированный текст файла:"""

    # Числа прописью для парсинга "удали три последние строки"
    _RUS_NUMBERS = {
        'одну': 1, 'одной': 1, 'один': 1, 'одно': 1,
        'две': 2, 'двух': 2, 'двум': 2,
        'три': 3, 'трёх': 3, 'трех': 3,
        'четыре': 4, 'четырёх': 4, 'четырех': 4,
        'пять': 5, 'пяти': 5,
        'шесть': 6, 'шести': 6,
        'семь': 7, 'семи': 7,
        'восемь': 8, 'восьми': 8,
        'девять': 9, 'девяти': 9,
        'десять': 10, 'десяти': 10,
    }

    # Простые операции которые выполняются детерминированно без вызова LLM
    _SIMPLE_EDIT_PATTERNS = [
        (re.compile(r'удали?\s+последнюю?\s+строку', re.IGNORECASE),
         lambda t, _m: '\n'.join(t.rstrip('\n').split('\n')[:-1])),
        (re.compile(r'удали?\s+первую?\s+строку', re.IGNORECASE),
         lambda t, _m: '\n'.join(t.split('\n')[1:])),
        (re.compile(r'удали?\s+пустые\s+строки', re.IGNORECASE),
         lambda t, _m: '\n'.join(line for line in t.split('\n') if line.strip())),
        # "удали три последние строки", "удали 3 последних строки"
        (re.compile(
            r'удали?\s+(\d+|одну?|дв[ае]|три|четыре|пять|шесть|семь|восемь|девять|десять)\s+последни[хе]\s+строк',
            re.IGNORECASE,
        ), None),  # обработчик устанавливается динамически в _try_simple_edit
    ]

    def _try_simple_edit(self, text: str, instruction: str) -> Optional[str]:
        """Пытается выполнить простую операцию детерминированно. Возвращает None если не распознано."""
        for pattern, apply in self._SIMPLE_EDIT_PATTERNS:
            m = pattern.search(instruction)
            if not m:
                continue
            if apply is not None:
                return apply(text, m)
            # Динамическая обработка для "N последних строк"
            raw = m.group(1)
            n = int(raw) if raw.isdigit() else self._RUS_NUMBERS.get(raw.lower(), 1)
            lines = text.rstrip('\n').split('\n')
            if n >= len(lines):
                return ''
            return '\n'.join(lines[:-n])
        return None

    async def _edit_file(
        self,
        state: AskState,
        user_id: int,
        agent_steps: list,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        search_query = state.get("search_query") or state.get("entity_name")
        edit_instruction = state.get("entity_content")
        namespace_id = state.get("namespace_id")

        # Если задан namespace без конкретного файла — режим bulk-edit: игнорируем history_file_id
        # (пользователь хочет редактировать все файлы пространства, а не тот что открыт)
        if namespace_id and not search_query:
            file_id = None
        else:
            file_id = state.get("history_file_id") or (
                state.get("search_file_ids") or [None]
            )[0]

        if not edit_instruction:
            return {
                "answer": "Укажите, что именно изменить в файле. Например: «Измени заметку Идеи: добавь Go в список языков».",
                "agent_steps": agent_steps,
            }

        if not file_id and search_query:
            file_id = await self._find_file_id_by_name(config, user_id, search_query)

        # Если конкретный файл не найден, но есть namespace — редактируем все файлы пространства
        if not file_id and namespace_id:
            file_ids = await self._find_all_file_ids_in_namespace(user_id, namespace_id)
            if not file_ids:
                return {
                    "answer": "В указанном пространстве не найдено файлов.",
                    "agent_steps": agent_steps,
                }
            
            # Синхронное редактирование всех файлов пространства
            logger.info("[CrudNode] Starting sync bulk edit: files=%d", len(file_ids))
            edited_names: list[str] = []
            failed_names: list[str] = []
            for fid in file_ids:
                result = await self._edit_single_file(
                    file_id=fid,
                    edit_instruction=edit_instruction,
                    user_id=user_id,
                    config=config,
                )
                if result.get("ok"):
                    edited_names.append(result["filename"])
                else:
                    failed_names.append(result.get("filename") or f"id={fid}")
            parts: list[str] = []
            if edited_names:
                parts.append(f"Отредактированы файлы: {', '.join(f'«{n}»' for n in edited_names)}.")
            if failed_names:
                parts.append(f"Не удалось изменить: {', '.join(f'«{n}»' for n in failed_names)}.")
            return {"answer": " ".join(parts), "agent_steps": agent_steps}

        if not file_id:
            hint = f" «{search_query}»" if search_query else ""
            return {
                "answer": f"Не нашёл файл{hint}. Уточните название.",
                "agent_steps": agent_steps,
            }

        result = await self._edit_single_file(
            file_id=file_id,
            edit_instruction=edit_instruction,
            user_id=user_id,
            config=config,
        )
        if result.get("ok"):
            return {
                "answer": f"Файл «{result['filename']}» был отредактирован. Содержимое обновлено.",
                "agent_steps": agent_steps,
            }
        return {"answer": result.get("error", "Ошибка при редактировании файла."), "agent_steps": agent_steps}

    async def _edit_single_file(
        self,
        *,
        file_id: int,
        edit_instruction: str,
        user_id: int,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """Редактирует один файл. Возвращает {'ok': True, 'filename': ...} или {'ok': False, 'error': ..., 'filename': ...}."""
        configurable = config.get("configurable") or {}
        file_repository = configurable.get("file_repository")
        user_file_repository = configurable.get("user_file_repository")
        storage = self.storage or configurable.get("storage")

        if not file_repository or not user_file_repository or not storage:
            return {"ok": False, "error": "Ошибка: не удалось получить доступ к хранилищу файлов."}

        # Получаем путь к файлу в MinIO
        user_file = await user_file_repository.get_by_id(file_id)
        if not user_file:
            return {"ok": False, "error": f"Файл (id={file_id}) не найден.", "filename": f"id={file_id}"}
        if user_file.user_id != user_id:
            return {"ok": False, "error": "У вас нет доступа к этому файлу.", "filename": f"id={file_id}"}

        content_file = await file_repository.get_by_id(user_file.file_id)
        if not content_file or not content_file.file_path:
            return {"ok": False, "error": "Не удалось найти содержимое файла в хранилище.", "filename": f"id={file_id}"}

        # Скачиваем текущее содержимое
        try:
            file_bytes = await storage.download_file(content_file.file_path)
        except Exception as exc:
            logger.warning("[CrudNode] Failed to download file %s: %s", content_file.file_path, exc)
            return {"ok": False, "error": "Не удалось скачать файл из хранилища.", "filename": f"id={file_id}"}

        # Определяем расширение и извлекаем текст
        meta = content_file.media_metadata or {}
        file_ext = meta.get("file_type", "md")
        filename = user_file.custom_title or meta.get("title", "document")

        # Запрещаем редактирование PDF файлов
        if file_ext.lower() == "pdf":
            return {"ok": False, "error": "Редактирование PDF файлов не поддерживается.", "filename": filename}

        from app.utils.file_readers import FileReaderFactory
        try:
            reader = FileReaderFactory().get_reader(file_ext)
            current_text = reader.read(file_bytes)
        except Exception:
            # Если парсер не справился, пробуем как plain text
            current_text = file_bytes.decode("utf-8", errors="replace")

        if not current_text or not current_text.strip():
            return {"ok": False, "error": "Файл пуст — нечего редактировать.", "filename": filename}

        # Применяем инструкцию через LLM
        if not self.llm_service:
            return {"ok": False, "error": "Ошибка: LLM-сервис недоступен для применения правки.", "filename": filename}

        # Ограничиваем текст файла, чтобы не превысить контекст LLM
        max_text_chars = 12000
        truncated = len(current_text) > max_text_chars
        text_for_llm = current_text[:max_text_chars] if truncated else current_text

        logger.info("[CrudNode] Editing file %s (ext=%s): instr=%r | truncated=%s original_len=%d text_for_llm_len=%d", 
                    filename, file_ext, edit_instruction, truncated, len(current_text), len(text_for_llm))

        # Пробуем выполнить простую операцию без LLM
        simple_result = self._try_simple_edit(current_text, edit_instruction)
        if simple_result is not None:
            edited_text = simple_result
            logger.info(
                "[CrudNode] Simple edit applied for file %s: orig_len=%d result_len=%d",
                filename, len(current_text), len(edited_text),
            )
        else:
            prompt = self._EDIT_FILE_PROMPT.format(
                current_text=text_for_llm,
                edit_instruction=edit_instruction,
            )
            if truncated:
                prompt += "\n\n[ВНИМАНИЕ: текст файла обрезан до первых 12000 символов. Остальной текст оставь без изменений.]"

            try:
                edited_text = await self.llm_service.complete(
                    [
                        {"role": "system", "text": "Ты редактор текстовых файлов. Возвращай ТОЛЬКО отредактированный текст файла, ничего больше."},
                        {"role": "user", "text": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=16384,
                )
                logger.info(
                    "[CrudNode] LLM returned for file %s: result_len=%d (from orig_len=%d, sent=%d) | changed=%s",
                    filename, len(edited_text or ""), len(current_text), len(text_for_llm),
                    edited_text != current_text if edited_text else False
                )
            except Exception as exc:
                logger.exception("[CrudNode] LLM edit call failed: %s", exc)
                return {"ok": False, "error": "Ошибка при применении правки через LLM.", "filename": filename}

            if not edited_text or not edited_text.strip():
                return {"ok": False, "error": "LLM вернул пустой результат. Файл не изменён.", "filename": filename}

            # Если текст был обрезан — присоединяем хвост оригинала
            if truncated:
                tail_len = len(current_text[max_text_chars:])
                edited_text = edited_text + current_text[max_text_chars:]
                logger.info("[CrudNode] Reappended tail: tail_len=%d final_len=%d", tail_len, len(edited_text))

        # Записываем отредактированный текст обратно в оригинальном формате
        if file_ext == "docx":
            # Пересобираем .docx из отредактированного текста
            import io
            from docx import Document as DocxDocument
            doc = DocxDocument()
            for paragraph_text in edited_text.split("\n"):
                doc.add_paragraph(paragraph_text)
            buf = io.BytesIO()
            doc.save(buf)
            save_content = buf.getvalue()
            save_filename = filename
            if not save_filename.lower().endswith(".docx"):
                save_filename = save_filename.rsplit(".", 1)[0] + ".docx" if "." in save_filename else save_filename + ".docx"
            logger.info("[CrudNode] Converting to DOCX for file %s: edited_text_len=%d → saved_content_bytes=%d", 
                        filename, len(edited_text), len(save_content))
        elif file_ext == "pdf":
            # PDF нельзя легко пересобрать — сохраняем как .md
            save_content = edited_text.encode("utf-8")
            save_filename = filename.rsplit(".", 1)[0] + ".md" if "." in filename else filename + ".md"
            if not save_filename.endswith(".md"):
                save_filename = f"{save_filename}.md"
            logger.info("[CrudNode] Converting PDF to MD for file %s: edited_text_len=%d → saved_filename=%s saved_bytes=%d", 
                        filename, len(edited_text), save_filename, len(save_content))
        else:
            save_content = edited_text.encode("utf-8")
            save_filename = filename
            if not save_filename.endswith(f".{file_ext}"):
                save_filename = f"{save_filename}.{file_ext}"
            logger.info("[CrudNode] Saving %s for file %s: edited_text_len=%d → saved_bytes=%d", 
                        file_ext, filename, len(edited_text), len(save_content))

        file_info = await self.file_service.replace_file_content(
            file_id=file_id,
            user_id=user_id,
            file_content=save_content,
            filename=save_filename,
        )
        logger.info("[CrudNode] File saved via replace_file_content: file_id=%d → new_filename=%s", file_id, file_info.filename)
        return {"ok": True, "filename": file_info.filename}

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    async def _resolve_namespace_id(
        self, config: RunnableConfig, user_id: int, name: str
    ) -> Optional[int]:
        """Ищет namespace по имени (case-insensitive)."""
        db = (config.get("configurable") or {}).get("async_db")
        if not db:
            return None
        return await resolve_namespace_id(db, user_id, name)

    async def _find_all_file_ids_in_namespace(
        self, user_id: int, namespace_id: int
    ) -> list[int]:
        """Возвращает все user_files.id пользователя в указанном пространстве."""
        try:
            return await self.file_service.list_user_file_ids_in_namespace(
                user_id, namespace_id
            )
        except Exception as exc:
            logger.warning(
                "[CrudNode] Failed to list files in namespace %s: %s", namespace_id, exc
            )
            return []

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
