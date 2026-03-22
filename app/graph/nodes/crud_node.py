"""CrudNode — выполнение CRUD-операций над пространствами и файлами.

Интенты:
- create_namespace: создать пространство
- delete_namespace: запросить подтверждение → set_pending_action
- move_file: переместить файл в пространство
- create_file: создать файл из текста
- delete_file: запросить подтверждение → set_pending_action
- edit_file: читает текущий текст файла, применяет инструкцию через LLM, сохраняет результат

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
from app.domain.protocols import LLMProvider, FileStorage, TaskPublisher
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
        llm_service: Optional[LLMProvider] = None,
        storage: Optional[FileStorage] = None,
        task_publisher: Optional[TaskPublisher] = None,
    ) -> None:
        self.file_service = file_service
        self.namespace_service = namespace_service
        self.llm_service = llm_service
        self.storage = storage
        self.task_publisher = task_publisher

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

    # Промпт для LLM, применяющего правку к тексту файла
    _EDIT_FILE_PROMPT = """\
ВАЖНО: Ты редактор текстовых файлов. Ты ДОЛЖЕН вернуть ПОЛНЫЙ отредактированный текст файла.
Применяй инструкцию внимательно. Возвращай ТОЛЬКО текст файла после изменений, без комментариев.

===== ТЕКУЩИЙ ТЕКСТ ФАЙЛА =====
{current_text}
===== КОНЕЦ ТЕКУЩЕГО ТЕКСТА =====

===== ИНСТРУКЦИЯ ПО РЕДАКТИРОВАНИЮ =====
{edit_instruction}
===== КОНЕЦ ИНСТРУКЦИИ =====

Теперь выполни редактирование и верни ПОЛНЫЙ результат:"""

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
        edit_instruction = state.get("entity_content")
        namespace_id = state.get("namespace_id")

        if not edit_instruction:
            return {
                "answer": "Укажите, что именно изменить в файле. Например: «Измени заметку Идеи: добавь Go в список языков».",
                "agent_steps": agent_steps,
            }

        if not file_id and search_query:
            file_id = await self._find_file_id_by_name(config, user_id, search_query)

        # Если конкретный файл не найден, но есть namespace — редактируем все файлы пространства
        if not file_id and namespace_id:
            file_ids = await self._find_all_file_ids_in_namespace(config, user_id, namespace_id)
            if not file_ids:
                return {
                    "answer": "В указанном пространстве не найдено файлов.",
                    "agent_steps": agent_steps,
                }
            
            # Запускаем фоновую задачу редактирования файлов
            if self.task_publisher:
                task_id = self.task_publisher.send_bulk_edit_task(
                    file_ids=file_ids,
                    user_id=user_id,
                    edit_instruction=edit_instruction,
                    namespace_id=namespace_id,
                )
                logger.info("[CrudNode] Bulk edit task started: task_id=%s files=%d", task_id, len(file_ids))
                return {
                    "answer": f"Начинаю редактирование {len(file_ids)} файлов пространства. Процесс запущен в фоне, это может занять несколько минут.",
                    "task_id": task_id,
                    "agent_steps": agent_steps,
                }
            
            # Fallback: синхронное редактирование если нет task_publisher
            logger.warning("[CrudNode] No task_publisher, falling back to sync bulk edit")
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

    async def _find_all_file_ids_in_namespace(
        self, config: RunnableConfig, user_id: int, namespace_id: int
    ) -> list[int]:
        """Возвращает все user_files.id пользователя в указанном пространстве."""
        try:
            db = (config.get("configurable") or {}).get("async_db")
            if not db:
                return []
            result = await db.execute(
                text(
                    "SELECT uf.id FROM user_files uf "
                    "WHERE uf.user_id = :user_id AND uf.namespace_id = :namespace_id "
                    "ORDER BY uf.created_at ASC"
                ),
                {"user_id": user_id, "namespace_id": namespace_id},
            )
            return [row["id"] for row in result.mappings().all()]
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
