"""MultiActionNode — последовательное выполнение нескольких действий за один запрос.

Поддерживает два вида действий:
- CRUD (create_namespace, create_file, edit_file, delete_*, move_file) — через CrudNode
- Pipeline (index_url, summarize) — через IndexUrlNode / SummaryNode

Pipeline-действия передают данные следующим шагам через pipeline_context:
  index_url → pipeline_context["history_file_id"] (сохранённый user_file.id)
  summarize → pipeline_context["entity_content"] (текст саммари)
              pipeline_context["entity_name_fallback"] (заголовок источника)

Результат передаётся в state["pipeline_report"] для последующей обработки MindBuddyAgent.
"""
import logging
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig

from app.graph.state import AskState
from app.graph.sub_state_builder import build_sub_state
from app.graph.nodes.crud_node import CrudNode
from app.graph.nodes.generate_content_node import GenerateContentNode
from app.graph.nodes.file_agent import FileAgent
from app.graph.nodes.save_file_node import SaveFileNode

logger = logging.getLogger(__name__)

# Интенты, которые маршрутизируются через pipeline (не CrudNode)
_PIPELINE_INTENTS = {"index_url", "summarize", "save_file"}


def _deduplicate_create_file_names(actions: List[dict]) -> List[dict]:
    """
    Если несколько create_file шагов имеют одинаковый непустой entity_name,
    добавляет числовой суффикс: «Совет» → «Совет 1», «Совет 2», «Совет 3».
    Шаги с entity_name=None не трогает (timestamp-имя гарантирует уникальность).
    """
    from collections import Counter

    name_counts: Counter = Counter(
        a["entity_name"]
        for a in actions
        if a.get("intent") == "create_file" and a.get("entity_name")
    )
    duplicates = {name for name, count in name_counts.items() if count > 1}
    if not duplicates:
        return actions

    seen: dict[str, int] = {}
    result = []
    for action in actions:
        if action.get("intent") == "create_file":
            name = action.get("entity_name") or ""
            if name in duplicates:
                seen[name] = seen.get(name, 0) + 1
                action = {**action, "entity_name": f"{name} {seen[name]}"}
        result.append(action)
    return result


def _expand_summarize_create_file(
    actions: List[dict], search_file_ids: List[int]
) -> List[dict]:
    """
    Обрабатывает пары summarize → create_file когда в state несколько file_ids.

    Два сценария:
    - LLM сгенерировал 1 пару → разворачиваем в N пар с _target_file_id
    - LLM уже сгенерировал N пар (знал о нескольких файлах) → только назначаем
      _target_file_id каждому summarize по порядку, без дублирования

    Пример (1 пара): [create_ns, summarize, create_file] + file_ids=[1,2,3]
    →  [create_ns, summarize(fid=1), create_file,
                   summarize(fid=2), create_file,
                   summarize(fid=3), create_file]

    Пример (N пар): [create_ns, summarize, create_file, summarize, create_file, summarize, create_file]
    → просто назначаем fid=1,2,3 соответствующим summarize шагам
    """
    if len(search_file_ids) <= 1:
        return actions

    summarize_count = sum(1 for a in actions if a.get("intent") == "summarize")

    if summarize_count > 1:
        # LLM уже раскрыл план — назначаем file_ids по порядку
        fid_iter = iter(search_file_ids)
        result = []
        for act in actions:
            if act.get("intent") == "summarize":
                fid = next(fid_iter, None)
                if fid is not None:
                    act = {**act, "_target_file_id": fid}
            result.append(act)
        return result

    # Ищем первую и единственную пару summarize → create_file
    summarize_idx = None
    for i, act in enumerate(actions):
        if act.get("intent") == "summarize":
            if i + 1 < len(actions) and actions[i + 1].get("intent") == "create_file":
                summarize_idx = i
                break

    if summarize_idx is None:
        return actions

    summarize_action = actions[summarize_idx]
    create_file_action = actions[summarize_idx + 1]
    before = actions[:summarize_idx]
    after = actions[summarize_idx + 2:]

    expanded = []
    for fid in search_file_ids:
        expanded.append({**summarize_action, "_target_file_id": fid})
        expanded.append(dict(create_file_action))

    return before + expanded + after


class MultiActionNode:
    """
    Выполняет список действий из state["pending_actions"] последовательно.

    CRUD-действия → CrudNode.
    Pipeline-действия (index_url, summarize) → IndexUrlNode / SummaryNode.
    Данные между шагами передаются через pipeline_context.
    Результат записывается в pipeline_report — MindBuddyAgent формирует финальный ответ.
    """

    def __init__(
        self,
        crud_node: CrudNode,
        index_url_node=None,             # Optional[IndexUrlNode]
        summary_node=None,               # Optional[SummaryNode]
        generate_content_node=None,      # Optional[GenerateContentNode]
        file_agent: Optional[FileAgent] = None,
        save_file_node: Optional[SaveFileNode] = None,
    ) -> None:
        self.crud_node = crud_node
        self.index_url_node = index_url_node
        self.summary_node = summary_node
        self.generate_content_node: Optional[GenerateContentNode] = generate_content_node
        self.file_agent = file_agent
        self.save_file_node = save_file_node

    async def run(self, state: AskState, config: RunnableConfig) -> dict[str, Any]:
        actions = state.get("pending_actions") or []
        if not actions:
            return {
                "pipeline_report": [{"step": "multi_action", "ok": False, "message": "Не указаны действия для выполнения."}],
                "agent_steps": list(state.get("agent_steps") or []),
            }

        # Разворачиваем summarize → create_file пары если несколько файлов
        search_file_ids: List[int] = state.get("search_file_ids") or []
        actions = _expand_summarize_create_file(actions, search_file_ids)
        # Если несколько create_file шагов имеют одинаковое имя — нумеруем их
        actions = _deduplicate_create_file_names(actions)

        all_steps = list(state.get("agent_steps") or [])
        total = len(actions)

        # Пространства, созданные в рамках этого же батча: name (lower) → id
        created_ns: dict[str, int] = {}

        # Данные, которые pipeline-шаги передают друг другу
        pipeline_context: dict[str, Any] = {}

        # Отчёт о выполнении шагов для MindBuddyAgent
        pipeline_report: list[dict[str, Any]] = []

        # Pending-действия от delete-операций — объединим в одно подтверждение
        pending_deletes: list[dict] = []
        delete_targets: list[str] = []

        for i, action in enumerate(actions):
            intent = action.get("intent") or ""

            # Резолвим namespace: сначала из батча, потом из действия
            ns_id = action.get("namespace_id")
            ns_hint = (action.get("namespace_name_hint") or "").strip()
            if ns_id is None and ns_hint:
                ns_id = created_ns.get(ns_hint.lower())
                if ns_id is not None:
                    logger.info(
                        "[MultiActionNode] action %d/%d: resolved ns '%s' → id=%d (from batch)",
                        i + 1, total, ns_hint, ns_id,
                    )

            base_step = f"[MultiActionNode] action {i + 1}/{total}: {intent}"
            # ns_id из батча/action перезаписывает UI только если не None;
            # иначе builder скопирует namespace_id из state через COMMON
            ns_overrides: dict[str, Any] = {"namespace_id": ns_id} if ns_id is not None else {}
            sub_state: AskState = build_sub_state(
                state, intent,
                namespace_name_hint=ns_hint or action.get("namespace_name_hint"),
                search_query=action.get("search_query"),
                search_limit=action.get("search_limit"),
                entity_name=action.get("entity_name"),
                entity_description=action.get("entity_description"),
                entity_content=action.get("entity_content"),
                answer=None,
                agent_steps=all_steps + [base_step],
                **ns_overrides,
            )

            try:
                if intent == "index_url":
                    result = await self._run_index_url(sub_state, config, pipeline_context)
                elif intent == "summarize":
                    result = await self._run_summarize(sub_state, config, pipeline_context, action)
                elif intent == "save_file":
                    result = await self._run_save_file(sub_state, config)
                else:
                    result = await self._run_crud(sub_state, config, pipeline_context, action)

                all_steps = result.get("agent_steps", all_steps)

                # Собираем file_id от save_file шагов — для записи в историю сообщения
                if intent == "save_file" and result.get("file_id") is not None:
                    pipeline_context.setdefault("saved_file_ids", []).append(result["file_id"])

                # Delete-действия возвращают pending_action — собираем в батч
                pa = result.get("pending_action")
                if pa:
                    pending_deletes.append(pa)
                    delete_targets.append(pa.get("target", "объект"))
                    pipeline_report.append({"step": intent, "ok": True, "message": pa.get("target", "")})
                else:
                    message = (result.get("answer") or "").strip()
                    pipeline_report.append({"step": intent, "ok": True, "message": message})

                # Запоминаем только что созданное пространство
                new_ns_id = result.get("created_namespace_id")
                new_ns_name = result.get("created_namespace_name")
                if new_ns_id and new_ns_name:
                    created_ns[new_ns_name.lower()] = new_ns_id
                    logger.info(
                        "[MultiActionNode] Registered new namespace '%s' → id=%d",
                        new_ns_name, new_ns_id,
                    )

            except Exception:
                logger.exception(
                    "[MultiActionNode] Error on action %d/%d: %s", i + 1, total, action
                )
                pipeline_report.append({"step": intent, "ok": False, "message": f"Ошибка при выполнении шага {i + 1}."})

        output: dict[str, Any] = {
            "pipeline_report": pipeline_report,
            "agent_steps": all_steps + [f"[MultiActionNode] Completed {total} actions"],
        }

        # Если в батче было создано новое пространство — пробрасываем ID наружу
        # (chat_service использует его для маршрутизации дополнительных файлов)
        if created_ns:
            last_ns_name = list(created_ns.keys())[-1]
            output["created_namespace_id"] = created_ns[last_ns_name]
            output["created_namespace_name"] = last_ns_name

        # Пробрасываем file_ids от save_file шагов для записи в историю сообщения
        saved_file_ids = pipeline_context.get("saved_file_ids") or []
        if saved_file_ids:
            output["file_ids"] = saved_file_ids

        # Pending-действия от delete-операций — передаём для подтверждения
        if len(pending_deletes) == 1:
            output["pending_action"] = pending_deletes[0]
        elif len(pending_deletes) > 1:
            output["pending_action"] = {"type": "batch_delete", "items": pending_deletes}

        return output

    # ------------------------------------------------------------------
    # Pipeline-шаги
    # ------------------------------------------------------------------

    async def _run_index_url(
        self,
        sub_state: AskState,
        config: RunnableConfig,
        pipeline_context: dict,
    ) -> dict[str, Any]:
        """Шаг 1 pipeline: загрузить URL, сохранить файл, записать file_id в контекст."""
        if not self.index_url_node:
            return {
                "answer": "Индексация URL недоступна (сервис не настроен).",
                "agent_steps": sub_state.get("agent_steps", []),
            }
        result = await self.index_url_node.run(sub_state, config)
        file_id = result.get("file_id")
        if file_id:
            pipeline_context["history_file_id"] = file_id
            logger.info("[MultiActionNode] index_url: captured file_id=%d", file_id)
        return result

    async def _user_file_ids_in_namespace_for_summarize(
        self,
        config: RunnableConfig,
        user_id: Optional[int],
        namespace_id: Optional[int],
    ) -> list[int]:
        """Те же правила, что в SummaryNode: user_id + namespace_id, порядок по created_at."""
        if user_id is None or namespace_id is None:
            return []
        file_service = ((config or {}).get("configurable") or {}).get("file_service")
        if not file_service:
            logger.warning(
                "[MultiActionNode] file_service отсутствует в config — "
                "не удаётся получить список файлов пространства для summarize"
            )
            return []
        return await file_service.list_user_file_ids_in_namespace(
            user_id, namespace_id
        )

    async def _run_summarize(
        self,
        sub_state: AskState,
        config: RunnableConfig,
        pipeline_context: dict,
        action: dict = None,
    ) -> dict[str, Any]:
        """Шаг 2 pipeline: суммаризовать файл из контекста, записать summary_text."""
        if not self.summary_node:
            return {
                "answer": "Суммаризация недоступна.",
                "agent_steps": sub_state.get("agent_steps", []),
            }

        # _target_file_id — конкретный файл из expanded плана (несколько файлов)
        target_fid = (action or {}).get("_target_file_id")
        action_ns_hint = (action or {}).get("namespace_name_hint") if action else None
        if target_fid is not None:
            sub_state = {
                **sub_state,
                "history_file_id": target_fid,
                "search_file_ids": [target_fid],
                "detected_url": None,
                "file_content": None,
            }
            pipeline_context.pop("entity_content", None)
            pipeline_context.pop("entity_name_fallback", None)
        elif "history_file_id" in pipeline_context:
            sub_state = {
                **sub_state,
                "history_file_id": pipeline_context["history_file_id"],
                "detected_url": None,
                "file_content": None,
            }
        elif (
            action_ns_hint
            and not sub_state.get("file_content")
            and not sub_state.get("detected_url")
        ):
            # Явный namespace в шаге плана → суммаризируем файлы пространства,
            # игнорируем history_file_id из роутера (пользователь спросил о пространстве)
            ns_id = sub_state.get("namespace_id")
            if ns_id is not None:
                ns_file_ids = await self._user_file_ids_in_namespace_for_summarize(
                    config, sub_state.get("user_id"), ns_id
                )
                if ns_file_ids:
                    logger.info(
                        "[MultiActionNode] summarize: expanding to %d files in namespace_id=%d (explicit ns hint)",
                        len(ns_file_ids), ns_id,
                    )
                    sub_state = {
                        **sub_state,
                        "search_file_ids": ns_file_ids,
                        "history_file_id": None,
                        "detected_url": None,
                        "file_content": None,
                    }
        elif (
            not sub_state.get("history_file_id")
            and not sub_state.get("file_content")
            and not sub_state.get("detected_url")
            and sub_state.get("namespace_id")
        ):
            # Нет конкретного файла — суммаризируем все файлы пространства
            ns_id = sub_state["namespace_id"]
            ns_file_ids = await self._user_file_ids_in_namespace_for_summarize(
                config, sub_state.get("user_id"), ns_id
            )
            if ns_file_ids:
                logger.info(
                    "[MultiActionNode] summarize: expanding to %d files in namespace_id=%d",
                    len(ns_file_ids), ns_id,
                )
                sub_state = {
                    **sub_state,
                    "search_file_ids": ns_file_ids,
                    "detected_url": None,
                    "file_content": None,
                }

        result = await self.summary_node.run(sub_state, config)
        summary_result = result.get("summary_result") or {}
        if summary_result.get("summary"):
            pipeline_context["entity_content"] = summary_result["summary"]
            pipeline_context["entity_name_fallback"] = summary_result.get("title") or ""
            logger.info(
                "[MultiActionNode] summarize: captured summary len=%d title=%r",
                len(summary_result["summary"]),
                summary_result.get("title"),
            )
        elif result.get("answer") and not result.get("answer", "").startswith("Не нашёл"):
            # Multi-file summarization — нет summary_result, но есть combined answer
            pipeline_context["entity_content"] = result["answer"]
            logger.info(
                "[MultiActionNode] summarize: captured multi-file answer len=%d",
                len(result["answer"]),
            )
        return result

    async def _run_save_file(
        self,
        sub_state: AskState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """Шаг pipeline: FileAgent (парсинг/дедупликация) → SaveFileNode (сохранение в БД)."""
        if not self.file_agent or not self.save_file_node:
            return {
                "answer": "Сохранение файла недоступно (сервис не настроен).",
                "agent_steps": sub_state.get("agent_steps", []),
            }
        file_result = await self.file_agent.run(sub_state, config)
        merged_state: AskState = {**sub_state, **file_result}
        save_result = await self.save_file_node.run(merged_state, config)
        combined = {**file_result, **save_result}
        logger.info(
            "[MultiActionNode] save_file: file_id=%s answer=%s",
            combined.get("file_id"),
            (combined.get("answer") or "")[:80],
        )
        return combined

    async def _run_crud(
        self,
        sub_state: AskState,
        config: RunnableConfig,
        pipeline_context: dict,
        action: dict,
    ) -> dict[str, Any]:
        """CRUD-шаг: create_file / create_namespace / edit_file / delete / move_file."""
        intent = action.get("intent") or ""

        # Для create_file подмешиваем данные из pipeline если LLM их не указал
        if intent == "create_file":
            entity_content = sub_state.get("entity_content")
            entity_name = sub_state.get("entity_name")

            if not entity_content and pipeline_context.get("entity_content"):
                entity_content = pipeline_context["entity_content"]
                logger.info("[MultiActionNode] create_file: injected entity_content from pipeline")

            if not entity_name and pipeline_context.get("entity_name_fallback"):
                entity_name = pipeline_context["entity_name_fallback"]
                logger.info(
                    "[MultiActionNode] create_file: injected entity_name=%r from pipeline",
                    entity_name,
                )

            # Если контент всё ещё отсутствует — генерируем через GenerateContentNode
            if not entity_content and self.generate_content_node:
                logger.info("[MultiActionNode] create_file: no content, delegating to GenerateContentNode")
                gen_state = {**sub_state, "entity_name": entity_name}
                gen_result = await self.generate_content_node.run(gen_state, config)
                entity_content = gen_result.get("entity_content") or entity_content

            sub_state = {**sub_state, "entity_content": entity_content, "entity_name": entity_name}

        return await self.crud_node.run(sub_state, config)
