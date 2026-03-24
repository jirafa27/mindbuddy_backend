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
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from app.graph.state import AskState
from app.graph.nodes.crud_node import CrudNode

logger = logging.getLogger(__name__)

# Интенты, которые маршрутизируются через pipeline (не CrudNode)
_PIPELINE_INTENTS = {"index_url", "summarize"}


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
        index_url_node=None,   # Optional[IndexUrlNode]
        summary_node=None,     # Optional[SummaryNode]
    ) -> None:
        self.crud_node = crud_node
        self.index_url_node = index_url_node
        self.summary_node = summary_node

    async def run(self, state: AskState, config: RunnableConfig) -> dict[str, Any]:
        actions = state.get("pending_actions") or []
        if not actions:
            return {
                "pipeline_report": [{"step": "multi_action", "ok": False, "message": "Не указаны действия для выполнения."}],
                "agent_steps": list(state.get("agent_steps") or []),
            }

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
            sub_state: AskState = {
                **state,
                "intent": intent,
                "namespace_id": ns_id,
                "namespace_name_hint": ns_hint or action.get("namespace_name_hint"),
                "search_query": action.get("search_query"),
                "entity_name": action.get("entity_name"),
                "entity_description": action.get("entity_description"),
                "entity_content": action.get("entity_content"),
                "answer": None,
                "agent_steps": all_steps + [base_step],
            }

            try:
                if intent == "index_url":
                    result = await self._run_index_url(sub_state, config, pipeline_context)
                elif intent == "summarize":
                    result = await self._run_summarize(sub_state, config, pipeline_context)
                else:
                    result = await self._run_crud(sub_state, config, pipeline_context, action)

                all_steps = result.get("agent_steps", all_steps)

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

    async def _run_summarize(
        self,
        sub_state: AskState,
        config: RunnableConfig,
        pipeline_context: dict,
    ) -> dict[str, Any]:
        """Шаг 2 pipeline: суммаризовать файл из контекста, записать summary_text."""
        if not self.summary_node:
            return {
                "answer": "Суммаризация недоступна.",
                "agent_steps": sub_state.get("agent_steps", []),
            }
        # Передаём file_id от предыдущего шага, убираем URL чтобы не перечитывать страницу
        if "history_file_id" in pipeline_context:
            sub_state = {
                **sub_state,
                "history_file_id": pipeline_context["history_file_id"],
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
        return result

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

            sub_state = {**sub_state, "entity_content": entity_content, "entity_name": entity_name}

        return await self.crud_node.run(sub_state, config)
