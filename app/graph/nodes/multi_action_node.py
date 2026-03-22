"""MultiActionNode — последовательное выполнение нескольких CRUD-действий за один запрос."""
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.graph.state import AskState
from app.graph.nodes.crud_node import CrudNode

logger = logging.getLogger(__name__)


class MultiActionNode:
    """
    Выполняет список действий из state["pending_actions"] последовательно,
    вызывая CrudNode для каждого. Результаты объединяются в один ответ.
    """

    def __init__(self, crud_node: CrudNode) -> None:
        self.crud_node = crud_node

    async def run(self, state: AskState, config: RunnableConfig) -> dict[str, Any]:
        actions = state.get("pending_actions") or []
        if not actions:
            return {
                "answer": "Не указаны действия для выполнения.",
                "agent_steps": list(state.get("agent_steps") or []),
            }

        results: list[str] = []
        all_steps = list(state.get("agent_steps") or [])
        total = len(actions)

        # Пространства, созданные в рамках этого же батча: name (lower) → id
        # Используется для подстановки namespace_id в последующих действиях
        created_ns: dict[str, int] = {}

        # Pending-действия от delete-операций — объединим в одно подтверждение
        pending_deletes: list[dict] = []
        delete_targets: list[str] = []

        for i, action in enumerate(actions):
            # Если действие ссылается на пространство которое только что создали — подставляем ID
            ns_id = action.get("namespace_id")
            ns_hint = (action.get("namespace_name_hint") or "").strip()
            if ns_id is None and ns_hint:
                ns_id = created_ns.get(ns_hint.lower())
                if ns_id is not None:
                    logger.info(
                        "[MultiActionNode] action %d/%d: resolved ns '%s' → id=%d (from batch)",
                        i + 1, total, ns_hint, ns_id,
                    )

            sub_state: AskState = {
                **state,
                "intent": action.get("intent"),
                "namespace_id": ns_id,
                "namespace_name_hint": action.get("namespace_name_hint"),
                "search_query": action.get("search_query"),
                "entity_name": action.get("entity_name"),
                "entity_description": action.get("entity_description"),
                "entity_content": action.get("entity_content"),
                "answer": None,
                "agent_steps": all_steps + [f"[MultiActionNode] action {i + 1}/{total}: {action.get('intent')}"],
            }

            try:
                result = await self.crud_node.run(sub_state, config)
                all_steps = result.get("agent_steps", all_steps)

                # Delete-действия возвращают pending_action — собираем в батч
                pa = result.get("pending_action")
                if pa:
                    pending_deletes.append(pa)
                    delete_targets.append(pa.get("target", "объект"))
                else:
                    answer = (result.get("answer") or "").strip()
                    if answer:
                        results.append(answer)

                # Запоминаем только что созданное пространство для последующих действий
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
                results.append(f"Ошибка при выполнении действия {i + 1}.")

        # Формируем одно подтверждение для всех delete-действий
        final_pending_action = None
        if len(pending_deletes) == 1:
            final_pending_action = pending_deletes[0]
            results.append(
                f"Вы уверены, что хотите удалить {delete_targets[0]}? "
                "Это действие нельзя отменить. Напишите «да» для подтверждения."
            )
        elif len(pending_deletes) > 1:
            targets_text = ", ".join(delete_targets)
            final_pending_action = {"type": "batch_delete", "items": pending_deletes}
            results.append(
                f"Вы уверены, что хотите удалить: {targets_text}? "
                "Это действие нельзя отменить. Напишите «да» для подтверждения."
            )

        combined = "\n\n".join(results) if results else "Все операции выполнены."
        output: dict[str, Any] = {
            "answer": combined,
            "agent_steps": all_steps + [f"[MultiActionNode] Completed {total} actions"],
        }
        if final_pending_action:
            output["pending_action"] = final_pending_action
        return output
