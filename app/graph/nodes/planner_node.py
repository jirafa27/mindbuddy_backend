"""PlannerNode — генерирует план (список шагов) через LLM и резолвит namespace для каждого шага."""
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from sqlalchemy import text

from app.core.enums import IntentType
from app.graph.state import AskState
from app.infrastructure.repositories.vector_queries import LIST_FILES_SQL
from app.services.planner_service import PlannerService, PlanStep

logger = logging.getLogger(__name__)

# Паттерн для поиска URL
_URL_RE = re.compile(
    r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b'
    r'[-a-zA-Z0-9()@:%_\+.~#?&//=]*',
    re.IGNORECASE,
)

# Количество сообщений истории, сканируемых для поиска URL/file_id
_HISTORY_SCAN_LIMIT = 10


class PlannerNode:
    """
    Генерирует план выполнения запроса пользователя:
    1. Собирает контекст (URL, файл, история)
    2. Вызывает PlannerService → список PlanStep
    3. Резолвит namespace_hint → namespace_id через БД
    4. Если 1 шаг → ставит intent + поля в state (дальше _route_after_planner)
       Если N шагов → ставит pending_actions (дальше MultiActionNode)
    """

    def __init__(self, planner_service: PlannerService) -> None:
        self.planner_service = planner_service

    async def run(self, state: AskState, config: RunnableConfig) -> AskState:
        question = state.get("question", "").strip()
        history = state.get("history") or []
        file_content = state.get("file_content")
        agent_steps = list(state.get("agent_steps") or [])

        # Контекст URL
        url_in_current_message: bool = state.get("url_in_current_message") or False
        has_history_url: bool = state.get("has_history_url") or False
        detected_url = state.get("detected_url")
        history_file_id = state.get("history_file_id")

        # Активный файл для разрешения местоимений
        active_file_ctx = self._extract_active_file_context(history, history_file_id)

        configurable = config.get("configurable") or {}
        db = configurable.get("async_db")
        user_id = state.get("user_id")

        # Активное пространство из URL (namespace_id) — для разрешения «это пространство»
        active_namespace_name: Optional[str] = None
        state_namespace_id = state.get("namespace_id")
        if state_namespace_id and db and user_id is not None:
            active_namespace_name = await self._resolve_namespace_name(db, state_namespace_id)

        # Генерируем план
        steps = await self.planner_service.plan(
            question,
            has_url=url_in_current_message,
            has_history_url=has_history_url,
            has_file=bool(file_content),
            active_file_context=active_file_ctx,
            active_namespace_name=active_namespace_name,
            history=history,
        )

        logger.info(
            "[PlannerNode] Plan generated: %s",
            " → ".join(s.tool for s in steps),
        )

        # Детерминированная коррекция: "что лежит/находится в пространстве X" → list_files
        steps = self._fix_list_vs_summarize(
            steps, question,
            has_file=bool(file_content),
            has_url=url_in_current_message or has_history_url,
            history_file_id=history_file_id,
            active_namespace_name=active_namespace_name,
            active_namespace_id=state_namespace_id,
        )

        # Детерминированная коррекция: убираем create_file после summarize
        # если пользователь не просил сохранять (нет слова "сохрани", "запиши", "добавь")
        steps = self._strip_implicit_create_file(steps, question)

        # "сохрани эту суммаризацию" → save_summary (берёт текст из истории, не перезапускает LLM)
        steps = self._collapse_save_summary(steps, question, history)

        # Резолвим namespace для каждого шага
        resolved_actions = await self._resolve_namespaces(steps, db, user_id)

        # Несколько send_file шагов не поддерживаются MultiActionNode — схлопываем в один
        resolved_actions = self._collapse_send_file_steps(resolved_actions)

        if len(resolved_actions) == 1:
            return self._build_single_step_state(
                state, resolved_actions[0], detected_url, history_file_id,
                agent_steps, question,
            )

        # Многошаговый план → MultiActionNode
        return {
            **state,
            "pending_actions": resolved_actions,
            "agent_steps": agent_steps + [
                f"[PlannerNode] Multi-step plan: {len(resolved_actions)} actions"
            ],
        }

    # ------------------------------------------------------------------
    # Резолв namespace
    # ------------------------------------------------------------------

    async def _resolve_namespaces(
        self,
        steps: List[PlanStep],
        db: Any,
        user_id: Optional[int],
    ) -> List[Dict[str, Any]]:
        """
        Для каждого шага резолвит namespace_hint → namespace_id через БД.
        Для create_namespace: переносит hint в entity_name.
        Пространства созданные в том же батче передаются следующим шагам.
        """
        resolved: List[Dict[str, Any]] = []
        pending_ns_names: list[str] = []  # имена пространств из create_namespace в батче

        for step in steps:
            ns_hint = step.namespace_hint
            ns_id: Optional[int] = None
            entity_name = step.entity_name

            # edit_namespace_name / edit_namespace_description / delete_namespace:
            # LLM часто кладёт текущее имя в search_query вместо namespace_hint — подхватываем его
            if (
                step.tool in (
                    IntentType.EDIT_NAMESPACE_NAME,
                    IntentType.EDIT_NAMESPACE_DESCRIPTION,
                    IntentType.DELETE_NAMESPACE,
                )
                and not ns_hint
                and step.search_query
            ):
                ns_hint = step.search_query
                logger.info(
                    "[PlannerNode] %s: fallback namespace_hint from search_query='%s'",
                    step.tool, ns_hint,
                )

            if ns_hint and db and user_id is not None:
                ns_id = await self._resolve_namespace_id(db, user_id, ns_hint)

            # create_namespace: имя может прийти в namespace_hint вместо entity_name
            if step.tool == IntentType.CREATE_NAMESPACE and not entity_name and ns_hint:
                entity_name = ns_hint
                ns_id = None
                ns_hint = None

            # Запоминаем создаваемые пространства
            if step.tool == IntentType.CREATE_NAMESPACE and entity_name:
                pending_ns_names.append(entity_name)

            # create_file/index_url без namespace_hint в батче с create_namespace → используем его
            if (
                step.tool in (IntentType.CREATE_FILE, IntentType.INDEX_URL)
                and ns_id is None
                and not ns_hint
                and pending_ns_names
            ):
                ns_hint = pending_ns_names[-1]
                logger.info(
                    "[PlannerNode] Inferred namespace '%s' for %s from batch create_namespace",
                    ns_hint, step.tool,
                )

            # save_file / summarize без namespace → по умолчанию кладём в Inbox
            if step.tool in (IntentType.SAVE_FILE, IntentType.SUMMARIZE) and ns_id is None and not ns_hint:
                if db and user_id is not None:
                    inbox_id = await self._resolve_namespace_id(db, user_id, "Inbox")
                    if inbox_id:
                        ns_id = inbox_id
                        ns_hint = "Inbox"
                        logger.info("[PlannerNode] %s: defaulting to Inbox (id=%d)", step.tool, inbox_id)

            resolved.append({
                "intent": step.tool,
                "namespace_id": ns_id,
                "namespace_name_hint": ns_hint,
                "search_query": step.search_query,
                "search_mode": step.search_mode,
                "entity_name": entity_name,
                "entity_description": step.entity_description,
                "entity_content": step.entity_content,
                "search_limit": step.search_limit,
            })

        return resolved

    async def _resolve_namespace_id(self, db: Any, user_id: int, name: str) -> Optional[int]:
        try:
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
            logger.warning("[PlannerNode] Failed to resolve namespace '%s': %s", name, exc)
            return None

    async def _resolve_namespace_name(self, db: Any, namespace_id: int) -> Optional[str]:
        try:
            result = await db.execute(
                text("SELECT name FROM namespaces WHERE id = :ns_id LIMIT 1"),
                {"ns_id": namespace_id},
            )
            row = result.mappings().first()
            return row["name"] if row else None
        except Exception as exc:
            logger.warning("[PlannerNode] Failed to resolve namespace name for id=%s: %s", namespace_id, exc)
            return None

    # ------------------------------------------------------------------
    # Схлопывание дублирующихся send_file шагов
    # ------------------------------------------------------------------

    @staticmethod
    def _collapse_send_file_steps(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Если все шаги плана — send_file, схлопываем их в один.
        Приоритет: шаг с search_query (by_topic) > шаг без него (all_in_namespace).
        namespace_id подхватывается из любого шага батча.
        """
        if not actions or not all(a["intent"] == IntentType.SEND_FILE for a in actions):
            return actions

        primary = next((a for a in actions if a.get("search_query")), actions[0])
        ns_id = primary.get("namespace_id")
        if ns_id is None:
            ns_id = next(
                (a["namespace_id"] for a in actions if a.get("namespace_id") is not None),
                None,
            )
        merged = {**primary, "namespace_id": ns_id}
        logger.info(
            "[PlannerNode] Collapsed %d send_file steps → 1 (mode=%s, query=%r, ns_id=%s)",
            len(actions), merged.get("search_mode"), merged.get("search_query"), ns_id,
        )
        return [merged]

    # ------------------------------------------------------------------
    # Формирование state для одношагового плана
    # ------------------------------------------------------------------

    def _build_single_step_state(
        self,
        state: AskState,
        action: Dict[str, Any],
        detected_url: Optional[str],
        history_file_id: Optional[int],
        agent_steps: list,
        question: str,
    ) -> AskState:
        """Строит state для одношагового плана — эквивалент логики маршрутизации в RouterNode."""
        intent = action["intent"]
        ns_id = action.get("namespace_id")
        ns_hint = action.get("namespace_name_hint")
        search_query = action.get("search_query")
        search_mode = action.get("search_mode")
        entity_name = action.get("entity_name")
        entity_description = action.get("entity_description")
        entity_content = action.get("entity_content")
        search_limit = action.get("search_limit")

        base = {
            **state,
            "intent": intent,
            "namespace_id": ns_id,
            "namespace_name_hint": ns_hint,
        }

        # Суммаризация
        if intent == IntentType.SUMMARIZE:
            url_in_current_message = state.get("url_in_current_message")
            effective_file_id = history_file_id or state.get("history_file_id")
            # URL в текущем сообщении имеет приоритет над файлом из истории
            if url_in_current_message and detected_url:
                effective_url = detected_url
                effective_file_id = None
            else:
                effective_url = detected_url if not effective_file_id else None
            return {
                **base,
                "detected_url": effective_url,
                "history_file_id": effective_file_id,
                "agent_steps": agent_steps + ["[PlannerNode] Single: summarize"],
            }

        # Поиск контента
        if intent == IntentType.RAG_QUERY:
            effective_search_file_ids = state.get("search_file_ids")
            if not effective_search_file_ids and history_file_id:
                effective_search_file_ids = [history_file_id]
            result: dict = {
                **base,
                "search_query": search_query or question,
                "search_file_ids": effective_search_file_ids,
                "agent_steps": agent_steps + ["[PlannerNode] Single: rag_query"],
            }
            if search_limit:
                result["search_limit"] = search_limit
            return result

        # Индексация URL
        if intent == IntentType.INDEX_URL:
            if not detected_url:
                return {
                    **state,
                    "intent": IntentType.RAG_QUERY,
                    "answer": "Не нашёл ссылку для сохранения.",
                    "agent_steps": agent_steps + ["[PlannerNode] Single: index_url, no URL"],
                }
            return {
                **base,
                "detected_url": detected_url,
                "agent_steps": agent_steps + ["[PlannerNode] Single: index_url"],
            }

        # Отправка файла
        if intent == IntentType.SEND_FILE:
            # Namespace берём как есть из плана LLM — промпт инструктирует ставить null
            # если пользователь не упомянул пространство явно.
            # Если LLM не вернул namespace — берём из state (прикреплённое через UI).
            effective_ns_id = ns_id if ns_id is not None else state.get("namespace_id")
            return {
                **base,
                "namespace_id": effective_ns_id,
                "search_query": search_query,
                "send_file_search_mode": search_mode or "by_topic",
                "history_file_id": history_file_id,
                "agent_steps": agent_steps + ["[PlannerNode] Single: send_file"],
            }

        # Список файлов
        if intent == IntentType.LIST_FILES:
            return {
                **base,
                "sql_query": LIST_FILES_SQL,
                "agent_steps": agent_steps + ["[PlannerNode] Single: list_files"],
            }

        # Сохранить последнее саммари из истории как файл
        if intent == IntentType.SAVE_SUMMARY:
            return {
                **base,
                "entity_name": entity_name,
                "agent_steps": agent_steps + ["[PlannerNode] Single: save_summary"],
            }

        # Сохранение загруженного файла
        if intent == IntentType.SAVE_FILE:
            return {
                **base,
                "agent_steps": agent_steps + ["[PlannerNode] Single: save_file"],
            }

        # Общий чат
        if intent == IntentType.GENERAL_CHAT:
            return {
                **base,
                "search_result": [],
                "agent_steps": agent_steps + ["[PlannerNode] Single: general_chat"],
            }

        # CRUD (create_namespace, edit_namespace*, delete_namespace,
        #        create_file, edit_file, delete_file, move_file)
        effective_entity_name = entity_name
        effective_ns_hint = ns_hint
        effective_ns_id = ns_id
        # create_namespace: LLM иногда кладёт название в namespace_hint
        if intent == IntentType.CREATE_NAMESPACE and not effective_entity_name and ns_hint:
            effective_entity_name = ns_hint
            effective_ns_hint = None
            effective_ns_id = None
        return {
            **base,
            "namespace_id": effective_ns_id,
            "namespace_name_hint": effective_ns_hint,
            "search_query": search_query,
            "entity_name": effective_entity_name,
            "entity_description": entity_description,
            "entity_content": entity_content,
            "agent_steps": agent_steps + [f"[PlannerNode] Single: {intent}"],
        }

    # ------------------------------------------------------------------
    # Утилиты для контекста
    # ------------------------------------------------------------------

    _LIST_TRIGGER = re.compile(
        r'(?:что|какие|какой|каков|что(?:\s+там)?|что\s+там)\s+'
        r'(?:лежит|находится|есть|содержится|хранится|у\s+меня|там)'
        r'|(?:покажи|перечисли|список|список\s+файлов|файлы(?:\s+в)?)\s+(?:в\s+)?(?:пространств|пространстве)',
        re.IGNORECASE,
    )

    _SUMMARIZE_TRIGGER = re.compile(
        r'о\s+чём|о\s+чем|что\s+содержит|что\s+(?:в\s+)?(?:этом\s+)?(?:файл|документ)|'
        r'расскажи\s+о\s+содержим|краткое\s+содержание|суммаризируй|суммаризация|'
        r'что\s+(?:написано|описано|говорится)\s+в',
        re.IGNORECASE,
    )

    _THIS_NS_TRIGGER = re.compile(
        r'(этого|этом|из этого|в этом|данного|данном)\s+пространства?'
        r'|пространства?\s+(этого|данного)',
        re.IGNORECASE,
    )

    def _fix_list_vs_summarize(
        self,
        steps: List[PlanStep],
        question: str,
        has_file: bool = False,
        has_url: bool = False,
        history_file_id: Optional[int] = None,
        active_namespace_name: Optional[str] = None,
        active_namespace_id: Optional[int] = None,
    ) -> List[PlanStep]:
        """
        Детерминированная коррекция часто путаемых инструментов.
        Применяется только к одношаговым планам.
        """
        if len(steps) != 1:
            return steps

        step = steps[0]

        # "Что лежит в пространстве X?" → list_files (даже если LLM решил summarize)
        if step.tool == IntentType.SUMMARIZE and self._LIST_TRIGGER.search(question):
            logger.info("[PlannerNode] Fix: summarize → list_files (list trigger in question)")
            return [PlanStep(
                tool=IntentType.LIST_FILES,
                namespace_hint=step.namespace_hint,
            )]

        # "суммаризируй файлы ЭТОГО пространства" + активное пространство →
        # принудительно используем активное пространство (не историческое)
        if (
            step.tool == IntentType.SUMMARIZE
            and not has_file
            and active_namespace_name
            and active_namespace_id
            and self._THIS_NS_TRIGGER.search(question)
        ):
            logger.info(
                "[PlannerNode] Fix: summarize namespace_hint → active namespace '%s'",
                active_namespace_name,
            )
            return [PlanStep(
                tool=IntentType.SUMMARIZE,
                namespace_hint=active_namespace_name,
            )]

        # Файл прикреплён + move_file → save_file (загрузить, а не перемещать существующий)
        if step.tool == IntentType.MOVE_FILE and has_file:
            logger.info("[PlannerNode] Fix: move_file → save_file (file attached)")
            return [PlanStep(
                tool=IntentType.SAVE_FILE,
                namespace_hint=step.namespace_hint,
            )]

        # "О чём этот файл?" / "что содержит" → summarize, но ТОЛЬКО если есть что суммаризировать
        if step.tool == IntentType.RAG_QUERY and self._SUMMARIZE_TRIGGER.search(question):
            if has_file or has_url or history_file_id:
                logger.info("[PlannerNode] Fix: rag_query → summarize (file/url in context)")
                return [PlanStep(
                    tool=IntentType.SUMMARIZE,
                    namespace_hint=step.namespace_hint,
                )]
            # Нет конкретного файла — rag_query по namespace подходит лучше
            logger.info("[PlannerNode] Fix: keep rag_query (summarize trigger but no file in context)")

        # summarize без файла/URL и без namespace → rag_query (обзорный запрос без контекста)
        # Если есть namespace_hint — суммаризируем все файлы пространства (не конвертируем)
        if (
            step.tool == IntentType.SUMMARIZE
            and not has_file and not has_url and not history_file_id
            and not step.namespace_hint
        ):
            logger.info("[PlannerNode] Fix: summarize → rag_query (no file/url/namespace to summarize)")
            return [PlanStep(
                tool=IntentType.RAG_QUERY,
                namespace_hint=step.namespace_hint,
                search_query=None,
                search_limit=25,
            )]

        return steps

    _SAVE_TRIGGER = re.compile(
        r'сохрани|запиши|добавь\s+в|положи\s+в|сохрани\s+туда|записать',
        re.IGNORECASE,
    )

    def _collapse_save_summary(
        self,
        steps: List[PlanStep],
        question: str,
        history: List[dict],
    ) -> List[PlanStep]:
        """
        summarize → create_file при наличии ответа ассистента в истории →
        схлопываем в save_summary (берёт текст из истории, не перезапускает LLM).
        """
        if len(steps) != 2:
            return steps

        if steps[0].tool != IntentType.SUMMARIZE or steps[1].tool != IntentType.CREATE_FILE:
            return steps

        has_assistant_msg = any(
            msg.get("role") == "assistant" and (msg.get("text") or "").strip()
            for msg in reversed(history)
        )
        if not has_assistant_msg:
            return steps

        ns_hint = steps[1].namespace_hint or steps[0].namespace_hint
        logger.info("[PlannerNode] Collapsed summarize→create_file into save_summary")
        return [PlanStep(tool=IntentType.SAVE_SUMMARY, namespace_hint=ns_hint)]

    def _strip_implicit_create_file(
        self,
        steps: List[PlanStep],
        question: str,
    ) -> List[PlanStep]:
        """
        Убирает create_file после summarize/save_summary, если пользователь
        просто спросил о содержании (без явной команды сохранить).
        Пример: "о чём это?" + [summarize, create_file] → [summarize]
        """
        if len(steps) < 2:
            return steps
        first_is_summarize = steps[0].tool in (IntentType.SUMMARIZE, IntentType.SAVE_SUMMARY)
        has_save_intent = self._SAVE_TRIGGER.search(question)
        if first_is_summarize and steps[1].tool == IntentType.CREATE_FILE and not has_save_intent:
            logger.info(
                "[PlannerNode] Fix: removed implicit create_file after %s (no save intent in question)",
                steps[0].tool,
            )
            return [steps[0]]
        return steps

    _FILENAME_PATTERN = re.compile(
        r'[«""]([^»""]{1,120}\.[a-zA-Z]{2,6})[»""]',
        re.IGNORECASE,
    )
    _ASSISTANT_FILE_ACTION_PATTERN = re.compile(
        r'[«""]([^»""]{1,120}\.[a-zA-Z]{2,6})[»""]\s+(?:был\s+)?(?:создан|отредактирован|перемещён|переименован)',
        re.IGNORECASE,
    )

    def _extract_active_file_context(
        self,
        history: List[Dict],
        history_file_id: Optional[int],
    ) -> Optional[str]:
        """Строит строку «filename.pdf (пространство: X)» для LLM-контекста."""
        filename: Optional[str] = None
        namespace_name: Optional[str] = None
        recent = list(reversed(history[-_HISTORY_SCAN_LIMIT:]))

        if history_file_id:
            for msg in recent:
                file_ids = msg.get("file_ids") or []
                if history_file_id in file_ids:
                    m = self._FILENAME_PATTERN.search(msg.get("text") or "")
                    if m:
                        filename = m.group(1)
                    break
            if not filename:
                for msg in recent:
                    m = self._FILENAME_PATTERN.search(msg.get("text") or "")
                    if m:
                        filename = m.group(1)
                        break
        else:
            for msg in recent:
                if msg.get("role") != "assistant":
                    continue
                m = self._ASSISTANT_FILE_ACTION_PATTERN.search(msg.get("text") or "")
                if m:
                    filename = m.group(1)
                    break

        ns_pattern = re.compile(
            r'в\s+пространств[еоу]\s+[«""]?([^»""«,.\n]{1,50})[»""]?', re.IGNORECASE
        )
        for msg in recent:
            m = ns_pattern.search(msg.get("text") or "")
            if m:
                namespace_name = m.group(1).strip()
                break

        if not filename:
            return None
        return f"{filename} (пространство: {namespace_name})" if namespace_name else filename
