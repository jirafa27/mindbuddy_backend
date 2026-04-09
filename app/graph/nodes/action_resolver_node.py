"""ActionResolverNode — определяет параметры для каждого интента через LLM + резолвит namespaces."""
import json
import logging
import asyncio
import re
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig

from app.core.enums import IntentType
from app.graph.state import AskState
from app.graph.sub_state_builder import build_sub_state
from app.graph.utils.namespace import resolve_namespace_id, resolve_namespace_name, list_namespace_names
from app.infrastructure.repositories.vector_queries import LIST_FILES_SQL
from app.domain.protocols import LLMProvider
from app.services.planner_service import PlanStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Мини-промпты по интентам
# ---------------------------------------------------------------------------

_PARAM_PROMPTS: Dict[str, str] = {
    IntentType.SEND_FILE: """\
Определи параметры для отправки файла пользователю.
Верни JSON: {"namespace_hint": ..., "search_query": ..., "search_mode": ...}

Правила:
- namespace_hint — название пространства ТОЛЬКО если пользователь ЯВНО назвал его в сообщении. Если нет — null.
- search_query — что искать (название или тема файла). Если все файлы — null.
- search_mode — "by_name" (по имени), "by_topic" (по теме), "all_in_namespace" (все файлы).
  "Скинь все файлы" / "скинь файлы из X" → search_mode="all_in_namespace", search_query=null.
  "Скинь файл про ML" → search_mode="by_topic", search_query="ML".
  "Скинь файл report.pdf" → search_mode="by_name", search_query="report.pdf".
- НЕ бери namespace_hint из истории диалога — только из текущего сообщения.""",

    IntentType.SAVE_FILE: """\
Определи параметры для сохранения файла.
Верни JSON: {"namespace_hint": ...}

Правила:
- namespace_hint — название пространства, куда сохранить файл.
- Если пользователь назвал пространство явно ("в пространство Тест") — указать его название.
- Если пользователь говорит "туда", "там", "в это пространство" — взять название из поля "Активное пространство" в контексте.
- Если пространство не указано и нет активного в контексте — namespace_hint=null.""",

    IntentType.SUMMARIZE: """\
Определи параметры для суммаризации.
Верни JSON: {"namespace_hint": ...}

Правила:
- namespace_hint — ТОЧНОЕ название пространства если пользователь назвал его явно.
- "О чём файлы в пространстве Архив?" / "Суммаризируй файлы из Архив" → namespace_hint="Архив".
- "Суммаризируй файлы этого пространства" / "из этого пространства" / "в этом пространстве":
  Если в контексте указано "Активное пространство" — использовать его название как namespace_hint.
  Если активного пространства нет — namespace_hint=null (возьмётся из UI).
- Если суммаризируется прикреплённый файл или URL — namespace_hint=null.
- Если в контексте есть "Активное пространство (из UI)" и оно отличается от контекста диалога — приоритет у UI.""",

    IntentType.SAVE_SUMMARY: """\
Определи параметры для сохранения суммаризации.
Верни JSON: {"namespace_hint": ..., "entity_name": ...}

Правила:
- namespace_hint — пространство, куда сохранить. null если не указано.
- entity_name — название файла, если указано. Обычно null.
- "Сохрани эту суммаризацию в Архив" → namespace_hint="Архив"
- "Сохрани туда это" → namespace_hint=null""",

    IntentType.CREATE_NAMESPACE: """\
Определи параметры для создания пространства.
Верни JSON: {"entity_name": ..., "entity_description": ...}

Правила:
- entity_name — ОБЯЗАТЕЛЬНО: название нового пространства.
- entity_description — описание, если указано. Обычно null.
- КРИТИЧНО: Слова-коррекции («нет», «не», «нет,», «не то», «не так») — НЕ название пространства. Ищи название после них.""",

    IntentType.EDIT_NAMESPACE_NAME: """\
Определи параметры для переименования пространства.
Верни JSON: {"namespace_hint": ..., "entity_name": ...}

Правила:
- namespace_hint — текущее название пространства из сообщения. Если в контексте есть список существующих пространств и пользователь допустил опечатку — выбери ближайшее по написанию.
- entity_name — новое название.
- КРИТИЧНО: Слова-коррекции («нет», «не», «нет,», «не то», «не так») — НЕ название пространства → namespace_hint=null.""",

    IntentType.EDIT_NAMESPACE_DESCRIPTION: """\
Определи параметры для изменения описания пространства.
Верни JSON: {"namespace_hint": ..., "entity_description": ...}

Правила:
- namespace_hint — название пространства из сообщения. Если в контексте есть список существующих пространств и пользователь допустил опечатку — выбери ближайшее по написанию.
- entity_description — новое описание.
- КРИТИЧНО: Слова-коррекции («нет», «не», «нет,», «не то», «не так») — НЕ название пространства → namespace_hint=null.""",

    IntentType.DELETE_NAMESPACE: """\
Определи параметры для удаления пространства.
Верни JSON: {"namespace_hint": ...}

Правила:
- namespace_hint — название пространства из сообщения пользователя.
- Если в контексте есть список существующих пространств и название из сообщения содержит опечатку — выбери БЛИЖАЙШЕЕ по написанию пространство из списка.
- Не подставляй активное пространство, если пользователь назвал другое (даже с опечаткой).""",

    IntentType.MOVE_FILE: """\
Определи параметры для перемещения файла.
Верни JSON: {"search_query": ..., "namespace_hint": ..., "entity_name": ...}

Правила:
- search_query — имя файла для перемещения. null если все файлы.
- namespace_hint — название пространства НАЗНАЧЕНИЯ из сообщения. Если в контексте есть список существующих пространств и пользователь допустил опечатку — выбери ближайшее по написанию.
- entity_name — название пространства ИСТОЧНИК из сообщения. Те же правила по опечаткам. null если один файл.""",

    IntentType.CREATE_FILE: """\
Определи параметры для создания файла/заметки.
Верни JSON: {"entity_name": ..., "entity_content": ..., "namespace_hint": ...}

Правила:
- entity_name — заголовок ФАЙЛА (НЕ название пространства!). Например: "Шутка1", "Заметка", "Отчёт".
- entity_content — текст содержимого файла. null если контент не указан (будет сгенерирован).
- namespace_hint — ТОЧНОЕ название пространства НАЗНАЧЕНИЯ из сообщения. null если не указано
  или если пространство создаётся в этом же запросе (будет подставлено автоматически).
- КРИТИЧНО: НЕ путай entity_name и namespace_hint! entity_name = название файла, namespace_hint = название пространства.""",

    IntentType.DELETE_FILE: """\
Определи параметры для удаления файла.
Верни JSON: {"search_query": ..., "namespace_hint": ..., "search_limit": ...}

Правила:
- search_query — имя файла. null если удаление из пространства (всех или любого).
- namespace_hint — название пространства из сообщения пользователя. Если в контексте есть список существующих пространств и пользователь допустил опечатку — выбери ближайшее по написанию. null если не указано.
- search_limit — целое число. Если "удали любой файл" / "удали один файл" → search_limit=1. Если "удали все файлы" или конкретный файл → null.""",

    IntentType.EDIT_FILE: """\
Определи параметры для редактирования файла.
Верни JSON: {"search_query": ..., "entity_content": ..., "namespace_hint": ...}

Правила:
- search_query — имя конкретного файла. null если нужно редактировать ВСЕ файлы пространства
  (пользователь говорит "в каждый файл", "во все файлы", "все файлы", "каждый файл", не называя конкретный).
- entity_content — инструкция что изменить. Если пользователь говорит "в каждый файл ..." —
  бери только ИНСТРУКЦИЮ (что добавить/изменить), без "в каждый файл".
- namespace_hint — пространство, если указано. Никогда не дублируй название пространства в search_query.""",

    IntentType.RENAME_FILE: """\
Определи параметры для переименования файла (только название, не содержимое).
Верни JSON: {"search_query": ..., "entity_name": ..., "namespace_hint": ..., "entity_content": ...}

Правила:
- Один конкретный файл: search_query — текущее имя или его часть; entity_name — новое имя (без лишних слов вроде «в», «на»).
- Несколько файлов в пространстве / «переименуй файлы так, чтобы назывались X, Y и Z»: search_query = null;
  entity_content — только список новых имён через запятую, в нужном порядке, например "Совет1, Совет2, Совет3"
  (слова «переименуй», «чтобы», «назывались» не включай — только имена).
- namespace_hint — пространство, если пользователь назвал явно; иначе null (активное пространство из UI).
- Для одного файла entity_content = null. Для пакета имён entity_name обычно null.""",

    IntentType.RAG_QUERY: """\
Определи параметры для поиска информации.
Верни JSON: {"search_query": ..., "namespace_hint": ...}

Правила:
- search_query — ОБЯЗАТЕЛЬНО: вопрос/тема в 2-5 словах.
- namespace_hint — пространство для поиска, ТОЛЬКО если пользователь ЯВНО его назвал.
- КРИТИЧНО: НЕ извлекай namespace_hint из URL (домен, путь). "ru.wikipedia.org", "youtube.com", "github.com" — НЕ пространства → null.""",

    IntentType.INDEX_URL: """\
Определи параметры для сохранения URL.
Верни JSON: {"namespace_hint": ...}

Правила:
- namespace_hint — название пространства ТОЛЬКО если пользователь ЯВНО его назвал ("сохрани ссылку в Архив").
- КРИТИЧНО: НЕ извлекай namespace_hint из самого URL (домена, пути, заголовка страницы)! "wikipedia", "youtube", "github" — это НЕ пространства.
- Если пользователь говорит "туда" / "в это пространство" — взять из поля "Активное пространство" в контексте.
- Если пользователь не указал пространство — namespace_hint=null.""",

    IntentType.LIST_FILES: """\
Определи параметры для списка файлов.
Верни JSON: {"namespace_hint": ...}

Правила:
- namespace_hint — пространство. null если все файлы пользователя.""",

    IntentType.GENERAL_CHAT: """\
Верни JSON: {}
Для обычного общения параметры не нужны.""",
}

_DEFAULT_PARAM_PROMPT = """\
Определи параметры для действия.
Верни JSON: {"namespace_hint": ..., "search_query": ..., "entity_name": ..., "entity_description": ..., "entity_content": ...}
Все поля nullable.

КРИТИЧНО: Слова-коррекции («нет», «не», «нет,», «не то», «не так», «неправильно») — это НЕ название пространства,
НЕ имя файла и НЕ поисковый запрос. Если сообщение начинается с такого слова — игнорируй его при извлечении параметров."""


# ---------------------------------------------------------------------------
# Узел
# ---------------------------------------------------------------------------

class ActionResolverNode:
    """
    Второй узел двухузловой системы планирования.
    Для каждого интента из planned_intents:
    1. LLM-вызов со специфичным промптом → параметры (namespace_hint, search_query, etc.)
    2. _resolve_namespaces → namespace_id через БД
    3. Формирует pending_actions или single-step state.
    """

    def __init__(self, llm_service: LLMProvider) -> None:
        self.llm_service = llm_service

    async def run(self, state: AskState, config: RunnableConfig) -> AskState:
        planned_intents: List[str] = state.get("planned_intents") or []
        question = state.get("question", "").strip()
        has_file = bool(state.get("attached_files"))
        agent_steps = list(state.get("agent_steps") or [])

        detected_url = state.get("detected_url")
        history_file_id = state.get("history_file_id")

        configurable = config.get("configurable") or {}
        db = configurable.get("async_db")
        user_id = state.get("user_id")

        # namespace_id от UI (клиент явно передал)
        active_namespace_name: Optional[str] = None
        state_namespace_id = state.get("namespace_id")
        if state_namespace_id and db and user_id is not None:
            active_namespace_name = await self._resolve_namespace_name(db, state_namespace_id)

        # Если клиент не передал namespace_id: берём active_namespace_id из Chat.context
        # (последнее пространство после предыдущего ответа) и подставляем в промпт только
        # для интентов из _CONTEXT_NS_INTENTS — глобальные запросы (rag_query и т.д.) не затрагиваем.
        conversation_context = state.get("conversation_context") or {}
        context_namespace_name: Optional[str] = None
        ctx_ns_id = conversation_context.get("active_namespace_id")
        if ctx_ns_id and not state_namespace_id and db and user_id is not None:
            context_namespace_name = await self._resolve_namespace_name(db, ctx_ns_id)

        available_ns_names: List[str] = []
        if db and user_id is not None:
            available_ns_names = await list_namespace_names(db, user_id)

        steps: List[PlanStep] = []
        resolved_so_far: List[str] = []
        for intent in planned_intents:
            params = await self._resolve_params(
                intent, question, active_namespace_name, has_file,
                plan_context=resolved_so_far if len(planned_intents) > 1 else None,
                context_namespace_name=context_namespace_name,
                available_namespaces=available_ns_names,
            )
            search_limit_raw = params.get("search_limit")
            search_limit = int(search_limit_raw) if search_limit_raw is not None else None

            step = PlanStep(
                tool=intent,
                namespace_hint=params.get("namespace_hint"),
                search_query=params.get("search_query"),
                search_mode=params.get("search_mode"),
                entity_name=params.get("entity_name"),
                entity_description=params.get("entity_description"),
                entity_content=params.get("entity_content"),
                search_limit=search_limit,
            )
            steps.append(step)

            if intent == IntentType.CREATE_NAMESPACE:
                resolved_so_far.append(f"{intent}(name={step.entity_name})")
            else:
                parts = [f"ns={step.namespace_hint}"]
                if step.search_query:
                    parts.append(f"query={step.search_query}")
                if step.entity_name:
                    parts.append(f"file={step.entity_name}")
                resolved_so_far.append(f"{intent}({', '.join(parts)})")

        logger.info(
            "[ActionResolverNode] Resolved plan: %s",
            " → ".join(f"{s.tool}(ns={s.namespace_hint})" for s in steps),
        )

        resolved_actions = await self._resolve_namespaces(steps, db, user_id, conversation_context)
        resolved_actions = self._collapse_send_file_steps(resolved_actions)

        if len(resolved_actions) == 1:
            return self._build_single_step_state(
                state, resolved_actions[0], detected_url, history_file_id,
                agent_steps, question,
            )

        return {
            **state,
            "pending_actions": resolved_actions,
            "agent_steps": agent_steps + [
                f"[ActionResolverNode] Multi-step plan: {len(resolved_actions)} actions"
            ],
        }

    # ------------------------------------------------------------------
    # LLM для параметров
    # ------------------------------------------------------------------

    _CONTEXT_NS_INTENTS = {
        IntentType.EDIT_FILE, IntentType.RENAME_FILE, IntentType.DELETE_FILE,
        IntentType.MOVE_FILE, IntentType.SEND_FILE, IntentType.SUMMARIZE,
        IntentType.EDIT_NAMESPACE_NAME, IntentType.EDIT_NAMESPACE_DESCRIPTION,
        IntentType.DELETE_NAMESPACE, IntentType.SAVE_FILE,
    }

    async def _resolve_params(
        self,
        intent: str,
        question: str,
        active_namespace_name: Optional[str],
        has_file: bool,
        plan_context: Optional[List[str]] = None,
        context_namespace_name: Optional[str] = None,
        available_namespaces: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        prompt_template = _PARAM_PROMPTS.get(intent, _DEFAULT_PARAM_PROMPT)

        context_parts = []
        if has_file:
            context_parts.append("Прикреплён файл.")
        if available_namespaces:
            context_parts.append(
                f"Существующие пространства пользователя: {', '.join(available_namespaces)}."
            )
        if active_namespace_name:
            context_parts.append(f"Активное пространство: {active_namespace_name}.")
        elif intent in self._CONTEXT_NS_INTENTS and context_namespace_name:
            context_parts.append(
                f"Активное пространство (из контекста диалога): {context_namespace_name}."
            )
        if plan_context:
            context_parts.append(
                f"Уже определены шаги: {' → '.join(plan_context)}. "
                f"Определяешь параметры для ДРУГОГО шага — не дублируй те же объекты!"
            )

        context_line = f"[Контекст: {' '.join(context_parts)}]\n" if context_parts else ""

        messages = [
            {"role": "system", "text": f"КРИТИЧНО: верни ТОЛЬКО JSON без пояснений.\n\n{prompt_template}"},
            {"role": "user", "text": f"{context_line}Запрос: {question}"},
        ]


        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                raw = await self.llm_service.complete(messages, temperature=0.0, max_tokens=256)
                logger.info("[ActionResolverNode] Params for %s: %r", intent, raw[:200])
                return self._parse_params(raw)
            except Exception as exc:
                if attempt < max_retries:
                    delay = 0.5 * (attempt + 1)
                    logger.warning(
                        "[ActionResolverNode] LLM params failed for %s (attempt %d/%d): %s — retrying in %.1fs",
                        intent, attempt + 1, max_retries + 1, exc, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("[ActionResolverNode] LLM params failed for %s after %d attempts: %s", intent, max_retries + 1, exc)
                    return {}

    @staticmethod
    def _parse_params(raw: str) -> Dict[str, Any]:
        text = raw.strip()
        md_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if md_match:
            text = md_match.group(1).strip()

        brace_match = re.search(r"\{[\s\S]*\}", text)
        if not brace_match:
            return {}

        try:
            data = json.loads(brace_match.group(0))
            if isinstance(data, dict):
                return {
                    k: (None if (v is None or v == "null" or v == "") else v)
                    for k, v in data.items()
                }
        except json.JSONDecodeError:
            pass
        return {}

    # ------------------------------------------------------------------
    # Резолв namespace (перенесено из PlannerNode)
    # ------------------------------------------------------------------

    async def _resolve_namespaces(
        self,
        steps: List[PlanStep],
        db: Any,
        user_id: Optional[int],
        conversation_context: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        resolved: List[Dict[str, Any]] = []
        pending_ns_names: list[str] = []

        for step in steps:
            ns_hint = step.namespace_hint
            ns_id: Optional[int] = None
            entity_name = step.entity_name

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
                    "[ActionResolverNode] %s: fallback namespace_hint from search_query='%s'",
                    step.tool, ns_hint,
                )

            if ns_hint and pending_ns_names and any(ns_hint.lower() == pn.lower() for pn in pending_ns_names):
                logger.info(
                    "[ActionResolverNode] %s: namespace '%s' is pending creation in batch — skipping DB resolution",
                    step.tool, ns_hint,
                )
            elif ns_hint and db and user_id is not None:
                ns_id = await self._resolve_namespace_id(db, user_id, ns_hint)

            if ns_id is not None and db and user_id is not None:
                resolved_name = await self._resolve_namespace_name(db, ns_id)
                if resolved_name:
                    ns_hint = resolved_name

            if step.tool == IntentType.CREATE_NAMESPACE and not entity_name and ns_hint:
                entity_name = ns_hint
                ns_id = None
                ns_hint = None

            if step.tool == IntentType.CREATE_NAMESPACE and entity_name:
                pending_ns_names.append(entity_name)

            if (
                step.tool in (IntentType.CREATE_FILE, IntentType.INDEX_URL, IntentType.SAVE_FILE, IntentType.SUMMARIZE)
                and ns_id is None
                and not ns_hint
                and pending_ns_names
            ):
                ns_hint = pending_ns_names[-1]
                logger.info(
                    "[ActionResolverNode] Inferred namespace '%s' for %s from batch create_namespace",
                    ns_hint, step.tool,
                )

            if step.tool == IntentType.SAVE_FILE and ns_id is None and not ns_hint:
                # Сначала пробуем активное пространство из контекста диалога
                ctx_ns_id = (conversation_context or {}).get("active_namespace_id")
                if ctx_ns_id and db and user_id is not None:
                    ns_id = ctx_ns_id
                    logger.info(
                        "[ActionResolverNode] %s: using active_namespace_id=%d from conversation_context",
                        step.tool, ns_id,
                    )
                elif db and user_id is not None:
                    inbox_id = await self._resolve_namespace_id(db, user_id, "Inbox")
                    if inbox_id:
                        ns_id = inbox_id
                        ns_hint = "Inbox"
                        logger.info("[ActionResolverNode] %s: defaulting to Inbox (id=%d)", step.tool, inbox_id)

            if step.tool == IntentType.INDEX_URL and ns_id is None and not ns_hint:
                if db and user_id is not None:
                    inbox_id = await self._resolve_namespace_id(db, user_id, "Inbox")
                    if inbox_id:
                        ns_id = inbox_id
                        ns_hint = "Inbox"
                        logger.info("[ActionResolverNode] %s: defaulting to Inbox (id=%d)", step.tool, inbox_id)

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
        return await resolve_namespace_id(db, user_id, name)

    async def _resolve_namespace_name(self, db: Any, namespace_id: int) -> Optional[str]:
        return await resolve_namespace_name(db, namespace_id)

    # ------------------------------------------------------------------
    # Несколько шагов send_file подряд → один шаг (дубликаты от LLM/плана).
    # Основной шаг — с search_query, иначе первый; namespace_id — из любого шага, где задан.
    # ------------------------------------------------------------------

    @staticmethod
    def _collapse_send_file_steps(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
            "[ActionResolverNode] Collapsed %d send_file steps → 1 (mode=%s, query=%r, ns_id=%s)",
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
        intent = action["intent"]
        ns_id = action.get("namespace_id")
        ns_hint = action.get("namespace_name_hint")
        search_query = action.get("search_query")
        search_mode = action.get("search_mode")
        entity_name = action.get("entity_name")
        entity_description = action.get("entity_description")
        entity_content = action.get("entity_content")
        search_limit = action.get("search_limit")

        # Если LLM не вернул id пространства — берём из UI (прикреплённое пространство)
        effective_ns_id = ns_id if ns_id is not None else state.get("namespace_id")

        # Фолбэк на последнее активное пространство из контекста диалога —
        # только для операций, которые напрямую изменяют пространство.
        # НЕ применяем для summarize/rag_query/index_url — там namespace_id
        # управляет поиском/сохранением и не должен браться из контекста.
        _CONTEXT_NS_FALLBACK_INTENTS = {
            IntentType.EDIT_NAMESPACE_NAME,
            IntentType.EDIT_NAMESPACE_DESCRIPTION,
            IntentType.DELETE_NAMESPACE,
            IntentType.CREATE_FILE,
        }
        if effective_ns_id is None and intent in _CONTEXT_NS_FALLBACK_INTENTS:
            ctx = state.get("conversation_context") or {}
            effective_ns_id = ctx.get("active_namespace_id") or None

        if intent == IntentType.SUMMARIZE:
            url_in_current_message = state.get("url_in_current_message")
            effective_file_id = history_file_id or state.get("history_file_id")

            if url_in_current_message and detected_url:
                effective_url = detected_url
                effective_file_id = None
            elif effective_ns_id and not detected_url:
                # Пользователь спрашивает "о чём файлы в пространстве" —
                # суммаризируем пространство, а не случайный файл из истории
                effective_url = None
                effective_file_id = None
            else:
                effective_url = detected_url if not effective_file_id else None

            return build_sub_state(
                state, intent,
                namespace_id=effective_ns_id,
                namespace_name_hint=ns_hint,
                detected_url=effective_url,
                history_file_id=effective_file_id,
                agent_steps=agent_steps + ["[ActionResolverNode] Single: summarize"],
            )

        if intent == IntentType.RAG_QUERY:
            effective_search_file_ids = state.get("search_file_ids")
            if not effective_search_file_ids and history_file_id:
                effective_search_file_ids = [history_file_id]

            if not state.get("explicit_file_ids") and not effective_ns_id:
                effective_search_file_ids = None

            return build_sub_state(
                state, intent,
                namespace_id=effective_ns_id,
                namespace_name_hint=ns_hint,
                search_query=search_query or question,
                search_file_ids=effective_search_file_ids,
                search_limit=search_limit,
                agent_steps=agent_steps + ["[ActionResolverNode] Single: rag_query"],
            )

        if intent == IntentType.INDEX_URL:
            if not detected_url:
                return build_sub_state(
                    state, IntentType.RAG_QUERY,
                    namespace_id=effective_ns_id,
                    answer="Не нашёл ссылку для сохранения.",
                    agent_steps=agent_steps + ["[ActionResolverNode] Single: index_url, no URL"],
                )
            return build_sub_state(
                state, intent,
                namespace_id=effective_ns_id,
                namespace_name_hint=ns_hint,
                detected_url=detected_url,
                agent_steps=agent_steps + ["[ActionResolverNode] Single: index_url"],
            )

        if intent == IntentType.SEND_FILE:
            return build_sub_state(
                state, intent,
                namespace_id=effective_ns_id,
                namespace_name_hint=ns_hint,
                search_query=search_query,
                send_file_search_mode=search_mode or "by_topic",
                history_file_id=history_file_id,
                agent_steps=agent_steps + ["[ActionResolverNode] Single: send_file"],
            )

        if intent == IntentType.LIST_FILES:
            return build_sub_state(
                state, intent,
                namespace_id=effective_ns_id,
                namespace_name_hint=ns_hint,
                sql_query=LIST_FILES_SQL,
                agent_steps=agent_steps + ["[ActionResolverNode] Single: list_files"],
            )

        if intent == IntentType.SAVE_SUMMARY:
            return build_sub_state(
                state, intent,
                namespace_id=effective_ns_id,
                namespace_name_hint=ns_hint,
                entity_name=entity_name,
                agent_steps=agent_steps + ["[ActionResolverNode] Single: save_summary"],
            )

        if intent == IntentType.SAVE_FILE:
            return build_sub_state(
                state, intent,
                namespace_id=effective_ns_id,
                namespace_name_hint=ns_hint,
                agent_steps=agent_steps + ["[ActionResolverNode] Single: save_file"],
            )

        if intent == IntentType.GENERAL_CHAT:
            return build_sub_state(
                state, intent,
                namespace_id=effective_ns_id,
                search_result=[],
                agent_steps=agent_steps + ["[ActionResolverNode] Single: general_chat"],
            )

        # CRUD: create_file, edit_file, delete_file, move_file,
        #       create_namespace, delete_namespace, edit_namespace_*
        effective_entity_name = entity_name
        effective_ns_hint = ns_hint
        if intent == IntentType.CREATE_NAMESPACE and not effective_entity_name and ns_hint:
            effective_entity_name = ns_hint
            effective_ns_hint = None
            effective_ns_id = None

        return build_sub_state(
            state, intent,
            namespace_id=effective_ns_id,
            namespace_name_hint=effective_ns_hint,
            search_query=search_query,
            search_limit=search_limit,
            entity_name=effective_entity_name,
            entity_description=entity_description,
            entity_content=entity_content,
            agent_steps=agent_steps + [f"[ActionResolverNode] Single: {intent}"],
        )
