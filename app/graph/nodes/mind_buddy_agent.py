"""MindBuddyAgent: синтез ответа по найденным чанкам (RAG)."""
import logging
from typing import Any, List

from langchain_core.runnables import RunnableConfig

from app.graph.state import AskState
from app.domain.protocols import LLMProvider
from app.infrastructure.llm.yandex_completion import LLMCompletionError

logger = logging.getLogger(__name__)

def _format_chunks(search_result: List[dict]) -> str:
    """Форматирует результаты поиска для LLM.
    
    Поддерживает два типа результатов:
    - Семантический поиск: chunk_text, filename, relevance
    - Структурные запросы: filename, file_size, created_at, namespace_name
    """
    chunks_text = []
    for i, row in enumerate(search_result, 1):
        chunk_text = row.get("chunk_text", "")
        filename = row.get("filename", "?")
        relevance = row.get("relevance")
        
        if chunk_text:
            # Семантический поиск — есть текст чанка
            rel_str = f" (релевантность: {relevance:.2f})" if relevance is not None else ""
            chunks_text.append(f"[{i}] Файл: {filename}{rel_str}\n{chunk_text}")
        else:
            # Структурный запрос — метаданные файлов/пространств
            parts = [f"[{i}] Файл: {filename}"]
            if row.get("namespace_name"):
                parts.append(f"Пространство: {row['namespace_name']}")
            if row.get("file_size"):
                size_kb = row["file_size"] / 1024
                parts.append(f"Размер: {size_kb:.1f} КБ")
            if row.get("created_at"):
                parts.append(f"Создан: {row['created_at']}")
            chunks_text.append(" | ".join(parts))
    
    return "\n\n".join(chunks_text)


def _history_to_llm_messages(history: List[dict]) -> List[dict]:
    """Последние N сообщений чата в формате для LLM: [{role, text}, ...]."""
    out = []
    for h in history:
        role = (h.get("role") or "user").strip().lower()
        if role not in ("user", "assistant"):
            role = "user"
        text = (h.get("text") or "").strip()
        if text:
            out.append({"role": role, "text": text})
    return out


class MindBuddyAgent:
    """Формирует ответ на вопрос только на основе переданных фрагментов (search_result)."""

    def __init__(self, *, llm_service: LLMProvider):
        self.llm_service = llm_service

    async def run(self, state: AskState, config: RunnableConfig) -> dict[str, Any]:
        question = state.get("question") or ""
        search_result = state.get("search_result") or []
        agent_steps = list(state.get("agent_steps") or [])
        file_save_notice = state.get("file_save_notice") or ""

        # Pipeline-отчёт от MultiActionNode — формируем связный ответ через LLM
        pipeline_report = state.get("pipeline_report")
        if pipeline_report:
            # Если в батче есть отложенные удаления — спрашиваем подтверждение, не сообщаем об успехе
            pending_action = state.get("pending_action")
            if pending_action:
                return self._format_pending_confirmation(pending_action, pipeline_report, agent_steps)
            return await self._summarize_pipeline(pipeline_report, question, agent_steps)

        # Если ответ уже сформирован ранее (например, RouterNode) — просто возвращаем его
        existing_answer = state.get("answer")
        if existing_answer and not search_result:
            logger.info("[MindBuddyAgent] Returning pre-set answer")
            return {
                "answer": existing_answer,
                "sources": [],
                "agent_steps": agent_steps + ["MindBuddyAgent (pass-through)"],
            }

        sources = []
        seen_files = set()
        for row in search_result:
            fn = row.get("filename", "?")
            rel = row.get("relevance")
            if fn not in seen_files:
                seen_files.add(fn)
                sources.append({"filename": fn, "relevance": float(rel) if rel is not None else 0.0})

        if not search_result:
            # Нет релевантных чанков — отвечаем как в общем диалоге (приветствия, шутки, общие вопросы)
            system_msg = {
                "role": "system",
                "text": (
                    "Ты дружелюбный ассистент с лёгким характером. Пользователь написал сообщение без контекста из базы знаний. "
                    "Отвечай кратко, естественно и по-человечески: можно с юмором или игрой слов, поддерживать неформальный тон. "
                    "Если вопрос явно про поиск по документам — предложи загрузить файлы или уточнить вопрос.\n"
                    "ВАЖНО: операции с файлами и пространствами (перемещение, удаление, создание) выполняет система, "
                    "и она передаёт тебе явный отчёт о результатах. "
                    "Если такой отчёт тебе НЕ передан — значит операция не выполнялась. "
                    "Не утверждай что что-то было сделано, если не получил подтверждения от системы. "
                    "В таком случае скажи что не удалось обработать запрос и попроси переформулировать."
                ),
            }
            history_msgs = _history_to_llm_messages(state.get("history") or [])
            messages = [system_msg] + history_msgs + [{"role": "user", "text": question}]
            try:
                answer = await self.llm_service.complete(
                    messages, temperature=0.7, max_tokens=256
                )
                final_no_ctx = (f"{file_save_notice}\n\n{answer.strip()}" if file_save_notice else answer.strip())
                return {
                    "answer": final_no_ctx,
                    "sources": [],
                    "agent_steps": agent_steps + ["MindBuddyAgent (no context)"],
                }
            except LLMCompletionError:
                no_ctx_fallback = "По выбранному пространству знаний релевантных фрагментов не найдено. Загрузите документы или уточните вопрос."
                return {
                    "answer": (f"{file_save_notice}\n\n{no_ctx_fallback}" if file_save_notice else no_ctx_fallback),
                    "sources": [],
                    "agent_steps": agent_steps + ["MindBuddyAgent"],
                }

        chunks_block = _format_chunks(search_result)

        system_msg = {
            "role": "system",
            "text": (
                "Ты — ассистент по базе знаний. Отвечай только на основе приведённой информации. "
                "Информация может быть двух типов:\n"
                "1. Фрагменты текста из документов — отвечай по их содержимому.\n"
                "2. Список файлов/папок с метаданными — перечисли их пользователю.\n"
                "Не придумывай факты. Если информации недостаточно, скажи об этом."
            ),
        }
        last_user_msg = {
            "role": "user",
            "text": f"Вопрос пользователя: {question}\n\nИнформация из базы знаний:\n\n{chunks_block}\n\nДай подробный и информативный ответ на основе приведённых фрагментов. Если вопрос про содержимое файла — опиши его полно. ВАЖНО: если в информации есть несколько фрагментов из одного файла — объедини их в один раздел ответа, не повторяй один файл несколько раз.",
        }
        history_msgs = _history_to_llm_messages(state.get("history") or [])
        messages = [system_msg] + history_msgs + [last_user_msg]

        try:
            answer = await self.llm_service.complete(messages, temperature=0.3, max_tokens=1024)
            logger.info(f"ANSWER: {answer.strip()}")
            final_answer = (f"{file_save_notice}\n\n{answer.strip()}" if file_save_notice else answer.strip())
            return {
                "answer": final_answer,
                "sources": sources,
                "agent_steps": agent_steps + ["MindBuddyAgent"],
            }
        except LLMCompletionError:
            fallback = "Не удалось сформировать ответ. Попробуйте переформулировать вопрос."
            return {
                "answer": (f"{file_save_notice}\n\n{fallback}" if file_save_notice else fallback),
                "sources": sources,
                "agent_steps": agent_steps + ["MindBuddyAgent"],
            }

    async def _summarize_pipeline(
        self,
        pipeline_report: list,
        question: str,
        agent_steps: list,
    ) -> dict[str, Any]:
        """Формирует связный ответ на основе отчёта о выполненных шагах pipeline."""
        _STEP_NAMES = {
            "index_url": "Загрузка и сохранение контента",
            "summarize": "Суммаризация",
            "create_file": "Сохранение файла",
            "create_namespace": "Создание пространства",
            "delete_file": "Удаление файла",
            "delete_namespace": "Удаление пространства",
            "edit_file": "Редактирование файла",
            "edit_namespace_name": "Переименование пространства",
            "edit_namespace_description": "Обновление описания пространства",
            "move_file": "Перемещение файла",
        }

        steps_block = "\n".join(
            f"- {_STEP_NAMES.get(r['step'], r['step'])}: {'✓ ' + r['message'] if r['ok'] else '✗ ' + r['message']}"
            for r in pipeline_report
        )
        errors = [r for r in pipeline_report if not r["ok"]]

        # Если pipeline заканчивается на summarize (без последующего create_file),
        # то пользователь хотел увидеть сам текст саммари — возвращаем его напрямую.
        step_names = [r["step"] for r in pipeline_report]
        last_step = pipeline_report[-1] if pipeline_report else None
        has_summarize_terminal = (
            last_step is not None
            and last_step["step"] == "summarize"
            and last_step["ok"]
            and last_step["message"]
            and "create_file" not in step_names
        )
        if has_summarize_terminal and not errors:
            return {
                "answer": last_step["message"],
                "sources": [],
                "agent_steps": agent_steps + ["MindBuddyAgent (pipeline summary→direct)"],
            }

        system_msg = {
            "role": "system",
            "text": (
                "Ты ассистент. Пользователь попросил выполнить задачу, и система выполнила цепочку действий. "
                "Кратко и по-человечески сообщи пользователю о результате: что было сделано и что пошло не так (если были ошибки). "
                "Не перечисляй технические детали. Не придумывай. Отвечай на русском."
            ),
        }
        user_msg = {
            "role": "user",
            "text": (
                f"Запрос пользователя: {question}\n\n"
                f"Что было выполнено:\n{steps_block}\n\n"
                + ("Были ошибки — обязательно упомяни их." if errors else "Всё выполнено успешно.")
            ),
        }
        try:
            answer = await self.llm_service.complete(
                [system_msg, user_msg], temperature=0.3, max_tokens=512
            )
            return {
                "answer": answer.strip(),
                "sources": [],
                "agent_steps": agent_steps + ["MindBuddyAgent (pipeline summary)"],
            }
        except LLMCompletionError:
            # Fallback: детерминированный ответ без LLM
            ok_steps = [_STEP_NAMES.get(r["step"], r["step"]) for r in pipeline_report if r["ok"]]
            err_steps = [f"{_STEP_NAMES.get(r['step'], r['step'])}: {r['message']}" for r in pipeline_report if not r["ok"]]
            parts = []
            if ok_steps:
                parts.append("Выполнено: " + ", ".join(ok_steps) + ".")
            if err_steps:
                parts.append("Ошибки: " + "; ".join(err_steps) + ".")
            return {
                "answer": " ".join(parts) or "Операции выполнены.",
                "sources": [],
                "agent_steps": agent_steps + ["MindBuddyAgent (pipeline fallback)"],
            }

    def _format_pending_confirmation(
        self,
        pending_action: dict,
        pipeline_report: list,
        agent_steps: list,
    ) -> dict[str, Any]:
        """Формирует запрос подтверждения для отложенных delete-операций из батча."""
        targets: list[str] = []

        if pending_action.get("type") == "batch_delete":
            for item in pending_action.get("items") or []:
                targets.append(item.get("target", "объект"))
        else:
            target = pending_action.get("target")
            if target:
                targets.append(target)

        if targets:
            items_list = "\n".join(f"— {t}" for t in targets)
            answer = (
                f"Подтвердите удаление следующих объектов:\n{items_list}\n\n"
                "Это действие нельзя отменить. Напишите «да» для подтверждения."
            )
        else:
            answer = "Подтвердите выполнение операции. Напишите «да» для подтверждения."

        return {
            "answer": answer,
            "sources": [],
            "agent_steps": agent_steps + ["MindBuddyAgent (pending confirmation)"],
        }
