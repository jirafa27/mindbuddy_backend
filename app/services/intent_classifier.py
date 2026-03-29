"""IntentClassifierService — определяет список намерений пользователя через LLM (без параметров)."""
import json
import logging
from typing import List, Optional

from app.core.enums import IntentType
from app.domain.protocols import LLMProvider

logger = logging.getLogger(__name__)

_VALID_INTENTS = {i.value for i in IntentType}

_HISTORY_CONTEXT_LIMIT = 4

_SYSTEM_PROMPT = """\
Ты классификатор намерений в приложении-базе знаний.
Определи, какие действия нужны для выполнения запроса пользователя.
Верни ТОЛЬКО JSON: {"intents": ["intent1", "intent2", ...]}

Доступные намерения:
- rag_query — вопрос к содержимому документов: найти фрагмент, факт, определение ("что написано про X", "расскажи про Y", "что говорится о Z", "найди в файле/документе", "в этом файле найди", "что такое X" в контексте открытого файла, "где в тексте про X"). НЕ целостное резюме всего файла!
- send_file — найти и отправить файл пользователю ("скинь", "отправь", "пришли", "дай", "найди файл", "найди документ", "поищи файл")
- list_files — показать ТОЛЬКО СПИСОК файлов (названия/метаданные) без содержимого ("покажи", "что лежит", "перечисли", "какие файлы есть")
- summarize — целостное краткое содержание файла/файлов ("о чём файл", "о чём файлы", "что в файлах целиком", "суммаризируй", "кратко перескажи документ", "о чём файлы в пространстве", "о чём эти файлы"). Если нужен ОТВЕТ НА КОНКРЕТНЫЙ ВОПРОС по тексту — это rag_query, не summarize!
- save_summary — сохранить ГОТОВОЕ саммари из предыдущего ответа ассистента как файл
- save_file — сохранить загруженный файл в базу знаний
- index_url — сохранить URL в базу знаний
- create_file — создать заметку/файл из текста
- edit_file — изменить содержимое файла
- rename_file — переименовать файл (только название: «переименуй файл», «назови файл», «дай файлу имя X»)
- delete_file — удалить файл
- move_file — переместить файл из одного пространства в другое (нужны ДВА пространства!)
- create_namespace — создать пространство знаний
- edit_namespace_name — переименовать пространство
- edit_namespace_description — изменить описание пространства
- delete_namespace — удалить пространство
- general_chat — обычный разговор, приветствие

Правила:
1. Для простых запросов — один интент.
2. "Создай пространство X и загрузи файл" → ["create_namespace", "save_file"]
3. "Суммаризируй и сохрани" → ["summarize", "create_file"]
4. "Суммаризируй" / "суммаризируй пространство" (без "сохрани") → ["summarize"] (ТОЛЬКО один, без create_file!
   Слова "пространство", "файлы", "документы" — это НЕ запрос на создание файла!)
5. КРИТИЧНО: Если прикреплён файл (флаг [Прикреплён файл]):
   - "загрузи/сохрани/добавь этот файл" → ["save_file"]
   - "этот файл" в любом контексте относится к прикреплённому файлу, НЕ к чему-то ещё
   - "создай пространство X и сохрани этот файл" → ["create_namespace", "save_file"]
   - НЕ используй save_summary когда есть прикреплённый файл!
6. Если прикреплён файл и пользователь говорит "о чём этот файл" / "кратко о документе" → ["summarize"]
6a. КРИТИЧНО: "найди в (этом) файле", "найди в документе", "что в файле про X", "информация о X в файле", "что такое X" (когда речь о фрагменте/определении из текста), "где сказано про X" → ["rag_query"], НЕ summarize!
7. save_summary — ТОЛЬКО если пользователь ЯВНО просит СОХРАНИТЬ существующую суммаризацию
   ("сохрани эту суммаризацию", "сохрани это резюме", "сохрани туда это").
   КРИТИЧНО: "о чём файлы" / "о чём файл" / "что в файлах" — это ВСЕГДА summarize, НИКОГДА save_summary!
   save_summary НЕ используется когда пользователь СПРАШИВАЕТ о содержимом.
8. "Скинь/отправь/найди файл" → ["send_file"] (НЕ rag_query, НЕ move_file!)
9. "Перемести файл из X в Y" → ["move_file"] (два пространства!)
10. "Скинь из X" (одно пространство) → ["send_file"] (НЕ move_file!)
11. "Создай пространство X и сохрани туда суммаризации" (нет прикреплённого файла) → ["create_namespace", "summarize", "create_file"]
    (пара summarize+create_file повторяется по числу файлов, если их несколько)
12. КРИТИЧНО: "отредактируй/допиши/измени все файлы / каждый файл / файлы пространства X" → ["edit_file"] (РОВНО ОДИН интент!
    Никогда НЕ повторяй edit_file несколько раз для одного запроса "в каждый файл" / "все файлы"!
12a. Переименование ФАЙЛА (не пространства!) → ["rename_file"]. "Переименуй пространство X" → ["edit_namespace_name"].
13. Если в одном сообщении НЕСКОЛЬКО команд (через точку, запятую, "и", "а также") — определи интент для КАЖДОЙ команды.
    "Удали пространство X, удали файл из Y" → ["delete_namespace", "delete_file"] (ДВА интента!)

Примеры:
"привет" → {"intents": ["general_chat"]}
"скинь файл про ML" → {"intents": ["send_file"]}
"найди файл про ML" → {"intents": ["send_file"]}
"найди документ с планом разработки" → {"intents": ["send_file"]}
"найди файл, который содержит X" → {"intents": ["send_file"]}
"что написано про Python?" → {"intents": ["rag_query"]}
"найди в этом файле информацию о том, что такое X" → {"intents": ["rag_query"]}
"в документе найди определение Y" → {"intents": ["rag_query"]}
"что в этом файле про информационную безопасность?" → {"intents": ["rag_query"]}
"суммаризируй" → {"intents": ["summarize"]}
"суммаризируй пространство" → {"intents": ["summarize"]}
"суммаризируй файлы" → {"intents": ["summarize"]}
"о чём этот файл?" → {"intents": ["summarize"]}
"создай пространство Работа" → {"intents": ["create_namespace"]}
"создай пространство ML и загрузи файл" (+ [Прикреплён файл]) → {"intents": ["create_namespace", "save_file"]}
"создай пространство Тест и сохрани туда этот файл" (+ [Прикреплён файл]) → {"intents": ["create_namespace", "save_file"]}
"загрузи этот файл в пространство Архив" (+ [Прикреплён файл]) → {"intents": ["save_file"]}
"сохрани этот файл" (+ [Прикреплён файл]) → {"intents": ["save_file"]}
"сохрани эту суммаризацию в Архив" (НЕТ прикреплённого файла) → {"intents": ["save_summary"]}
"сохрани туда это" (НЕТ прикреплённого файла, в истории есть суммаризация) → {"intents": ["save_summary"]}
"перемести файл из Inbox в Работа" → {"intents": ["move_file"]}
"скинь все файлы из пространства Шутки" → {"intents": ["send_file"]}
"что лежит в пространстве Архив?" → {"intents": ["list_files"]}
"какие файлы есть в пространстве Работа?" → {"intents": ["list_files"]}
"о чём файлы в этом пространстве?" → {"intents": ["summarize"]}
"о чём файлы в пространстве Диплом?" → {"intents": ["summarize"]}
"что в файлах этого пространства?" → {"intents": ["summarize"]}
"расскажи о содержимом файлов" → {"intents": ["summarize"]}
"переименуй пространство Работа в Задачи" → {"intents": ["edit_namespace_name"]}
"переименуй файлы в Совет1, Совет2 и Совет3" → {"intents": ["rename_file"]}
"назови файл отчётом" → {"intents": ["rename_file"]}
"удали все файлы из Inbox" → {"intents": ["delete_file"]}
"удали пространство X и удали файл из Y" → {"intents": ["delete_namespace", "delete_file"]}
"удали пространство Тест, а из Архива удали любой файл" → {"intents": ["delete_namespace", "delete_file"]}
"допиши в каждый файл шутку" → {"intents": ["edit_file"]}
"добавь во все файлы пространства X строку Y" → {"intents": ["edit_file"]}
"""


class IntentClassifierService:
    """Классифицирует намерения пользователя через один LLM-вызов с коротким промптом."""

    def __init__(self, llm_service: LLMProvider) -> None:
        self.llm_service = llm_service

    async def classify(
        self,
        question: str,
        *,
        has_file: bool = False,
        has_url: bool = False,
        has_history_url: bool = False,
        has_history_summary: bool = False,
        history: Optional[List[dict]] = None,
    ) -> List[str]:
        if not question.strip():
            return [IntentType.GENERAL_CHAT.value]

        user_text = self._build_user_text(
            question,
            has_file=has_file,
            has_url=has_url,
            has_history_url=has_history_url,
            has_history_summary=has_history_summary,
            history=history,
        )

        messages = [
            {"role": "system", "text": _SYSTEM_PROMPT},
            {"role": "user", "text": user_text},
        ]

        try:
            raw = await self.llm_service.complete(messages, temperature=0.0, max_tokens=256)
            logger.info("[IntentClassifier] Question: %r | Raw: %r", question[:80], raw)
            intents = self._parse_intents(raw)
            if intents is None:
                logger.warning("[IntentClassifier] Parse failed, falling back to rag_query")
                return [IntentType.RAG_QUERY.value]
            logger.info("[IntentClassifier] Intents: %s", " → ".join(intents))
            return intents
        except Exception as exc:
            logger.error("[IntentClassifier] LLM call failed: %s", exc)
            return [IntentType.RAG_QUERY.value]

    def _build_user_text(
        self,
        question: str,
        *,
        has_file: bool,
        has_url: bool,
        has_history_url: bool,
        has_history_summary: bool,
        history: Optional[List[dict]],
    ) -> str:
        context_parts = []
        if has_file:
            context_parts.append("Прикреплён файл.")
        if has_url:
            context_parts.append("В сообщении есть ссылка.")
        if has_history_url and not has_url:
            context_parts.append("В истории есть ссылка (уже сохранена).")
        if has_history_summary:
            context_parts.append("В истории есть суммаризация от ассистента.")

        history_block = ""
        if history:
            recent = history[-_HISTORY_CONTEXT_LIMIT:]
            lines = []
            for msg in recent:
                role = "assistant" if (msg.get("role") or "").strip().lower() == "assistant" else "user"
                text = (msg.get("text") or "").strip()
                if text:
                    lines.append(f"  {role}: {text[:150]}")
            if lines:
                history_block = "[История диалога]\n" + "\n".join(lines) + "\n\n"

        user_text = question
        if context_parts:
            user_text = f"[Контекст: {' '.join(context_parts)}]\n{question}"
        if history_block:
            user_text = history_block + user_text

        return user_text

    @staticmethod
    def _parse_intents(raw: str) -> Optional[List[str]]:
        import re
        text = raw.strip()
        md_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if md_match:
            text = md_match.group(1).strip()

        brace_match = re.search(r"\{[\s\S]*\}", text)
        if not brace_match:
            return None

        try:
            data = json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            return None

        intents_raw = data.get("intents")
        if not isinstance(intents_raw, list):
            if isinstance(intents_raw, str):
                intents_raw = [intents_raw]
            else:
                return None

        intents = [i.strip() for i in intents_raw if isinstance(i, str) and i.strip() in _VALID_INTENTS]
        return intents if intents else None

