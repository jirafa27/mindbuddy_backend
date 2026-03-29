"""PlannerService — LLM генерирует список шагов (план) для выполнения запроса пользователя."""
import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from app.core.enums import IntentType
from app.domain.protocols import LLMProvider

logger = logging.getLogger(__name__)

_VALID_TOOLS = {i.value for i in IntentType}
_HISTORY_CONTEXT_LIMIT = 4

_SYSTEM_PROMPT = """\
КРИТИЧНО: ты возвращаешь ТОЛЬКО JSON. Никогда не пиши текст, объяснения или уточнения вместо JSON.
Даже если запрос кажется уже выполненным, неоднозначным или ты хочешь что-то уточнить — всё равно верни JSON.

Ты планировщик задач в приложении-базе знаний.
Верни ТОЛЬКО JSON без пояснений и markdown-блоков: {"steps": [...]}

Каждый шаг — объект с полями:
tool, namespace_hint, search_query, search_mode, entity_name, entity_description, entity_content

tool — одно из:
rag_query | send_file | list_files | summarize | save_summary | save_file | index_url |
create_file | edit_file | rename_file | delete_file | create_namespace | edit_namespace_name |
edit_namespace_description | delete_namespace | move_file | general_chat

Описание инструментов:
- rag_query: поиск КОНКРЕТНОЙ ИНФОРМАЦИИ из документов. Используй когда пользователь спрашивает "что написано про X", "найди информацию о Y", "расскажи про Z". search_query = суть вопроса.
- send_file: ОТПРАВИТЬ/СКАЧАТЬ файл пользователю (пользователь хочет ПОЛУЧИТЬ файл).
  КЛЮЧЕВЫЕ СЛОВА: "скинь", "отправь", "пришли", "дай", "скачай", "поделись", "вышли", "кинь".
  "скинь файлы из пространства X" = send_file (НЕ move_file!).
  "отправь все из этого пространства" = send_file (НЕ move_file!).
  search_query = что ищем, search_mode: by_name / by_topic / all_in_namespace.
  Все файлы из пространства: search_query = null, search_mode = "all_in_namespace", namespace_hint = X.
- list_files: ПОКАЗАТЬ СПИСОК файлов (только текст, без отправки). КЛЮЧЕВЫЕ СЛОВА: "покажи", "что лежит", "перечисли", "какие файлы есть". namespace_hint = пространство (если указано).
- summarize: создать обзор/краткое содержание файла или URL. Используй когда пользователь спрашивает "о чём файл/этот файл/эти файлы", "что содержит файл", "расскажи о содержимом". НЕ использовать для поиска конкретных фактов.
- save_summary: сохранить УЖЕ ГОТОВОЕ саммари из предыдущего ответа ассистента как файл.
- save_file: сохранить только что загруженный файл.
- index_url: сохранить URL в базу знаний (без суммаризации).
- create_file: создать заметку из текста. entity_content = текст, entity_name = заголовок.
- edit_file: изменить содержимое файла. search_query = имя файла, entity_content = инструкция.
- rename_file: переименовать файл(ы) (только отображаемое имя). Один файл: search_query = текущее имя, entity_name = новое имя. Несколько файлов в пространстве: search_query = null, namespace_hint = пространство, entity_content = новые имена через запятую в порядке файлов (например "Совет1, Совет2, Совет3").
- delete_file: удалить файл. search_query = имя файла.
  Для удаления ВСЕХ файлов из пространства: search_query = null, namespace_hint = название пространства.
- move_file: ПЕРЕМЕСТИТЬ файл ИЗ одного пространства В другое (требуются ДВА пространства!).
  КЛЮЧЕВЫЕ СЛОВА: "перемести", "перенеси", "помести", "переложи".
  ВАЖНО: если указано только ОДНО пространство — это НЕ move_file! "скинь из X" = send_file, "перемести из X в Y" = move_file.
  Один файл: search_query = имя файла, namespace_hint = куда.
  Все файлы из пространства: search_query = null, entity_name = исходное пространство (откуда), namespace_hint = пространство назначения (куда).
- create_namespace: создать пространство. entity_name = название.
- edit_namespace_name: ТОЛЬКО переименовать пространство. namespace_hint = текущее название, entity_name = новое название.
- edit_namespace_description: ТОЛЬКО изменить описание пространства. namespace_hint = название пространства, entity_description = новое описание.
- delete_namespace: удалить пространство. namespace_hint = какое.
- general_chat: ответить на вопрос без поиска по файлам.

ПРАВИЛА планирования:

0. entity_content для create_file:
   - Если create_file идёт ПОСЛЕ summarize в том же плане — entity_content = null (контент передаётся автоматически).
   - Если create_file АВТОНОМНЫЙ (пользователь просит создать файл с конкретным содержимым, придумать текст, написать заметку) — ЗАПОЛНИ entity_content нужным текстом.
   - entity_description ВСЕГДА = null для всех инструментов кроме edit_namespace_description.

1. Верни ОДИН шаг для простых запросов без явного "сохрани".

2. КРИТИЧНО — summarize ТОЛЬКО показывает резюме, НЕ сохраняет его.
   Если нужно и показать, и сохранить → всегда добавляй create_file шагом после summarize.

3. Цепочки "сохрани резюме/саммари/краткое содержание":
   - "сохрани резюме в пространство X" → [summarize, create_file(ns=X)]
   - "сохрани резюме в НОВОЕ пространство X" → [create_namespace(X), summarize, create_file(ns=X)]
   - "сохрани текст ссылки в пространство X" → [index_url(ns=X)]
   - "сохрани саммари ссылки в пространство X" → [index_url, summarize, create_file(ns=X)]
   - "сохрани саммари в новое пространство X" → [create_namespace(X), summarize, create_file(ns=X)]
   - "сохрани саммари ссылки в НОВОЕ пространство X" → [create_namespace(X), index_url, summarize, create_file(ns=X)]
   - "создай пространство X и добавь туда файл" → [create_namespace(X), save_file(ns=X)]
   - "создай пространство X и сохрани туда эти суммаризации/резюме НЕСКОЛЬКИМИ файлами" →
     [create_namespace(X), summarize, create_file(ns=X), summarize, create_file(ns=X), summarize, create_file(ns=X)]
     (количество пар summarize+create_file = количество файлов в контексте, обычно 2-4)

   КРИТИЧНО: "сохрани СУММАРИЗАЦИЮ/РЕЗЮМЕ/КРАТКОЕ СОДЕРЖАНИЕ" не равно перемещение файла!
   - "суммаризации/резюме/краткое содержание" → СОЗДАТЬ НОВЫЕ файлы с текстом саммари → summarize + create_file
   - "перемести/перенеси/помести ФАЙЛ/ФАЙЛЫ" → переместить существующие файлы → move_file

4. Ключевое слово "НОВОЕ пространство X" → сначала create_namespace(entity_name=X), затем остальные шаги с namespace_hint=X.

5. Контекст ссылки:
   - [Контекст: В сообщении есть ссылка] → можно использовать index_url/summarize.
   - [Контекст: В истории есть ссылка] → ссылка УЖЕ сохранена. Используй summarize/rag_query, НЕ index_url.

6. Контекст файла:
   - [Контекст: Прикреплён файл] → прикреплённый файл — это и есть "этот файл"/"этот документ"/"его".
     Слова "этот", "его", "этот файл", "этот документ" в запросе ВСЕГДА относятся к прикреплённому файлу, НЕ к файлам из истории диалога или пространства.
     Используй save_file (сохранить) или summarize (о чём файл), НЕ create_file.

7. Простая суммаризация без "сохрани":
   - "суммаризируй" / "о чём это?" / "о чём эта статья?" / "о чём этот файл?" → ТОЛЬКО [summarize], БЕЗ create_file.
   - ЗАПРЕЩЕНО добавлять create_file если пользователь просто спрашивает о содержании — он хочет увидеть текст, не сохранить файл.
   - "суммаризируй и сохрани в X" → [summarize, create_file(ns=X)]
   - "сохрани последний ответ ассистента / это саммари / эту суммаризацию в X" → [save_summary(ns=X)]
   - ВАЖНО: если суммаризация уже была сделана ранее в диалоге и пользователь просит
     "сохрани эту суммаризацию" / "сохрани туда это" — используй save_summary, НЕ summarize.

8. Если пользователь говорит "из этого файла", "в этом документе" → rag_query.

9. Как отличить send_file / move_file / list_files:
   - "скинь/отправь/пришли/дай файлы из X" → send_file (одно пространство, пользователь хочет ПОЛУЧИТЬ файлы)
   - "перемести/перенеси файлы из X в Y" → move_file (два пространства, файлы меняют место)
   - "покажи/перечисли что в X" → list_files (пользователь хочет увидеть список)
   Для send_file: search_query = тема/название из ТЕКУЩЕГО вопроса, НЕ имя файла из истории.
   namespace_hint для send_file — ТОЛЬКО если пользователь явно назвал пространство в своём сообщении.
   НЕ брать namespace_hint из истории диалога. Если пространство не упомянуто — namespace_hint=null (поиск по всем файлам).

10. Приоритет контекста:
    - Прикреплённый файл (в сообщении) → «этот файл» / «этот документ» относится к нему.
    - Активное пространство (из контекста [Активное пространство: X]) → «это пространство» / «данное пространство» относится к нему.
    - История диалога — только справочный контекст, не источник для namespace_hint или файлов.

10. Для rag_query: search_query = конкретный вопрос/тема в 2-5 словах.
    НЕ перечисляй несколько имён файлов через запятую — это плохой поисковый запрос.
    НЕ используй rag_query для "о чём файл / о чём эти файлы" — используй summarize.

Примеры:
"найди файл про машинное обучение" → {"steps":[{"tool":"send_file","search_query":"машинное обучение","search_mode":"by_topic","namespace_hint":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"скинь все файлы из пространства Шутки" → {"steps":[{"tool":"send_file","search_query":null,"search_mode":"all_in_namespace","namespace_hint":"Шутки","entity_name":null,"entity_description":null,"entity_content":null}]}
"отправь все файлы из пространства Архив" → {"steps":[{"tool":"send_file","search_query":null,"search_mode":"all_in_namespace","namespace_hint":"Архив","entity_name":null,"entity_description":null,"entity_content":null}]}
"скинь файлы из этого пространства" → {"steps":[{"tool":"send_file","search_query":null,"search_mode":"all_in_namespace","namespace_hint":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"отправь все из этого пространства" → {"steps":[{"tool":"send_file","search_query":null,"search_mode":"all_in_namespace","namespace_hint":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"пришли мне все файлы" → {"steps":[{"tool":"send_file","search_query":null,"search_mode":"all_in_namespace","namespace_hint":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"что написано про Python?" → {"steps":[{"tool":"rag_query","search_query":"Python","namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"суммаризируй" → {"steps":[{"tool":"summarize","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"создай пространство Работа" → {"steps":[{"tool":"create_namespace","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":"Работа","entity_description":null,"entity_content":null}]}
"привет" → {"steps":[{"tool":"general_chat","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"переименуй пространство Работа в Задачи" → {"steps":[{"tool":"edit_namespace_name","search_query":null,"namespace_hint":"Работа","search_mode":null,"entity_name":"Задачи","entity_description":null,"entity_content":null}]}
"переименуй файлы в Совет1, Совет2 и Совет3" → {"steps":[{"tool":"rename_file","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":"Совет1, Совет2, Совет3"}]}
"переименуй файл отчёт в финальный отчёт" → {"steps":[{"tool":"rename_file","search_query":"отчёт","namespace_hint":null,"search_mode":null,"entity_name":"финальный отчёт","entity_description":null,"entity_content":null}]}
"добавь описание «Мои задачи» к пространству Задачи" → {"steps":[{"tool":"edit_namespace_description","search_query":null,"namespace_hint":"Задачи","search_mode":null,"entity_name":null,"entity_description":"Мои задачи","entity_content":null}]}
"сохрани резюме в новое пространство Резюме" → {"steps":[{"tool":"create_namespace","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":"Резюме","entity_description":null,"entity_content":null},{"tool":"summarize","search_query":null,"namespace_hint":"Резюме","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null},{"tool":"create_file","search_query":null,"namespace_hint":"Резюме","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"сохрани краткое содержание в пространство Статьи" → {"steps":[{"tool":"summarize","search_query":null,"namespace_hint":"Статьи","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null},{"tool":"create_file","search_query":null,"namespace_hint":"Статьи","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"сохрани эту суммаризацию в пространство Архив" → {"steps":[{"tool":"save_summary","search_query":null,"namespace_hint":"Архив","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"создай пространство Сводки и сохрани туда эту суммаризацию" → {"steps":[{"tool":"create_namespace","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":"Сводки","entity_description":null,"entity_content":null},{"tool":"save_summary","search_query":null,"namespace_hint":"Сводки","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"сохрани туда это" → {"steps":[{"tool":"save_summary","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"создай пространство Резюме и сохрани туда этот файл" → {"steps":[{"tool":"create_namespace","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":"Резюме","entity_description":null,"entity_content":null},{"tool":"move_file","search_query":null,"namespace_hint":"Резюме","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"создай пространство ML и сохрани туда эти суммаризации тремя отдельными файлами" → {"steps":[{"tool":"create_namespace","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":"ML","entity_description":null,"entity_content":null},{"tool":"summarize","search_query":null,"namespace_hint":"ML","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null},{"tool":"create_file","search_query":null,"namespace_hint":"ML","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null},{"tool":"summarize","search_query":null,"namespace_hint":"ML","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null},{"tool":"create_file","search_query":null,"namespace_hint":"ML","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null},{"tool":"summarize","search_query":null,"namespace_hint":"ML","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null},{"tool":"create_file","search_query":null,"namespace_hint":"ML","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"создай пространство Архив и сохрани туда резюме этих файлов" → {"steps":[{"tool":"create_namespace","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":"Архив","entity_description":null,"entity_content":null},{"tool":"summarize","search_query":null,"namespace_hint":"Архив","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null},{"tool":"create_file","search_query":null,"namespace_hint":"Архив","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null},{"tool":"summarize","search_query":null,"namespace_hint":"Архив","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null},{"tool":"create_file","search_query":null,"namespace_hint":"Архив","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"перемести все файлы из пространства Статьи в пространство Резюме" → {"steps":[{"tool":"move_file","search_query":null,"namespace_hint":"Резюме","search_mode":null,"entity_name":"Статьи","entity_description":null,"entity_content":null}]}
"перенеси все из Архив в Работа" → {"steps":[{"tool":"move_file","search_query":null,"namespace_hint":"Работа","search_mode":null,"entity_name":"Архив","entity_description":null,"entity_content":null}]}
"удали все файлы из пространства Inbox" → {"steps":[{"tool":"delete_file","search_query":null,"namespace_hint":"Inbox","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"удали все из пространства Шутки" → {"steps":[{"tool":"delete_file","search_query":null,"namespace_hint":"Шутки","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"что у меня находится в пространстве ТЕСТ2?" → {"steps":[{"tool":"list_files","search_query":null,"namespace_hint":"ТЕСТ2","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"что лежит в пространстве Архив?" → {"steps":[{"tool":"list_files","search_query":null,"namespace_hint":"Архив","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"о чём этот файл?" → {"steps":[{"tool":"summarize","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"о чём эти файлы?" → {"steps":[{"tool":"summarize","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"что содержит этот документ?" → {"steps":[{"tool":"summarize","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"расскажи о содержимом файла" → {"steps":[{"tool":"summarize","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"""


@dataclass
class PlanStep:
    tool: str
    namespace_hint: Optional[str] = None
    search_query: Optional[str] = None
    search_mode: Optional[str] = None
    entity_name: Optional[str] = None
    entity_description: Optional[str] = None
    entity_content: Optional[str] = None
    search_limit: Optional[int] = None

    def to_action_dict(self) -> dict:
        """Преобразует шаг в формат pending_actions для MultiActionNode."""
        return {
            "intent": self.tool,
            "namespace_id": None,
            "namespace_name_hint": self.namespace_hint,
            "search_query": self.search_query,
            "search_mode": self.search_mode,
            "entity_name": self.entity_name,
            "entity_description": self.entity_description,
            "entity_content": self.entity_content,
            "search_limit": self.search_limit,
        }


def _sanitize_json_string(raw: str) -> str:
    """Очищает управляющие символы внутри JSON-строк."""
    def replace_controls(m: re.Match) -> str:
        return m.group(0).replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return re.sub(r'"(?:[^"\\]|\\.)*"', replace_controls, raw)


def _parse_steps(raw: str) -> List[PlanStep]:
    """Парсит JSON-ответ LLM в список шагов. При ошибке возвращает [rag_query]."""
    text = raw.strip()

    # Убираем markdown-блок если есть
    md_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if md_match:
        text = md_match.group(1).strip()

    # Ищем JSON объект
    brace_match = re.search(r"\{[\s\S]*\}", text)
    data: Optional[dict] = None

    if brace_match:
        json_str = brace_match.group(0)
        for attempt in (json_str, _sanitize_json_string(json_str)):
            try:
                data = json.loads(attempt)
                break
            except json.JSONDecodeError:
                pass

        if data is None:
            try:
                data, _ = json.JSONDecoder().raw_decode(_sanitize_json_string(text))
            except json.JSONDecodeError:
                pass

    if not data or "steps" not in data:
        # Модель вернула одиночный шаг без обёртки: {"tool": "...", ...}
        if data and "tool" in data:
            logger.info("[PlannerService] Recovering single-step JSON without 'steps' wrapper")
            data = {"steps": [data]}
        # Модель вернула steps как объект вместо массива: {"steps": {"tool": ...}}
        elif data and isinstance(data.get("steps"), dict):
            logger.info("[PlannerService] Recovering steps dict → list")
            data = {"steps": [data["steps"]]}
        else:
            logger.warning("[PlannerService] Could not parse steps from: %r", raw[:200])
            return None  # сигнал для retry

    steps = []
    has_summarize = any(
        (item.get("tool") or "") == IntentType.SUMMARIZE.value
        for item in (data.get("steps") or [])
        if isinstance(item, dict)
    )
    for item in data.get("steps") or []:
        if not isinstance(item, dict):
            continue
        tool = (item.get("tool") or "").strip()
        if tool not in _VALID_TOOLS:
            logger.warning("[PlannerService] Unknown tool %r, skipping step", tool)
            continue
        raw_content = item.get("entity_content") or None
        # Зачищаем entity_content только если create_file идёт в пайплайне после summarize
        # (контент передаётся через pipeline_context). В автономных create_file — сохраняем.
        if raw_content and tool == IntentType.CREATE_FILE.value and has_summarize:
            logger.info("[PlannerService] Stripping entity_content from create_file (pipeline after summarize)")
            raw_content = None
        steps.append(PlanStep(
            tool=tool,
            namespace_hint=item.get("namespace_hint") or None,
            search_query=item.get("search_query") or None,
            search_mode=item.get("search_mode") or None,
            entity_name=item.get("entity_name") or None,
            entity_description=item.get("entity_description") or None,
            entity_content=raw_content,
        ))

    return steps or None  # None если все шаги оказались невалидными


_RETRY_SYSTEM_PROMPT = (
    "Верни ТОЛЬКО JSON в формате {\"steps\": [...]}. "
    "Каждый шаг: {\"tool\": \"...\", \"namespace_hint\": null, \"search_query\": null, "
    "\"entity_name\": null, \"entity_description\": null, \"entity_content\": null, \"search_mode\": null}. "
    "Никакого другого текста и ключей."
)

_FILE_OP_DISAMBIGUATE_PROMPT = """\
Определи намерение пользователя. Ответь ОДНИМ словом из трёх вариантов:

send_file — пользователь хочет ПОЛУЧИТЬ/СКАЧАТЬ файлы себе
  Ключевые слова: "скинь", "отправь", "пришли", "дай мне", "скачать", "кинь", "вышли", "поделись"

move_file — пользователь хочет ПЕРЕМЕСТИТЬ файлы из одного пространства в другое
  Ключевые слова: "перемести", "перенеси", "помести", "переложи" + ВСЕГДА указаны два пространства (откуда и куда)

list_files — пользователь хочет ПОСМОТРЕТЬ СПИСОК файлов (не скачать, не переместить)
  Ключевые слова: "покажи", "что лежит", "перечисли", "какие файлы есть", "что там"

Вопрос: {question}

Ответь одним словом: send_file, move_file или list_files."""


class PlannerService:
    """
    Генерирует план выполнения запроса пользователя через LLM.
    Возвращает список шагов (PlanStep), каждый из которых соответствует
    одному инструменту (tool = IntentType).
    """

    def __init__(self, llm_service: LLMProvider) -> None:
        self.llm_service = llm_service

    async def plan(
        self,
        question: str,
        *,
        has_url: bool = False,
        has_history_url: bool = False,
        has_file: bool = False,
        active_file_context: Optional[str] = None,
        active_namespace_name: Optional[str] = None,
        history: Optional[List[dict]] = None,
    ) -> List[PlanStep]:
        """
        Генерирует список шагов для выполнения запроса.

        has_url — URL в текущем сообщении пользователя.
        has_history_url — URL найден в истории, но не в текущем сообщении.
        has_file — к текущему сообщению прикреплён файл.
        active_file_context — контекст активного файла ("filename.pdf (пространство: Inbox)").
        history — последние сообщения чата.
        """
        if not question.strip():
            return [PlanStep(tool=IntentType.GENERAL_CHAT.value)]

        user_text = self._build_user_text(
            question,
            has_url=has_url,
            has_history_url=has_history_url,
            has_file=has_file,
            active_file_context=active_file_context,
            active_namespace_name=active_namespace_name,
            history=history,
        )

        messages = [
            {"role": "system", "text": _SYSTEM_PROMPT},
            {"role": "user", "text": user_text},
        ]

        try:
            raw = await self.llm_service.complete(messages, temperature=0.0, max_tokens=1024)
            logger.info("[PlannerService] Question: %r | Raw: %r", question[:80], raw)
            steps = _parse_steps(raw)
            if steps is None:
                # LLM вернул текст вместо JSON — повторяем, явно указывая намерение
                logger.warning("[PlannerService] Parse failed, retrying with strict format prompt")
                retry_messages = messages + [
                    {"role": "assistant", "text": raw},
                    {
                        "role": "user",
                        "text": (
                            "Ты уже определил намерение пользователя. Теперь верни ТОЛЬКО JSON без какого-либо пояснительного текста.\n"
                            + _RETRY_SYSTEM_PROMPT
                        ),
                    },
                ]
                raw2 = await self.llm_service.complete(retry_messages, temperature=0.0, max_tokens=512)
                logger.info("[PlannerService] Retry raw: %r", raw2)
                steps = _parse_steps(raw2)
            if steps is None:
                logger.error("[PlannerService] Could not parse steps after retry, falling back to rag_query")
                steps = [PlanStep(tool=IntentType.RAG_QUERY.value, search_query=question or None)]
            # Уточняем send_file/move_file/list_files через мини-вызов если план одношаговый
            _AMBIGUOUS = {IntentType.SEND_FILE.value, IntentType.MOVE_FILE.value, IntentType.LIST_FILES.value}
            if len(steps) == 1 and steps[0].tool in _AMBIGUOUS:
                corrected = await self._disambiguate_file_op(question, steps[0].tool)
                if corrected != steps[0].tool:
                    steps[0] = PlanStep(
                        tool=corrected,
                        namespace_hint=steps[0].namespace_hint,
                        search_query=steps[0].search_query,
                        search_mode=steps[0].search_mode or (
                            "all_in_namespace" if corrected == IntentType.SEND_FILE.value else None
                        ),
                        entity_name=steps[0].entity_name,
                        entity_description=steps[0].entity_description,
                        entity_content=steps[0].entity_content,
                    )
            logger.info(
                "[PlannerService] Plan: %s",
                " → ".join(s.tool for s in steps),
            )
            return steps
        except Exception as exc:
            logger.error("[PlannerService] LLM call failed: %s", exc)
            return [PlanStep(
                tool=IntentType.RAG_QUERY.value,
                search_query=question or None,
                search_mode="by_topic",
            )]

    async def _disambiguate_file_op(self, question: str, current_tool: str) -> str:
        """Мини-вызов LLM для уточнения send_file / move_file / list_files."""
        prompt = _FILE_OP_DISAMBIGUATE_PROMPT.format(question=question)
        try:
            raw = await self.llm_service.complete(
                [{"role": "user", "text": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            token = raw.strip().lower().split()[0] if raw.strip() else ""
            valid = {IntentType.SEND_FILE.value, IntentType.MOVE_FILE.value, IntentType.LIST_FILES.value}
            if token in valid:
                if token != current_tool:
                    logger.info("[PlannerService] Disambiguate: %r → %r", current_tool, token)
                return token
            logger.warning("[PlannerService] Disambiguate returned unknown token %r, keeping %r", token, current_tool)
        except Exception as exc:
            logger.warning("[PlannerService] Disambiguate failed: %s", exc)
        return current_tool

    def _build_user_text(
        self,
        question: str,
        *,
        has_url: bool,
        has_history_url: bool,
        has_file: bool,
        active_file_context: Optional[str],
        active_namespace_name: Optional[str] = None,
        history: Optional[List[dict]],
    ) -> str:
        context_parts = []
        if has_file:
            context_parts.append("Прикреплён файл. Слово «этот файл»/«этот документ»/«его» относится к прикреплённому файлу, НЕ к файлам из истории.")
        if has_url:
            context_parts.append("В сообщении есть ссылка.")
        if has_history_url and not has_url:
            context_parts.append(
                "В истории диалога есть ссылка, но в текущем сообщении её нет — "
                "ссылка уже сохранена. Используй summarize/rag_query, НЕ index_url."
            )

        history_block = ""
        if history:
            recent = history[-_HISTORY_CONTEXT_LIMIT:]
            lines = []
            for msg in recent:
                role = (msg.get("role") or "user").strip().lower()
                text = (msg.get("text") or "").strip()
                if text:
                    label = "assistant" if role == "assistant" else "user"
                    lines.append(f"  {label}: {text}")
            if lines:
                history_block = (
                    "[История диалога — только для извлечения контекста (имена файлов, пространства)."
                    " НЕ используй историю как основание для отказа от планирования или замены JSON текстом]\n"
                    + "\n".join(lines) + "\n\n"
                )

        active_file_block = ""
        if active_file_context:
            active_file_block = f"[Активный файл: {active_file_context}]\n"

        active_namespace_block = ""
        if active_namespace_name:
            active_namespace_block = (
                f"[Активное пространство: {active_namespace_name} — "
                f"«это пространство»/«данное пространство» в запросе означает именно его]\n"
            )

        user_text = question
        if context_parts:
            user_text = f"[Контекст: {' '.join(context_parts)}]\n{question}"
        if active_file_block:
            user_text = active_file_block + user_text
        if active_namespace_block:
            user_text = active_namespace_block + user_text
        if history_block:
            user_text = history_block + user_text

        return user_text
