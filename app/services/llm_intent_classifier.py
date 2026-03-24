"""LLMIntentClassifier — разбор намерения пользователя через YandexGPT (JSON-вывод)."""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.core.enums import IntentType
from app.domain.protocols import LLMProvider

logger = logging.getLogger(__name__)

_VALID_INTENTS = {i.value for i in IntentType}
_VALID_SEARCH_MODES = {"by_topic", "by_name", "by_content"}

# Слова-подтверждения — только «да» / «yes». Всё остальное считается отказом.
_CONFIRM_WORDS = {"да", "yes"}

# Количество последних сообщений истории, включаемых в контекст классификатора
_HISTORY_CONTEXT_LIMIT = 4

_SYSTEM_PROMPT = """\
Ты классификатор намерений пользователя в приложении-базе знаний.
Верни ТОЛЬКО JSON без пояснений и markdown-блоков.

JSON-поля:
intent, search_query, namespace_hint, search_mode, entity_name, entity_description, entity_content

intent — одно из: rag_query | send_file | list_files | summarize | save_summary | save_file | index_url | create_file | edit_file | delete_file | create_namespace | edit_namespace | delete_namespace | move_file | general_chat

КЛЮЧЕВЫЕ ПРАВИЛА (в порядке приоритета):

1. send_file — пользователь хочет ПОЛУЧИТЬ/НАЙТИ/СКАЧАТЬ сам файл.
   Слова-триггеры: "найди файл", "найди документ", "найди реферат/конспект/презентацию/pdf", "скинь файл", "отправь файл", "дай файл", "скачать", "есть ли у меня файл/конспект/реферат".
   search_mode: by_name (если указано название), by_topic (если тема), by_content (если содержимое).
   ВАЖНО: "найди файл про X" и "найди реферат про X" → send_file, НЕ rag_query.

2. rag_query — пользователь ищет ИНФОРМАЦИЮ из документов.
   Слова-триггеры: "что написано про X", "расскажи про X", "что я писал про X", "найди информацию про X", "о чём файл", "что в файле".
   ВАЖНО: "найди что написано" → rag_query; "найди файл/документ" → send_file.

3. summarize — создать краткое содержание ПРЯМО СЕЙЧАС.
   Слова-триггеры: "суммаризируй", "сделай суммаризацию", "сделай краткое содержание", "сделай пересказ".

4. save_summary — сохранить УЖЕ ГОТОВЫЙ ответ ассистента как файл.
   Признак: местоимение "это/эту/этот" + "сохрани/запиши" + ссылка на предыдущий ответ.
   entity_content = null (контент берётся из истории).
   entity_name — название файла, если указано ("назови X", "файл X").
   НЕ используй если нет местоимения или если пользователь говорит "скинь/отправь/дай" (→ send_file).

5. list_files — показать список файлов: "покажи файлы", "перечисли документы".

6. create_file — создать заметку/файл из нового текста пользователя.
   entity_content = текст файла, entity_name = заголовок.

7. save_file — сохранить ТОЛЬКО ЧТО ЗАГРУЖЕННЫЙ файл (к сообщению прикреплён файл).
   ВАЖНО: если файл не прикреплён, но пользователь просит сохранить/переместить файл по имени → move_file, НЕ save_file.

8. move_file — переместить СУЩЕСТВУЮЩИЙ файл (уже в системе) в другое пространство.
   Признак: нет прикреплённого файла, но есть название файла + пространство назначения.
   entity_name / search_query = название файла, namespace_hint = пространство назначения.
   Примеры: "сохрани его в КВА", "перемести этот файл в Архив", "положи документ в Учёба".

9. CRUD пространств: create_namespace, edit_namespace, delete_namespace.
   create_namespace: entity_name = название (namespace_hint = null).
   edit_namespace: namespace_hint = редактируемое пространство; entity_description = новое описание; entity_content = новое название (при переименовании).
   delete_namespace: namespace_hint = удаляемое пространство.

10. edit_file — изменить содержимое существующего файла.
   entity_content = КРАТКАЯ ИНСТРУКЦИЯ по изменению (НЕ полный текст файла!).
   Примеры инструкций: "добавь в начало: "Привет", "замени X на Y", "удали абзац про Z", "добавь в конец: текст".
   entity_name / search_query = название файла для поиска.
   Слова-триггеры: "допиши", "добавь в файл", "измени файл", "отредактируй", "в начало/конец файла".
   ВАЖНО: "допиши в файл X" → edit_file, НЕ create_file.

11. general_chat — всё остальное: приветствия, болтовня.

Если сомневаешься между rag_query и send_file:
- пользователь просит ПОЛУЧИТЬ/НАЙТИ/СКАЧАТЬ сам файл (без конкретного вопроса о содержимом) → send_file
- пользователь задаёт вопрос или просит достать информацию «из файла/документа» → rag_query
- «скинь/отправь» + «из этого файла/документа» → rag_query (запрос контента, не файла)

search_query — поисковый запрос из ТЕКУЩЕГО вопроса пользователя.
Историю диалога используй для раскрытия местоимений ("это", "этот", "его", "её", "там"), явно неполных вопросов, а также для понимания подтверждений и отказов на вопросы ассистента.
Если вопрос конкретный и самодостаточный — НЕ замещай search_query темами из предыдущих ответов ассистента.

Местоимения ("это", "этот", "его", "её") — раскрой через историю диалога.
entity_description = null если пользователь явно не указал описание.

Примеры:
"найди файл про инди разработку" → {"intent":"send_file","search_query":"инди разработка","namespace_hint":null,"search_mode":"by_topic","entity_name":null,"entity_description":null,"entity_content":null}
"найди реферат про машинное обучение" → {"intent":"send_file","search_query":"машинное обучение","namespace_hint":null,"search_mode":"by_topic","entity_name":null,"entity_description":null,"entity_content":null}
"есть ли у меня конспект по питону" → {"intent":"send_file","search_query":"питон python","namespace_hint":null,"search_mode":"by_topic","entity_name":null,"entity_description":null,"entity_content":null}
"найди в пространстве Inbox реферат на тему инди разработки" → {"intent":"send_file","search_query":"инди разработка","namespace_hint":"Inbox","search_mode":"by_topic","entity_name":null,"entity_description":null,"entity_content":null}
"скинь файл с названием отчёт_2024" → {"intent":"send_file","search_query":"отчёт_2024","namespace_hint":null,"search_mode":"by_name","entity_name":null,"entity_description":null,"entity_content":null}
"скинь суммаризацию" → {"intent":"send_file","search_query":"суммаризация","namespace_hint":null,"search_mode":"by_topic","entity_name":null,"entity_description":null,"entity_content":null}
"найди что я писала про инди разработку" → {"intent":"rag_query","search_query":"инди разработка","namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}
"поищи в пространстве Работа что я писала про ИБ" → {"intent":"rag_query","search_query":"информационная безопасность","namespace_hint":"Работа","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}
"покажи мои файлы в пространстве Учёба" → {"intent":"list_files","search_query":null,"namespace_hint":"Учёба","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}
"суммаризируй этот файл" → {"intent":"summarize","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}
"сделай краткое содержание" → {"intent":"summarize","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}
"создай заметку Идеи: нужно изучить Python и Rust" → {"intent":"create_file","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":"Идеи","entity_description":null,"entity_content":"нужно изучить Python и Rust"}
"запиши в пространство Inbox: встреча в пятницу в 15:00" → {"intent":"create_file","search_query":null,"namespace_hint":"Inbox","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":"встреча в пятницу в 15:00"}
"удали файл отчёт_2024.pdf" → {"intent":"delete_file","search_query":"отчёт_2024.pdf","namespace_hint":null,"search_mode":null,"entity_name":"отчёт_2024.pdf","entity_description":null,"entity_content":null}
"измени заметку Идеи: добавь Go в список языков" → {"intent":"edit_file","search_query":"Идеи","namespace_hint":null,"search_mode":null,"entity_name":"Идеи","entity_description":null,"entity_content":"добавь Go в список языков"}
"Отредактируй файл Экзамен. Нужно в начале файла написать: КВАААААААААА" → {"intent":"edit_file","search_query":"Экзамен","namespace_hint":null,"search_mode":null,"entity_name":"Экзамен","entity_description":null,"entity_content":"добавь в начало файла: КВАААААААААА"}
"создай пространство Учёба с описанием университетские материалы" → {"intent":"create_namespace","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":"Учёба","entity_description":"университетские материалы","entity_content":null}
"удали пространство Работа" → {"intent":"delete_namespace","search_query":null,"namespace_hint":"Работа","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}
"переименуй пространство Пурум в Архив" → {"intent":"edit_namespace","search_query":null,"namespace_hint":"Пурум","search_mode":null,"entity_name":"Пурум","entity_description":null,"entity_content":"Архив"}
"измени описание пространства Пурум на: Ква ква ква" → {"intent":"edit_namespace","search_query":null,"namespace_hint":"Пурум","search_mode":null,"entity_name":"Пурум","entity_description":"Ква ква ква","entity_content":null}
"перемести файл отчёт.pdf в пространство Архив" → {"intent":"move_file","search_query":"отчёт.pdf","namespace_hint":"Архив","search_mode":null,"entity_name":"отчёт.pdf","entity_description":null,"entity_content":null}
"привет" → {"intent":"general_chat","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}

Примеры save_file vs move_file:
[Контекст: К сообщению прикреплён файл.]
user: "Сохрани его в пространство КВА" → {"intent":"save_file","search_query":null,"namespace_hint":"КВА","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}

[Активный файл: КурсовойпроектАвс.pdf (пространство: Inbox)]
user: "Сохрани его в пространство КВА" → {"intent":"move_file","search_query":"КурсовойпроектАвс.pdf","namespace_hint":"КВА","search_mode":null,"entity_name":"КурсовойпроектАвс.pdf","entity_description":null,"entity_content":null}

[Активный файл: отчёт_2024.docx (пространство: Inbox)]
user: "Перемести в Архив" → {"intent":"move_file","search_query":"отчёт_2024.docx","namespace_hint":"Архив","search_mode":null,"entity_name":"отчёт_2024.docx","entity_description":null,"entity_content":null}

Примеры с историей диалога:
[История] assistant: У вас есть учебно-методическое пособие «Предпринимательство в информационной сфере»...
user: "Скинь это пособие" → {"intent":"send_file","search_query":"предпринимательство в информационной сфере","namespace_hint":null,"search_mode":"by_topic","entity_name":null,"entity_description":null,"entity_content":null}

[История] assistant: Что в них написано? — ответ про SWOT-анализ и ИТ-проекты
user: "Что в них написано?" → {"intent":"rag_query","search_query":"SWOT-анализ бизнес-модель экономическая эффективность ИТ","namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}

[История] assistant: Суммаризация: Инди-разработка — небольшие команды без крупных издателей...
user: "Сохрани эту суммаризацию в пространство КВА" → {"intent":"save_summary","search_query":null,"namespace_hint":"КВА","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}

[История] assistant: Суммаризация: Инди-разработка...
user: "Сохрани суммаризацию в пространство КВА. Файл назови Суммаризация Реферата" → {"intent":"save_summary","search_query":null,"namespace_hint":"КВА","search_mode":null,"entity_name":"Суммаризация Реферата","entity_description":null,"entity_content":null}

[История] assistant: Краткий итог: Python — интерпретируемый язык...
user: "Запиши это в пространство Учёба" → {"intent":"save_summary","search_query":null,"namespace_hint":"Учёба","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}

[История] user: "Расскажи про аналитическую часть" → assistant: "В разделе 3.2.2 описана аналитическая часть..."
user: "Какие группы принято выделять среди участников рынка?" → {"intent":"rag_query","search_query":"группы участников рынка","namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}

[История] assistant: Вот ваш файл «Экзамен.docx». Нажмите, чтобы скачать.
user: "Что такое вырожденное решение?" → {"intent":"rag_query","search_query":"вырожденное решение","namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}

[История] assistant: Вот ваш файл «Лекция_5.pdf». Нажмите, чтобы скачать.
user: "Какие методы оптимизации там описаны?" → {"intent":"rag_query","search_query":"методы оптимизации","namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}

[История] assistant: "Изучил контент и добавил в вашу базу знаний. Long Now Foundation — Википедия. Хотите сделать краткое резюме?"
user: "Да" → {"intent":"summarize","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}

[История] assistant: "Хотите суммаризировать этот файл?"
user: "Да, давай" → {"intent":"summarize","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}

[История] assistant: "Хотите сделать краткое содержание?"
user: "Нет" → {"intent":"general_chat","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}

[История] user: "https://ru.wikipedia.org/wiki/... Письма к госпоже Каландрини — Википедия"
user: "Суммаризируй" → {"intent":"summarize","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}

[История] assistant: "Изучил контент и добавил в базу знаний."
user: "Сделай краткое содержание" → {"intent":"summarize","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}

[История] assistant: "Изучил контент и добавил в базу знаний. Статья о Python."
user: "Сделай краткое содержание и сохрани в пространство Учёба" → {"actions":[{"intent":"summarize","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null},{"intent":"create_file","search_query":null,"namespace_hint":"Учёба","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}

[Контекст: К сообщению прикреплён файл.]
user: "Скинь требования к содержанию пояснительной записки из этого файла" → {"intent":"rag_query","search_query":"требования к содержанию пояснительной записки","namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}

[Контекст: К сообщению прикреплён файл.]
user: "Что там написано про архитектуру системы из этого документа?" → {"intent":"rag_query","search_query":"архитектура системы","namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}

МУЛЬТИ-ДЕЙСТВИЕ:
Если пользователь просит выполнить НЕСКОЛЬКО ОТДЕЛЬНЫХ операций в одном сообщении, верни JSON с полем "actions" — массивом, где каждый элемент содержит те же поля что и для одного действия.
Используй для CRUD-операций (create_file, create_namespace, move_file, edit_file, delete_file, edit_namespace, delete_namespace) и для pipeline-цепочек с URL.
НЕ используй для rag_query, summarize (без URL), send_file, general_chat.
КРИТИЧНО: включай в "actions" ТОЛЬКО операции, явно перечисленные в ТЕКУЩЕМ сообщении пользователя. НЕ добавляй операции из истории диалога и НЕ домысливай дополнительные действия.
КРИТИЧНО: если пользователь говорит только "Суммаризируй" / "Сделай краткое содержание" без указания "сохрани/добавь в пространство" — это ОДИН интент summarize, НЕ multi_action.

Pipeline-цепочки (когда в сообщении есть URL):
- "сохрани содержимое / добавь текст статьи в пространство X" → одно действие index_url с namespace_hint=X
- "сохрани саммари / краткое / резюме статьи в пространство X" → цепочка из трёх шагов: index_url → summarize → create_file(namespace_hint=X)
- если пространство X **новое** (пользователь явно говорит "новое пространство") — добавь create_namespace(entity_name=X) ПЕРВЫМ шагом, затем обычная цепочка

Примеры pipeline-цепочек:
[Контекст: В сообщении есть URL]
user: "https://ru.wikipedia.org/wiki/... Добавь текст этой статьи в пространство Статьи" → {"actions":[{"intent":"index_url","search_query":null,"namespace_hint":"Статьи","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}

[Контекст: В сообщении есть URL]
user: "https://ru.wikipedia.org/wiki/... Добавь текст в новое пространство Имя_пространства" → {"actions":[{"intent":"create_namespace","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":"ХИ","entity_description":null,"entity_content":null},{"intent":"index_url","search_query":null,"namespace_hint":"ХИ","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}

[Контекст: В сообщении есть URL]
user: "https://ru.wikipedia.org/wiki/... Сохрани саммари этой статьи в пространство Статьи" → {"actions":[{"intent":"index_url","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null},{"intent":"summarize","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null},{"intent":"create_file","search_query":null,"namespace_hint":"Статьи","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}

[Контекст: В сообщении есть URL]
user: "https://ru.wikipedia.org/wiki/... Добавь суммаризацию в новое пространство Имя_пространства" → {"actions":[{"intent":"create_namespace","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":"ХИ","entity_description":null,"entity_content":null},{"intent":"index_url","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null},{"intent":"summarize","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null},{"intent":"create_file","search_query":null,"namespace_hint":"ХИ","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}

[Контекст: В сообщении есть URL]
user: "https://example.com/article Сделай краткое содержание и сохрани в пространство Работа" → {"actions":[{"intent":"index_url","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null},{"intent":"summarize","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null},{"intent":"create_file","search_query":null,"namespace_hint":"Работа","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}

Примеры CRUD мульти-действия:
"Создай файлы Шутка1, Шутка2, Шутка3 в пространстве Юмор" → {"actions":[{"intent":"create_file","search_query":null,"namespace_hint":"Юмор","search_mode":null,"entity_name":"Шутка1","entity_description":null,"entity_content":null},{"intent":"create_file","search_query":null,"namespace_hint":"Юмор","search_mode":null,"entity_name":"Шутка2","entity_description":null,"entity_content":null},{"intent":"create_file","search_query":null,"namespace_hint":"Юмор","search_mode":null,"entity_name":"Шутка3","entity_description":null,"entity_content":null}]}
"Создай пространства Работа и Учёба" → {"actions":[{"intent":"create_namespace","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":"Работа","entity_description":null,"entity_content":null},{"intent":"create_namespace","search_query":null,"namespace_hint":null,"search_mode":null,"entity_name":"Учёба","entity_description":null,"entity_content":null}]}
"В пространстве Архив создай файл Итоги и удали пространство Черновики" → {"actions":[{"intent":"create_file","search_query":null,"namespace_hint":"Архив","search_mode":null,"entity_name":"Итоги","entity_description":null,"entity_content":null},{"intent":"delete_namespace","search_query":null,"namespace_hint":"Черновики","search_mode":null,"entity_name":null,"entity_description":null,"entity_content":null}]}
"Удали файлы Шутка1, Шутка2 и Шутка3 из пространства Юмор" → {"actions":[{"intent":"delete_file","search_query":"Шутка1","namespace_hint":"Юмор","search_mode":null,"entity_name":"Шутка1","entity_description":null,"entity_content":null},{"intent":"delete_file","search_query":"Шутка2","namespace_hint":"Юмор","search_mode":null,"entity_name":"Шутка2","entity_description":null,"entity_content":null},{"intent":"delete_file","search_query":"Шутка3","namespace_hint":"Юмор","search_mode":null,"entity_name":"Шутка3","entity_description":null,"entity_content":null}]}
"Допиши в файлы Совет1.md, Совет2.md и Совет3.md по дополнительному совету" → {"actions":[{"intent":"edit_file","search_query":"Совет1.md","namespace_hint":null,"search_mode":null,"entity_name":"Совет1.md","entity_description":null,"entity_content":"добавь в конец дополнительный совет"},{"intent":"edit_file","search_query":"Совет2.md","namespace_hint":null,"search_mode":null,"entity_name":"Совет2.md","entity_description":null,"entity_content":"добавь в конец дополнительный совет"},{"intent":"edit_file","search_query":"Совет3.md","namespace_hint":null,"search_mode":null,"entity_name":"Совет3.md","entity_description":null,"entity_content":"добавь в конец дополнительный совет"}]}
ВАЖНО для мульти-действия: "удали" → delete_file или delete_namespace. НИКОГДА не используй create_file для запросов с "удали/удалить/стереть".
Если пользователь говорит "удали ВСЕ файлы" без перечисления конкретных имён — используй ОДИН интент delete_namespace для удаления всего пространства, а НЕ пытайся угадать список файлов.
"""


@dataclass
class ParsedIntent:
    intent: str
    search_query: Optional[str] = None
    namespace_hint: Optional[str] = None
    search_mode: Optional[str] = None
    entity_name: Optional[str] = None
    entity_description: Optional[str] = None
    entity_content: Optional[str] = None
    actions: Optional[list] = field(default=None)  # List[ParsedIntent] для multi_action


def _make_fallback(question: str) -> ParsedIntent:
    return ParsedIntent(
        intent=IntentType.RAG_QUERY.value,
        search_query=question or None,
        namespace_hint=None,
        search_mode="by_topic",
    )


def _try_extract_intent_from_raw(raw: str) -> Optional[str]:
    """Пытается вытащить значение intent из сырой строки через regex, когда JSON сломан."""
    m = re.search(r'"intent"\s*:\s*"([^"]+)"', raw)
    if m:
        value = m.group(1)
        return value if value in _VALID_INTENTS else None
    return None


def _try_extract_str_field(raw: str, field: str) -> Optional[str]:
    """Вытаскивает строковое поле из сырого JSON через regex."""
    m = re.search(r'"' + field + r'"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    return m.group(1) if m else None


def _sanitize_json_string(raw: str) -> str:
    """
    Заменяет управляющие символы (перевод строки, таб и т.п.) внутри
    JSON-строковых значений на их escape-последовательности, чтобы исправить
    ошибку «Invalid control character».
    Работает до парсинга — просто чистит сырой текст.
    """
    def replace_controls(m: re.Match) -> str:
        return m.group(0).replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

    # Заменяем управляющие символы только внутри строковых значений JSON
    return re.sub(r'"(?:[^"\\]|\\.)*"', replace_controls, raw)


def _parse_json(raw: str, question: str) -> ParsedIntent:
    """Извлекает и валидирует JSON из ответа LLM."""
    text = raw.strip()

    # Убираем markdown-блок ```json ... ``` если есть
    md_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if md_match:
        text = md_match.group(1).strip()

    # Ищем первый {...} блок
    brace_match = re.search(r"\{[\s\S]*\}", text)
    data: Optional[dict] = None

    if not brace_match:
        logger.warning("[LLMIntentClassifier] No JSON object found in response: %r", raw[:200])
    else:
        json_str = brace_match.group(0)

        # Попытка 1: парсим как есть
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.warning("[LLMIntentClassifier] JSON parse error: %s | raw: %r", exc, raw[:200])

        # Попытка 2: чистим управляющие символы внутри строк
        if data is None:
            try:
                data = json.loads(_sanitize_json_string(json_str))
                logger.info("[LLMIntentClassifier] Parsed JSON after sanitizing control chars")
            except json.JSONDecodeError:
                pass

        # Попытка 3: обрезаем лишние данные после первого корректного объекта
        if data is None:
            try:
                decoder = json.JSONDecoder()
                data, _ = decoder.raw_decode(_sanitize_json_string(text))
                logger.info("[LLMIntentClassifier] Parsed JSON via raw_decode")
            except json.JSONDecodeError:
                pass

    # Если JSON так и не распарсился — пробуем вытащить intent regex-ом
    if data is None:
        intent_raw = _try_extract_intent_from_raw(raw)
        if intent_raw:
            logger.warning(
                "[LLMIntentClassifier] JSON broken, but recovered intent=%r via regex", intent_raw
            )
            return ParsedIntent(
                intent=intent_raw,
                search_query=_try_extract_str_field(raw, "search_query"),
                namespace_hint=_try_extract_str_field(raw, "namespace_hint"),
                search_mode=None,
                entity_name=_try_extract_str_field(raw, "entity_name"),
                entity_description=_try_extract_str_field(raw, "entity_description"),
                entity_content=_try_extract_str_field(raw, "entity_content"),
            )
        return _make_fallback(question)

    intent = data.get("intent", "")

    # Проверяем multi-action формат: {"actions": [...]} — с intent или без
    actions_raw = data.get("actions")
    if isinstance(actions_raw, list) and actions_raw:
        sub_actions = _parse_actions_list(actions_raw)
        if sub_actions:
            logger.info("[LLMIntentClassifier] Parsed multi_action with %d actions", len(sub_actions))
            return ParsedIntent(intent="multi_action", actions=sub_actions)

    if intent not in _VALID_INTENTS:
        logger.warning("[LLMIntentClassifier] Unknown intent '%s', falling back", intent)
        return _make_fallback(question)

    search_mode = data.get("search_mode") or None
    if search_mode and search_mode not in _VALID_SEARCH_MODES:
        search_mode = "by_topic"

    return ParsedIntent(
        intent=intent,
        search_query=data.get("search_query") or None,
        namespace_hint=data.get("namespace_hint") or None,
        search_mode=search_mode,
        entity_name=data.get("entity_name") or None,
        entity_description=data.get("entity_description") or None,
        entity_content=data.get("entity_content") or None,
    )


def _parse_actions_list(actions_raw: list) -> list:
    """Разбирает список действий из multi-action JSON."""
    result = []
    for item in actions_raw:
        if not isinstance(item, dict):
            continue
        item_intent = item.get("intent", "")
        if item_intent not in _VALID_INTENTS:
            continue
        item_search_mode = item.get("search_mode") or None
        if item_search_mode and item_search_mode not in _VALID_SEARCH_MODES:
            item_search_mode = "by_topic"
        result.append(ParsedIntent(
            intent=item_intent,
            search_query=item.get("search_query") or None,
            namespace_hint=item.get("namespace_hint") or None,
            search_mode=item_search_mode,
            entity_name=item.get("entity_name") or None,
            entity_description=item.get("entity_description") or None,
            entity_content=item.get("entity_content") or None,
        ))
    return result


class LLMIntentClassifier:
    """
    Классификатор намерений на базе YandexGPT.

    Делает один LLM-вызов, возвращает ParsedIntent с:
    - intent: тип намерения
    - search_query: очищенный поисковый запрос
    - namespace_hint: имя пространства из текста
    - search_mode: режим поиска для send_file
    """

    def __init__(self, llm_service: LLMProvider) -> None:
        self.llm_service = llm_service

    @staticmethod
    def is_confirmation(text: str) -> bool:
        """Возвращает True если текст — явное «да» / «yes». Всё остальное считается отказом."""
        normalized = text.strip().lower().rstrip("!.,")
        return normalized in _CONFIRM_WORDS

    async def parse(
        self,
        question: str,
        *,
        has_file: bool = False,
        has_url: bool = False,
        has_history_url: bool = False,
        history: Optional[List[dict]] = None,
        active_file_context: Optional[str] = None,
        history_limit: Optional[int] = None,
        namespace_files_context: Optional[str] = None,
    ) -> ParsedIntent:
        """
        Разбирает запрос пользователя.

        has_file / has_url — контекст текущего сообщения.
        has_history_url — в истории есть URL, но в текущем сообщении его нет.
        history — последние сообщения чата, позволяют разрешить
        местоимения ("это пособие", "скинь его") через контекст диалога.
        active_file_context — строка вида "filename.pdf (пространство: Inbox)",
        помогает LLM понять референт местоимений типа "его", "этот файл".
        history_limit — переопределяет _HISTORY_CONTEXT_LIMIT (например, 1-2 для multi-action).
        namespace_files_context — список файлов выбранного пространства,
        позволяет LLM использовать реальные имена при "каждый файл", "все файлы".
        """
        if not question.strip():
            return _make_fallback(question)

        context_parts = []
        if has_file:
            context_parts.append("К сообщению прикреплён файл.")
        if has_url:
            context_parts.append("В сообщении есть ссылка.")
        if has_history_url and not has_url:
            context_parts.append(
                "В истории диалога есть ссылка, но в текущем сообщении ссылки нет — "
                "не нужно делать index_url; используй summarize, rag_query или другой подходящий интент."
            )
        context = " ".join(context_parts)

        # Формируем блок с историей для разрешения местоимённых ссылок
        history_block = ""
        if history:
            limit = history_limit if history_limit is not None else _HISTORY_CONTEXT_LIMIT
            recent = history[-limit:] if limit > 0 else []
            lines = []
            for msg in recent:
                role = (msg.get("role") or "user").strip().lower()
                text = (msg.get("text") or "").strip()
                if text:
                    label = "assistant" if role == "assistant" else "user"
                    lines.append(f"  {label}: {text}")
            if lines:
                history_block = "[История диалога]\n" + "\n".join(lines) + "\n\n"

        # Блок с активным файлом
        active_file_block = ""
        if active_file_context:
            active_file_block = f"[Активный файл: {active_file_context}]\n"

        # Блок с файлами выбранного пространства
        ns_files_block = ""
        if namespace_files_context:
            ns_files_block = f"[Файлы выбранного пространства: {namespace_files_context}]\n"

        user_text = question
        if context:
            user_text = f"[Контекст: {context}]\n{question}"
        if active_file_block:
            user_text = active_file_block + user_text
        if ns_files_block:
            user_text = ns_files_block + user_text
        if history_block:
            user_text = history_block + user_text

        messages = [
            {"role": "system", "text": _SYSTEM_PROMPT},
            {"role": "user", "text": user_text},
        ]

        try:
            raw = await self.llm_service.complete(
                messages, temperature=0.0, max_tokens=1024
            )
            logger.info("[LLMIntentClassifier] Question: %r | Raw: %r", question[:80], raw[:200])
            result = _parse_json(raw, question)
            logger.info(
                "[LLMIntentClassifier] intent=%s search_query=%r namespace_hint=%r search_mode=%s",
                result.intent, result.search_query, result.namespace_hint, result.search_mode,
            )
            return result
        except Exception as exc:
            logger.error("[LLMIntentClassifier] LLM call failed: %s", exc)
            return _make_fallback(question)
