"""Схемы запроса и ответа для графа /ask."""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

from app.schemas.summary import SummaryResult


# Типы интентов для override
OverrideIntentType = Literal["summarize", "index_url", "save_file", "rag_query", "send_file"]


class AskRequest(BaseModel):
    """Вход эндпоинта /ask (question, namespace_id; file передаётся отдельно в multipart)."""
    question: str = Field(..., description="Вопрос пользователя")
    namespace_id: int = Field(..., description="ID пространства знаний")
    user_id: int = Field(..., description="ID пользователя (для прав и сохранения файла)")
    override_intent: Optional[OverrideIntentType] = Field(
        None,
        description="Принудительный интент (если задан — RouterNode не анализирует текст)"
    )


class SourceItem(BaseModel):
    """Один источник в ответе."""
    filename: str = Field(..., description="Имя файла")
    relevance: float = Field(..., description="Релевантность (0–1)")
    file_id: Optional[int] = Field(None, description="ID файла для скачивания (только для send_file)")


class AskResponse(BaseModel):
    """Ответ эндпоинта /ask."""
    answer: str = Field(..., description="Текстовый ответ на вопрос")
    sources: list[SourceItem] = Field(default_factory=list, description="Источники (файлы и релевантность)")
    agent_steps: list[str] = Field(default_factory=list, description="Цепочка агентов, выполнявших запрос")
    file_ids: List[int] = Field(default_factory=list, description="Список ID файлов (для скачивания или один созданный)")
    chat_id: Optional[int] = Field(None, description="ID чата (текущий или созданный)")
    task_id: Optional[str] = Field(None, description="ID фоновой задачи (для полинга статуса)")
