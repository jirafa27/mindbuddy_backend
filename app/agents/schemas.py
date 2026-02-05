"""Схемы запроса и ответа для POST /ask."""
from pydantic import BaseModel, Field
from typing import Optional


class AskRequest(BaseModel):
    """Вход эндпоинта /ask (question, namespace_id; file передаётся отдельно в multipart)."""
    question: str = Field(..., description="Вопрос пользователя")
    namespace_id: int = Field(..., description="ID пространства знаний")
    user_id: int = Field(..., description="ID пользователя (для прав и сохранения файла)")


class SourceItem(BaseModel):
    """Один источник в ответе."""
    filename: str = Field(..., description="Имя файла")
    relevance: float = Field(..., description="Релевантность (0–1)")


class AskResponse(BaseModel):
    """Ответ эндпоинта /ask."""
    answer: str = Field(..., description="Текстовый ответ на вопрос")
    sources: list[SourceItem] = Field(default_factory=list, description="Источники (файлы и релевантность)")
    agent_steps: list[str] = Field(default_factory=list, description="Цепочка агентов, выполнявших запрос")
