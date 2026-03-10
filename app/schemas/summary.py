"""Схемы для суммаризации."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.core.enums import SummaryMethod


@dataclass
class SummaryResult:
    """Результат суммаризации от агента (LLM)."""
    content: str
    model_name: str
    method: SummaryMethod
    chunks_processed: int


@dataclass
class ContentToSummarize:
    """Контент, подготовленный сервисом для суммаризации (текст уже получен, файл сохранён)."""
    text: str
    title: str
    source_url: Optional[str]
    content_file_id: int  # для save_summary
    user_file_id: Optional[int]


class SummaryRequest(BaseModel):
    """Запрос на суммаризацию."""
    url: Optional[str] = Field(None, description="URL источника (YouTube, веб-страница)")


class SummaryResponse(BaseModel):
    """Ответ с результатом суммаризации."""
    user_file_id: Optional[int]
    content_file_id: int
    summary: str
    title: str
    source_url: Optional[str] = None
    is_cached: bool = False
    method: SummaryMethod = SummaryMethod.STUFFING
    
    model_config = {"from_attributes": True}


class SummaryInfo(BaseModel):
    """Информация о суммаризации."""
    model_config = {"from_attributes": True, "protected_namespaces": ()}

    user_file_id: int
    content_file_id: int
    file_id: int
    content: str
    model_name: str
    created_at: datetime
    updated_at: datetime


class SummaryCreateResult(BaseModel):
    """Результат создания суммаризации (внутренний)."""
    summary_id: int
    user_file_id: int
    content_file_id: int
    content: str
    is_new: bool
