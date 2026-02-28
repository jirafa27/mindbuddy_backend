"""
Универсальные классы ответов API.
"""
from pydantic import BaseModel, Field, model_validator
from typing import Generic, TypeVar, List, Optional
from datetime import datetime


T = TypeVar("T")


class APIError(BaseModel):
    """Информация об ошибке"""
    detail: str = Field(..., description="Детальное описание ошибки")


class PaginationInfo(BaseModel):
    """
    Информация о пагинации.
    
    Поля total_pages, has_next, has_previous вычисляются автоматически
    на основе total, page и page_size.
    
    Использование:
        pagination = PaginationInfo(total=100, page=2, page_size=20)
        # total_pages=5, has_next=True, has_previous=True - вычислятся сами
    """
    total: int = Field(0, description="Общее количество элементов")
    page: int = Field(1, ge=1, description="Текущая страница")
    page_size: int = Field(20, ge=1, description="Размер страницы")
    total_pages: int = Field(0, description="Общее количество страниц")
    has_next: bool = Field(False, description="Есть ли следующая страница")
    has_previous: bool = Field(False, description="Есть ли предыдущая страница")

    @model_validator(mode='after')
    def compute_pagination_fields(self) -> "PaginationInfo":
        """Автоматически вычисляет total_pages, has_next, has_previous"""
        if self.page_size > 0:
            self.total_pages = (self.total + self.page_size - 1) // self.page_size
        else:
            self.total_pages = 0
        self.has_next = self.page < self.total_pages
        self.has_previous = self.page > 1
        return self

class ResponseMessage(BaseModel, Generic[T]):
    """Базовый ответ API"""
    status: str = Field("success", description="Статус: 'success' или 'error'")
    message: str = Field("", description="Сообщение о результате операции")
    data: Optional[T] = Field(None, description="Данные ответа")
    error: Optional[APIError] = Field(None, description="Информация об ошибке (если status='error')")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Время ответа")


class ListResponseData(BaseModel, Generic[T]):
    """Данные ответа со списком объектов и пагинацией"""
    items: List[T] = Field(default_factory=list, description="Список элементов")
    pagination: PaginationInfo = Field(..., description="Информация о пагинации")


class HistoryMessage(BaseModel):
    """Сообщение из истории чата."""
    role: str = Field(..., description="Роль: 'user' или 'assistant'")
    text: str = Field(..., description="Текст сообщения")
    file_id: Optional[int] = Field(None, description="ID файла, если к сообщению был приложен файл")

