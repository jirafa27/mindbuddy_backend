"""Эндпоинты для управления фоновыми задачами (статус, результаты)."""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, Any, Literal

from app.schemas.base import ResponseMessage
from app.schemas.user import UserResponse
from app.core.dependencies import get_current_user
from app.infrastructure.workers.celery_app import celery_app

router = APIRouter()


class TaskStatusResponse(BaseModel):
    """Статус фоновой Celery-задачи."""
    task_id: str = Field(..., description="ID задачи")
    status: Literal["pending", "started", "progress", "success", "failure", "retry", "revoked"] = Field(
        ..., description="Статус задачи (pending|started|progress|success|failure|retry|revoked)"
    )
    result: Optional[Any] = Field(None, description="Результат (только для success)")
    error: Optional[str] = Field(None, description="Сообщение об ошибке (только для failure)")
    traceback: Optional[str] = Field(None, description="Полное описание ошибки (только для failure)")


@router.get("/tasks/{task_id}/status", response_model=ResponseMessage[TaskStatusResponse])
async def get_task_status(
    task_id: str,
    user: UserResponse = Depends(get_current_user),
) -> ResponseMessage[TaskStatusResponse]:
    """Получить статус фоновой задачи по ID.
    
    Возвращает текущий статус, результат (если успех) или ошибку.
    """
    try:
        result = celery_app.AsyncResult(task_id)
        
        response_data: dict[str, Any] = {
            "task_id": task_id,
            "status": result.status,
        }
        
        if result.status == "success":
            response_data["result"] = result.result
        elif result.status in ["failure", "retry"]:
            response_data["error"] = str(result.info) if result.info else "Unknown error"
            if hasattr(result, "traceback"):
                response_data["traceback"] = result.traceback
        
        return ResponseMessage(
            data=TaskStatusResponse(**response_data),
            message="Task status retrieved successfully",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}"
        )
