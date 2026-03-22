"""API v1: роутеры (auth, files, namespaces, users, chat, summary, content, tasks, user_files)."""
from fastapi import APIRouter

from . import auth, files, namespaces, users, chat, summary, content, tasks, user_files

api_router = APIRouter()

api_router.include_router(auth.router)
api_router.include_router(files.router, prefix="/files", tags=["files"])
api_router.include_router(namespaces.router, prefix="/namespaces", tags=["namespaces"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(user_files.router, prefix="/user-files", tags=["user-files"])
api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(summary.router, tags=["summary"])
api_router.include_router(content.router, tags=["content"])
api_router.include_router(tasks.router, tags=["tasks"])
