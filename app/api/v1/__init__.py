"""API v1: роутеры (files, namespaces, users, watcher, chat)."""
from fastapi import APIRouter

from . import files, namespaces, users, watcher, chat

api_router = APIRouter()

api_router.include_router(files.router, prefix="/files", tags=["files"])
api_router.include_router(namespaces.router, prefix="/namespaces", tags=["namespaces"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(watcher.router, prefix="/watcher", tags=["watcher"])
api_router.include_router(chat.router, tags=["chat"])
