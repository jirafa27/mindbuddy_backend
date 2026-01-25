from fastapi import APIRouter
from app.api import files

api_router = APIRouter()

api_router.include_router(files.router, prefix="/files", tags=["files"])

