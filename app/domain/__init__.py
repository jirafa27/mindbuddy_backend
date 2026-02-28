"""
Доменный слой
Содержит сущности, протоколы и другие компоненты, которые используются в приложении.
"""
from app.domain.entities.user import UserEntity
from app.domain.entities.namespace import NamespaceEntity
from app.domain.entities.file import FileEntity
from app.domain.entities.chunk import ChunkEntity
from app.domain.entities.vector_embedding import VectorEmbeddingEntity
from app.domain.entities.summary import SummaryEntity
from app.domain.entities.search_result import SearchResultRow
from app.domain.entities.parsed_content import ParsedContent

__all__ = [
    "UserEntity",
    "NamespaceEntity",
    "FileEntity",
    "ChunkEntity",
    "VectorEmbeddingEntity",
    "SummaryEntity",
    "SearchResultRow",
    "ParsedContent",
]
