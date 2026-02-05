from datetime import datetime
from typing import Optional
from sqlalchemy import String, Integer, BigInteger, DateTime, JSON, ForeignKey, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from app.infrastructure.db.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    telegram_id: Mapped[Optional[int]] = mapped_column(BigInteger, unique=True, nullable=True, index=True)
    username: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    watcher_token: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    namespaces: Mapped[list["Namespace"]] = relationship("Namespace", back_populates="user")


class Namespace(Base):
    __tablename__ = "namespaces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    user: Mapped["User"] = relationship("User", back_populates="namespaces")
    files: Mapped[list["File"]] = relationship(
        "File", back_populates="namespace", cascade="all, delete-orphan"
    )
    embeddings: Mapped[list["VectorEmbedding"]] = relationship(
        "VectorEmbedding", back_populates="namespace", cascade="all, delete-orphan"
    )


class File(Base):
    __tablename__ = "files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    namespace_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("namespaces.id"), nullable=True, index=True
    )
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    file_type: Mapped[str] = mapped_column(String(50), nullable=False, default="md")
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    namespace: Mapped[Optional["Namespace"]] = relationship("Namespace", back_populates="files")
    embeddings: Mapped[list["VectorEmbedding"]] = relationship(
        "VectorEmbedding", back_populates="file", cascade="all, delete-orphan"
    )


class VectorEmbedding(Base):
    __tablename__ = "vector_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    file_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("files.id"), nullable=False, index=True
    )
    namespace_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("namespaces.id"), nullable=True, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(
        Vector(256), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    file: Mapped["File"] = relationship("File", back_populates="embeddings")
    namespace: Mapped[Optional["Namespace"]] = relationship(
        "Namespace", back_populates="embeddings"
    )
