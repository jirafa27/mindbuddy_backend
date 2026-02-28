from datetime import datetime
from typing import Optional
from sqlalchemy import String, Integer, BigInteger, DateTime, JSON, ForeignKey, Text, Boolean, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from app.infrastructure.db.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True, index=True)
    password_hash: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
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
    user_files: Mapped[list["UserFile"]] = relationship("UserFile", back_populates="user")


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
    user_files: Mapped[list["UserFile"]] = relationship(
        "UserFile", back_populates="namespace", cascade="all, delete-orphan"
    )


class File(Base):
    __tablename__ = "files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    source_url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True, index=True)
    transcript_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    file_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)  # MinIO path for uploads
    media_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # duration, author, title, etc.
    processing_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    user_files: Mapped[list["UserFile"]] = relationship(
        "UserFile", back_populates="file", cascade="all, delete-orphan"
    )
    embeddings: Mapped[list["VectorEmbedding"]] = relationship(
        "VectorEmbedding", back_populates="file", cascade="all, delete-orphan"
    )
    summaries: Mapped[list["Summary"]] = relationship(
        "Summary", back_populates="file", cascade="all, delete-orphan"
    )



class UserFile(Base):
    __tablename__ = "user_files"
    __table_args__ = (
        UniqueConstraint("user_id", "file_id", name="uq_user_file"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    file_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("files.id"), nullable=False, index=True
    )
    namespace_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("namespaces.id"), nullable=True, index=True
    )
    custom_title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    user: Mapped["User"] = relationship("User", back_populates="user_files")
    file: Mapped["File"] = relationship("File", back_populates="user_files")
    namespace: Mapped[Optional["Namespace"]] = relationship(
        "Namespace", back_populates="user_files"
    )


class VectorEmbedding(Base):
    __tablename__ = "vector_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    file_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("files.id"), nullable=False, index=True
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


class Summary(Base):
    __tablename__ = "summaries"
    __table_args__ = (
        UniqueConstraint("file_id", "lookup_key", name="uq_summary_file_lookup"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    file_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("files.id"), nullable=False, index=True
    )
    lookup_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    text: Mapped[str] = mapped_column("content", Text, nullable=False)
    used_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, default="yandexgpt")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    file: Mapped["File"] = relationship("File", back_populates="summaries")

    @property
    def content(self) -> str:
        return self.text
