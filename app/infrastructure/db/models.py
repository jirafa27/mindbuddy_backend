from datetime import datetime
from typing import Optional
from sqlalchemy import String, Integer, DateTime, JSON, ForeignKey, Text, Boolean, UniqueConstraint, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from app.infrastructure.db.base import Base
from app.schemas.file import CommandType, CommandStatus


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
    chats: Mapped[list["Chat"]] = relationship("Chat", back_populates="user")


class Namespace(Base):
    __tablename__ = "namespaces"
    __table_args__ = (
        Index(
            "uq_namespaces_root_name",
            "user_id", "name",
            unique=True,
            postgresql_where="parent_id IS NULL",
        ),
        Index(
            "uq_namespaces_child_name",
            "user_id", "parent_id", "name",
            unique=True,
            postgresql_where="parent_id IS NOT NULL",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    parent_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("namespaces.id"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, default="regular", index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    user: Mapped["User"] = relationship("User", back_populates="namespaces")
    parent: Mapped[Optional["Namespace"]] = relationship(
        "Namespace",
        remote_side="Namespace.id",
        back_populates="children",
        foreign_keys=[parent_id],
    )
    children: Mapped[list["Namespace"]] = relationship(
        "Namespace",
        back_populates="parent",
        cascade="all, delete-orphan",
        single_parent=True,
    )
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
        # Два частичных уникальных индекса вместо одного:
        # 1) для файлов без пространства (namespace_id IS NULL)
        Index(
            "uq_user_file_no_ns",
            "user_id", "file_id",
            unique=True,
            postgresql_where="namespace_id IS NULL",
        ),
        # 2) для файлов в конкретном пространстве (namespace_id IS NOT NULL)
        Index(
            "uq_user_file_with_ns",
            "user_id", "file_id", "namespace_id",
            unique=True,
            postgresql_where="namespace_id IS NOT NULL",
        ),
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
    vault_relative_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    desktop_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    app_updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_update_source: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    is_conflict_copy: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    conflict_origin_user_file_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("user_files.id"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    user: Mapped["User"] = relationship("User", back_populates="user_files")
    file: Mapped["File"] = relationship("File", back_populates="user_files")
    namespace: Mapped[Optional["Namespace"]] = relationship(
        "Namespace", back_populates="user_files"
    )
    conflict_origin: Mapped[Optional["UserFile"]] = relationship(
        "UserFile",
        remote_side="UserFile.id",
        foreign_keys=[conflict_origin_user_file_id],
    )


class SyncCommand(Base):
    __tablename__ = "sync_commands"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    user_file_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("user_files.id", ondelete="SET NULL"), nullable=True, index=True
    )
    command_type: Mapped[CommandType] = mapped_column(String(32), nullable=False, index=True)
    payload_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[CommandStatus] = mapped_column(String(32), nullable=False, default=CommandStatus.PENDING.value, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )
    acked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class VectorEmbedding(Base):
    __tablename__ = "vector_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    file_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("files.id"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(
        Vector(3584), nullable=False
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


class Chat(Base):
    __tablename__ = "chats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False, index=True
    )
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    pending_action: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    context: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    user: Mapped["User"] = relationship("User", back_populates="chats")
    messages: Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage", back_populates="chat", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    chat_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("chats.id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)  # user | assistant
    text: Mapped[str] = mapped_column(Text, nullable=False)
    file_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    namespace_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("namespaces.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    chat: Mapped["Chat"] = relationship("Chat", back_populates="messages")
