from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    """A registered RegCheck account.

    Profile fields mirror the per-run survey (use_case / academic_position /
    research_field) so a logged-in user fills them once and the post-run survey
    can be skipped.
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str | None] = mapped_column(String(320), unique=True, index=True, nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # Public profile slug used in /u/<handle>; unique when set.
    handle: Mapped[str | None] = mapped_column(String(64), unique=True, index=True, nullable=True)
    is_public_profile: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Profile fields (correspond to the survey questions).
    use_case: Mapped[str | None] = mapped_column(String(100), nullable=True)
    academic_position: Mapped[str | None] = mapped_column(String(100), nullable=True)
    research_field: Mapped[str | None] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, server_default=func.now(), nullable=False
    )

    identities: Mapped[list["OAuthIdentity"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    api_keys: Mapped[list["ApiKey"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    reports: Mapped[list["Report"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )


class OAuthIdentity(Base):
    """A federated identity (Google / ORCID) linked to a user. Lets one account
    link multiple providers."""

    __tablename__ = "oauth_identities"
    __table_args__ = (UniqueConstraint("provider", "subject", name="uq_oauth_provider_subject"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="identities")


class ApiKey(Base):
    """An issued RegCheck API key. Only the SHA-256 hash is stored; the plaintext
    token is shown to the user once at creation. `prefix` is a short, non-secret
    fragment for display/identification."""

    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    prefix: Mapped[str] = mapped_column(String(24), index=True, nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    request_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), nullable=False
    )

    user: Mapped["User"] = relationship(back_populates="api_keys")

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None


class Report(Base):
    """Durable ownership/visibility metadata for a report. The report content
    (items, manifest, evidence) lives in Redis under `task_id`. Anonymous runs
    do NOT get a row here — they are Redis-only with a short TTL."""

    __tablename__ = "reports"

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=True
    )
    visibility: Mapped[str] = mapped_column(String(16), default="unlisted", nullable=False)
    title: Mapped[str | None] = mapped_column(String(300), nullable=True)
    comparison_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    source: Mapped[str] = mapped_column(String(16), default="ui", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), index=True, nullable=False
    )

    owner: Mapped["User | None"] = relationship(back_populates="reports")
    shares: Mapped[list["ReportShare"]] = relationship(
        back_populates="report", cascade="all, delete-orphan"
    )


class ReportShare(Base):
    """A grant of view access to a `restricted` report. Each row names one
    grantee by verified email (Google) OR ORCID iD — exactly one is set. A
    viewer is allowed when their account matches a grant (see
    `services.sharing.user_can_view`). Owners are always allowed and need no row.
    """

    __tablename__ = "report_shares"
    __table_args__ = (
        UniqueConstraint("task_id", "grantee_email", name="uq_share_task_email"),
        UniqueConstraint("task_id", "grantee_orcid", name="uq_share_task_orcid"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    task_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("reports.task_id", ondelete="CASCADE"), index=True, nullable=False
    )
    grantee_email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    grantee_orcid: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), nullable=False
    )

    report: Mapped["Report"] = relationship(back_populates="shares")
