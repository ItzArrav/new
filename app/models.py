from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    role: Mapped[str] = mapped_column(String(16), default="user")  # "admin" | "user"
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AdminRequest(Base):
    """A user asking to become an admin. Only a current admin can approve or deny it; every request is kept."""

    __tablename__ = "admin_requests"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending | approved | denied
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    decided_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    decision_note: Mapped[str | None] = mapped_column(Text, nullable=True)


class UsageLog(Base):
    """One row per LLM call (including retries and errors)."""

    __tablename__ = "usage_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    agent: Mapped[str] = mapped_column(String(64), index=True)
    model: Mapped[str] = mapped_column(String(128))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16))  # "success" | "error"
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    # False if the response carried no usage data (never silently estimated)
    usage_reported: Mapped[bool] = mapped_column(Boolean, default=True)
    raw_usage: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON


# ---- Tables for later steps (defined now so the schema is stable) ----


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    filename: Mapped[str] = mapped_column(String(256))
    result_json: Mapped[str] = mapped_column(Text)  # validated agent output
    # Extracted CV text (personal data!). Only used server-side, e.g. to write feedback about this CV; never sent
    # to the browser. NULL for CVs classified before this column existed.
    cv_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Quality-check result (JSON) computed automatically with the classification; NULL if unavailable or older CV.
    quality_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The one AI-written feedback comment about this CV (with its quote and sentiment); NULL until generated.
    feedback_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Role-fit results keyed by role name (JSON); NULL until someone checks a role.
    role_fit_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Decision(Base):
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    recommendation_id: Mapped[int] = mapped_column(ForeignKey("recommendations.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    original_json: Mapped[str] = mapped_column(Text)
    decision: Mapped[str] = mapped_column(String(16))  # accept | modify | reject
    final_categories_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)  # required for modify/reject
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    status: Mapped[str] = mapped_column(String(16), default="running")
    total: Mapped[int] = mapped_column(Integer, default=0)
    done: Mapped[int] = mapped_column(Integer, default=0)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class JobItem(Base):
    """One file inside a batch upload."""

    __tablename__ = "job_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), index=True)
    filename: Mapped[str] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending | done | error
    recommendation_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class SentimentResult(Base):
    __tablename__ = "sentiment_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    text: Mapped[str] = mapped_column(Text)
    result_json: Mapped[str] = mapped_column(Text)
    # "user" = feedback a person pasted or uploaded (real data, counts in time saved);
    # "ai_sample" = feedback the AI wrote about a CV (synthetic demo data, never counted in time saved)
    source: Mapped[str] = mapped_column(String(16), default="user", server_default="user")
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ExternalUsage(Base):
    """Rows imported from another agent's log file (separate dashboard tab)."""

    __tablename__ = "external_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime)
    user: Mapped[str] = mapped_column(String(128))
    agent: Mapped[str] = mapped_column(String(128))
    model: Mapped[str] = mapped_column(String(128))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16))
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)


class JobRole(Base):
    """A job added for role fit, on top of the 8 built-in roles (those live in code, see agents/role_fit.py).
    Usually drafted by the AI from a job description and reviewed by the person who adds it. Shared by every user;
    the person who added it or an admin can delete it. Saved role-fit results keep their copy of the checklist."""

    __tablename__ = "job_roles"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    definition: Mapped[str] = mapped_column(Text)  # one-line profile, used in the role-fit prompt
    requirements_json: Mapped[str] = mapped_column(Text)  # [{id, weight, text}], weight 2 = must-have, 1 = nice-to-have
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class UserLimit(Base):
    """Per-user override of the monthly token budget (NULL = use the default; 0 = unlimited)."""

    __tablename__ = "user_limits"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), primary_key=True)
    monthly_token_budget: Mapped[int | None] = mapped_column(Integer, nullable=True)


class RateEvent(Base):
    """One row per admitted agent request (weight = number of items), for rate limiting.
    Stored in the database so every uvicorn worker sees the same counts."""

    __tablename__ = "rate_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    weight: Mapped[int] = mapped_column(Integer, default=1)


class SentEmail(Base):
    """Tracks candidate reply emails (interviews, rejections, shortlists) sent both individually and in batch."""

    __tablename__ = "sent_emails"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    recommendation_id: Mapped[int | None] = mapped_column(ForeignKey("recommendations.id"), nullable=True, index=True)
    recipient_email: Mapped[str] = mapped_column(String(256), index=True)
    candidate_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    subject: Mapped[str] = mapped_column(String(256))
    body: Mapped[str] = mapped_column(Text)
    template_type: Mapped[str] = mapped_column(String(32), default="custom")  # interview | shortlist | reject | custom
    status: Mapped[str] = mapped_column(String(16), default="sent")  # sent | simulated | failed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

