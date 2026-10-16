"""Cost controls: monthly token budget per user and a per-user rate limit."""
from datetime import timedelta

from fastapi import HTTPException
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from .config import settings
from .db import SessionLocal
from .metrics import month_start, next_month_start, retry_after_seconds
from .models import RateEvent, UsageLog, User, UserLimit, utcnow


def effective_budget(db: Session, user: User) -> int:
    """Tokens per month; 0 means unlimited."""
    override = db.get(UserLimit, user.id)
    if override is not None and override.monthly_token_budget is not None:
        return override.monthly_token_budget
    return 0 if user.role == "admin" else settings.default_monthly_token_budget


def budget_status(db: Session, user: User) -> dict:
    used = db.scalar(
        select(func.coalesce(func.sum(UsageLog.input_tokens + UsageLog.output_tokens), 0))
        .where(UsageLog.user_id == user.id, UsageLog.ts >= month_start())
    ) or 0
    limit = effective_budget(db, user)
    return {
        "used_tokens": int(used),
        "limit_tokens": limit,  # 0 = unlimited
        "remaining_tokens": None if limit == 0 else max(0, limit - int(used)),
        "percent_used": None if limit == 0 else round(100 * used / limit, 1),
        "resets_at": next_month_start().isoformat(),
    }


def _over_message(s: dict) -> str:
    return (f"Monthly token budget reached ({s['used_tokens']:,} of {s['limit_tokens']:,} tokens). "
            f"It resets on {s['resets_at'][:10]} (UTC); ask an admin for more.")


def enforce_budget(db: Session, user: User) -> None:
    s = budget_status(db, user)
    if s["limit_tokens"] and s["used_tokens"] >= s["limit_tokens"]:
        raise HTTPException(429, _over_message(s))


def budget_block_message(user: User) -> str | None:
    """For background work (batch items, sentiment lines): a message if the budget is used up.
    Calls already in flight are only logged when they finish, so a batch can overshoot by a few calls."""
    with SessionLocal() as db:
        s = budget_status(db, user)
    return _over_message(s) if s["limit_tokens"] and s["used_tokens"] >= s["limit_tokens"] else None


def enforce_rate_limit(db: Session, user: User, weight: int = 1) -> None:
    limit = settings.rate_limit_per_minute
    if not limit:
        return
    if weight > limit:
        raise HTTPException(400, f"Too many items at once ({weight}); the limit is {limit} per minute.")
    now = utcnow()
    window_start = now - timedelta(seconds=60)
    used, oldest = db.execute(
        select(func.coalesce(func.sum(RateEvent.weight), 0), func.min(RateEvent.ts))
        .where(RateEvent.user_id == user.id, RateEvent.ts >= window_start)
    ).one()
    if used + weight > limit:
        wait = retry_after_seconds(oldest, now) if oldest else 1
        raise HTTPException(
            429, f"Rate limit: {limit} items per minute ({used} used). Try again in {wait}s.",
            headers={"Retry-After": str(wait)},
        )
    db.add(RateEvent(user_id=user.id, weight=weight))
    db.execute(delete(RateEvent).where(RateEvent.ts < now - timedelta(hours=1)))  # keep the table small
    db.commit()
