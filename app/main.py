import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import SECRET_HELP, jwt_secret_problem, settings
from .db import get_db, init_db
from .limits import enforce_budget, enforce_rate_limit
from .llm import get_chat_model
from .models import UsageLog, User
from .security import (
    create_token, get_current_user, require_admin, scope_to_user, verify_password,
)
from .routes_admin import router as admin_router
from .routes_batch import router as batch_router
from .routes_cv import router as cv_router
from .routes_dashboard import router as dashboard_router, dev_router
from .routes_feedback import router as feedback_router
from .routes_limits import router as limits_router
from .routes_rolefit import router as rolefit_router
from .routes_roles import router as roles_router
from .routes_decisions import router as decisions_router
from .routes_email import router as email_router
from .routes_sentiment import router as sentiment_router
from .usage import usage_config

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    problem = jwt_secret_problem(settings.jwt_secret)
    if problem:  # refuse to start: with a known secret anyone can sign their own admin login
        raise RuntimeError(f"{problem}. Anyone who knows it can sign their own admin login. {SECRET_HELP}")
    try:
        init_db()  # idempotent; the seed step normally creates tables first
    except Exception:
        log.exception("init_db failed (tables may already exist)")
    yield


app = FastAPI(title="Multi-user agent platform", lifespan=lifespan)
for _r in (cv_router, decisions_router, email_router, batch_router, sentiment_router, feedback_router, rolefit_router, roles_router, dashboard_router, limits_router, admin_router, dev_router):
    app.include_router(_r)



@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


@app.post("/auth/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # usernames are stored lowercase (sign-up and admin-created accounts normalise them), so match the same way
    user = db.scalar(select(User).where(User.username == form.username.strip().lower()))
    if user is None or not verify_password(form.password, user.password_hash):
        raise HTTPException(401, "Incorrect username or password")
    return Token(access_token=create_token(user))


@app.get("/me")
def me(user: User = Depends(get_current_user)):
    return {"id": user.id, "username": user.username, "role": user.role}


@app.post("/dev/ping")
def ping(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Smoke test: one logged LLM call as the authenticated user.
    With LLM_PROVIDER=anthropic this makes a real (tiny) API call."""
    enforce_budget(db, user)
    enforce_rate_limit(db, user)
    reply = get_chat_model().invoke(
        "Reply with the single word: pong", config=usage_config(user, "ping")
    )
    return {"reply": reply.content}


@app.get("/usage/summary")
def usage_summary(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Admin: all users. User: only their own rows. Same code path for both."""
    def agg(*cols):
        return scope_to_user(select(*cols), UsageLog.user_id, user)

    t = db.execute(
        agg(
            func.count(UsageLog.id),
            func.coalesce(func.sum(UsageLog.input_tokens), 0),
            func.coalesce(func.sum(UsageLog.output_tokens), 0),
            func.coalesce(func.sum(UsageLog.cost_usd), 0.0),
        )
    ).one()

    by_user = db.execute(
        agg(
            User.username,
            func.count(UsageLog.id),
            func.sum(UsageLog.input_tokens),
            func.sum(UsageLog.output_tokens),
        )
        .join(User, User.id == UsageLog.user_id)
        .group_by(User.username)
    ).all()

    by_agent = db.execute(
        agg(
            UsageLog.agent,
            func.count(UsageLog.id),
            func.sum(UsageLog.input_tokens),
            func.sum(UsageLog.output_tokens),
        ).group_by(UsageLog.agent)
    ).all()

    return {
        "scope": "all users" if user.role == "admin" else user.username,
        "calls": t[0], "input_tokens": int(t[1]), "output_tokens": int(t[2]),
        "estimated_cost_usd": round(float(t[3]), 6),
        "by_user": [dict(user=r[0], calls=r[1], input=int(r[2] or 0), output=int(r[3] or 0)) for r in by_user],
        "by_agent": [dict(agent=r[0], calls=r[1], input=int(r[2] or 0), output=int(r[3] or 0)) for r in by_agent],
    }


@app.get("/usage/logs")
def usage_logs(
    limit: int = Query(50, le=500),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    stmt = scope_to_user(select(UsageLog), UsageLog.user_id, user)
    rows = db.scalars(stmt.order_by(UsageLog.id.desc()).limit(limit)).all()
    return [
        dict(
            ts=r.ts.isoformat(), user_id=r.user_id, agent=r.agent, model=r.model,
            input_tokens=r.input_tokens, output_tokens=r.output_tokens,
            latency_ms=r.latency_ms, status=r.status, cost_usd=r.cost_usd,
        )
        for r in rows
    ]


@app.get("/admin/users")
def list_users(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    return [dict(id=u.id, username=u.username, role=u.role) for u in db.scalars(select(User))]
