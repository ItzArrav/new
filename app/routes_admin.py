"""User management. Admins create accounts; a user who wants admin rights files a request that a current admin
must approve. Roles are read from the database on every request, so an approval takes effect immediately."""
import re
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db
from .models import AdminRequest, User
from .security import create_token, get_current_user, hash_password, require_admin

router = APIRouter(tags=["admin"])

USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _clean_username(v: str) -> str:
    v = v.strip().lower()
    if not USERNAME_RE.match(v):
        raise ValueError("username: 3-32 characters, lowercase letters, digits, '.', '_' or '-', starting with a letter or digit")
    return v


class NewUser(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=72)  # 72 bytes is bcrypt's limit
    role: Literal["user", "admin"] = "user"

    @field_validator("username")
    @classmethod
    def _username(cls, v: str) -> str:
        return _clean_username(v)


class SignupIn(BaseModel):
    """Self sign-up. There is deliberately no role field: new accounts are always plain users."""

    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=72)

    @field_validator("username")
    @classmethod
    def _username(cls, v: str) -> str:
        return _clean_username(v)


@router.post("/auth/signup", status_code=201)
def signup(body: SignupIn, db: Session = Depends(get_db)):
    if not settings.allow_signup:
        raise HTTPException(403, "Self sign-up is turned off. Ask an admin to create your account")
    since = _now() - timedelta(hours=1)
    recent = db.scalar(select(func.count(User.id)).where(User.created_at >= since)) or 0
    if recent >= settings.signup_max_per_hour:
        raise HTTPException(429, "Too many new accounts were created recently. Try again later or ask an admin")
    if db.scalar(select(User).where(User.username == body.username)) is not None:
        raise HTTPException(409, f"The username '{body.username}' is already taken")
    user = User(username=body.username, role="user", password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    return {"access_token": create_token(user), "token_type": "bearer", "username": user.username, "role": user.role}


@router.post("/admin/users", status_code=201)
def create_user(body: NewUser, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Create an account. Creating an admin account is itself the permission: only a current admin can do it."""
    if db.scalar(select(User).where(User.username == body.username)) is not None:
        raise HTTPException(409, f"The username '{body.username}' is already taken")
    user = User(username=body.username, role=body.role, password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    return {"id": user.id, "username": user.username, "role": user.role, "created_by": admin.username}


class AdminRequestIn(BaseModel):
    reason: str = Field(min_length=10, max_length=500)


def _request_view(r: AdminRequest, names: dict[int, str]) -> dict:
    return {"id": r.id, "user_id": r.user_id, "username": names.get(r.user_id, f"user #{r.user_id}"),
            "reason": r.reason, "status": r.status, "requested_at": r.ts.isoformat(),
            "decided_by": names.get(r.decided_by) if r.decided_by else None,
            "decided_at": r.decided_at.isoformat() if r.decided_at else None, "decision_note": r.decision_note}


@router.get("/me/admin-request")
def my_admin_request(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """The caller's most recent request (or null), so the UI can show 'pending' or 'denied'."""
    r = db.scalar(select(AdminRequest).where(AdminRequest.user_id == user.id).order_by(AdminRequest.id.desc()))
    names = {u.id: u.username for u in db.scalars(select(User))}
    return {"is_admin": user.role == "admin", "request": _request_view(r, names) if r else None}


@router.post("/me/admin-request", status_code=201)
def request_admin(body: AdminRequestIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if user.role == "admin":
        raise HTTPException(400, "You are already an admin")
    pending = db.scalar(select(AdminRequest).where(AdminRequest.user_id == user.id, AdminRequest.status == "pending"))
    if pending is not None:
        raise HTTPException(409, "You already have a pending request. A current admin has to decide on it first")
    r = AdminRequest(user_id=user.id, reason=body.reason.strip())
    db.add(r)
    db.commit()
    return {"id": r.id, "status": r.status}


@router.get("/admin/admin-requests")
def list_admin_requests(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    names = {u.id: u.username for u in db.scalars(select(User))}
    rows = db.scalars(select(AdminRequest).order_by(AdminRequest.id.desc()).limit(50)).all()
    return [_request_view(r, names) for r in rows]


class Decision(BaseModel):
    note: str | None = Field(default=None, max_length=300)


def _decide(request_id: int, approve: bool, note: str | None, admin: User, db: Session) -> dict:
    r = db.get(AdminRequest, request_id)
    if r is None:
        raise HTTPException(404, "Request not found")
    if r.status != "pending":
        raise HTTPException(409, f"This request was already {r.status}")
    r.status = "approved" if approve else "denied"
    r.decided_by, r.decided_at, r.decision_note = admin.id, _now(), (note or "").strip() or None
    if approve:
        target = db.get(User, r.user_id)
        if target is not None:
            target.role = "admin"
    db.commit()
    return {"id": r.id, "status": r.status}


@router.post("/admin/admin-requests/{request_id}/approve")
def approve(request_id: int, body: Decision | None = None, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    return _decide(request_id, True, body.note if body else None, admin, db)


@router.post("/admin/admin-requests/{request_id}/deny")
def deny(request_id: int, body: Decision | None = None, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    return _decide(request_id, False, body.note if body else None, admin, db)
