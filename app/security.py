from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from .config import SECRET_HELP, jwt_secret_problem, settings
from .db import get_db
from .models import User

oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def _secret() -> str:
    """The signing secret, refused if unsafe (also guarded at startup; this covers any path that skips startup)."""
    problem = jwt_secret_problem(settings.jwt_secret)
    if problem:
        raise RuntimeError(f"{problem}. {SECRET_HELP}")
    return settings.jwt_secret


def create_token(user: User) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_minutes)
    return jwt.encode({"sub": str(user.id), "exp": exp}, _secret(), algorithm="HS256")


def get_current_user(token: str = Depends(oauth2), db: Session = Depends(get_db)) -> User:
    """The ONLY source of user identity. Never read a user id from the URL or body."""
    unauthorized = HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    try:
        payload = jwt.decode(token, _secret(), algorithms=["HS256"])
        user = db.get(User, int(payload["sub"]))
    except (jwt.PyJWTError, KeyError, ValueError):
        raise unauthorized
    if user is None:
        raise unauthorized
    return user  # role is read from the DB, so role changes apply immediately


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin only")
    return user


def scope_to_user(stmt, user_id_column, user: User):
    """Every query on per-user data goes through this: admin sees all, user sees own."""
    return stmt if user.role == "admin" else stmt.where(user_id_column == user.id)
