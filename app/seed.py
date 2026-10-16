"""Create tables and demo accounts (idempotent). Demo passwords only: change for real use."""
from sqlalchemy import select

from .config import settings
from .db import SessionLocal, init_db
from .models import User
from .security import hash_password

USERS = [("admin", "admin"), ("alice", "user"), ("bob", "user"), ("carol", "user")]


def seed() -> None:
    init_db()
    with SessionLocal() as db:
        for name, role in USERS:
            if db.scalar(select(User).where(User.username == name)) is None:
                db.add(User(username=name, role=role, password_hash=hash_password(settings.seed_password)))
        db.commit()


if __name__ == "__main__":
    seed()
    print("Seeded users:", ", ".join(n for n, _ in USERS))
