import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import jwt_secret_problem, settings
from app.db import SessionLocal
from app.main import app
from app.models import User
from app.security import create_token

GOOD = "k3J9vX2mQ7pL1sR8tY4uW6zA0bC5dE7fG9hH2jK4"


@pytest.mark.parametrize("bad", ["", "change-me", "change-me-to-a-long-random-string", "CHANGE-ME-please", "short-secret"])
def test_placeholder_empty_or_short_secrets_are_refused(bad):
    assert jwt_secret_problem(bad)


def test_a_long_random_secret_is_accepted():
    assert jwt_secret_problem(GOOD) is None


def test_the_server_refuses_to_start_with_the_example_secret(monkeypatch):
    monkeypatch.setattr(settings, "jwt_secret", "change-me-to-a-long-random-string")
    with pytest.raises(RuntimeError, match="placeholder"):
        with TestClient(app):
            pass


def test_the_server_starts_with_a_good_secret(monkeypatch):
    monkeypatch.setattr(settings, "jwt_secret", GOOD)
    with TestClient(app) as c:
        assert c.post("/auth/login", data={"username": "alice", "password": "test-pass"}).status_code == 200


def test_tokens_are_never_signed_or_accepted_with_a_bad_secret(monkeypatch):
    with SessionLocal() as db:
        alice = db.scalar(select(User).where(User.username == "alice"))
    monkeypatch.setattr(settings, "jwt_secret", "change-me")
    with pytest.raises(RuntimeError):
        create_token(alice)


def test_a_token_signed_with_another_secret_is_rejected(monkeypatch):
    with SessionLocal() as db:
        alice = db.scalar(select(User).where(User.username == "alice"))
    monkeypatch.setattr(settings, "jwt_secret", GOOD)
    forged = create_token(alice)  # e.g. an attacker who knew an old secret
    monkeypatch.setattr(settings, "jwt_secret", GOOD[::-1])
    c = TestClient(app)
    assert c.get("/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401
