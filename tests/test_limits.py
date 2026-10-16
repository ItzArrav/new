from datetime import timedelta

from fastapi.testclient import TestClient
from sqlalchemy import select

from app import routes_batch
from app.config import settings
from app.db import SessionLocal
from app.main import app
from app.metrics import month_start
from app.models import UsageLog, User, UserLimit, utcnow

client = TestClient(app)


def uid(name):
    with SessionLocal() as db:
        return db.scalar(select(User.id).where(User.username == name))


def auth(name):
    r = client.post("/auth/login", data={"username": name, "password": "test-pass"})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def add_usage(name, tokens, when=None, run_id=None):
    with SessionLocal() as db:
        db.add(UsageLog(run_id=run_id or f"u-{name}-{tokens}-{when}", user_id=uid(name), agent="cv_agent", model="m",
                        input_tokens=tokens, output_tokens=0, latency_ms=1, status="success",
                        cost_usd=0.0, ts=when or utcnow()))
        db.commit()


def set_budget(name, tokens):
    with SessionLocal() as db:
        db.merge(UserLimit(user_id=uid(name), monthly_token_budget=tokens))
        db.commit()


def test_budget_blocks_only_the_user_who_used_it_up():
    set_budget("alice", 100)
    add_usage("alice", 100)
    assert client.post("/dev/ping", headers=auth("alice")).status_code == 429
    assert "budget" in client.post("/dev/ping", headers=auth("alice")).json()["detail"].lower()
    assert client.post("/dev/ping", headers=auth("bob")).status_code == 200  # bob has the default budget


def test_last_months_usage_does_not_count():
    set_budget("alice", 100)
    add_usage("alice", 5000, when=month_start() - timedelta(days=2), run_id="old")  # previous month
    add_usage("alice", 50, run_id="new")
    assert client.post("/dev/ping", headers=auth("alice")).status_code == 200
    b = client.get("/me/budget", headers=auth("alice")).json()
    assert b["limit_tokens"] == 100 and b["used_tokens"] >= 50 and b["used_tokens"] < 5000


def test_admin_can_override_and_users_cannot():
    set_budget("alice", 100)
    add_usage("alice", 100)
    assert client.put(f"/admin/users/{uid('alice')}/budget", json={"monthly_token_budget": 0},
                      headers=auth("bob")).status_code == 403
    r = client.put(f"/admin/users/{uid('alice')}/budget", json={"monthly_token_budget": 0}, headers=auth("admin"))
    assert r.status_code == 200 and r.json()["effective_limit_tokens"] == 0  # unlimited
    assert client.post("/dev/ping", headers=auth("alice")).status_code == 200
    assert client.put(f"/admin/users/{uid('alice')}/budget", json={"monthly_token_budget": -5},
                      headers=auth("admin")).status_code == 422


def test_rate_limit_is_per_user_and_has_retry_after(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 3)
    h = auth("alice")
    assert [client.post("/dev/ping", headers=h).status_code for _ in range(3)] == [200, 200, 200]
    r = client.post("/dev/ping", headers=h)
    assert r.status_code == 429 and 1 <= int(r.headers["Retry-After"]) <= 60
    assert client.post("/dev/ping", headers=auth("bob")).status_code == 200


def test_batch_counts_each_file_against_the_rate_limit(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)
    files = [("files", (f"{i}.txt", b"cv text", "text/plain")) for i in range(3)]
    r = client.post("/cv/batch", files=files, headers=auth("alice"))
    assert r.status_code == 400 and "limit" in r.json()["detail"]  # 3 items can never fit in a limit of 2


def test_batch_stops_spending_when_budget_runs_out(monkeypatch):
    called = []
    monkeypatch.setattr(routes_batch, "budget_block_message", lambda user: "Monthly token budget reached")
    monkeypatch.setattr(routes_batch, "classify_cv", lambda *a: called.append(1))
    files = [("files", ("a.txt", b"cv text", "text/plain")), ("files", ("b.txt", b"cv text", "text/plain"))]
    h = auth("alice")
    job = client.post("/cv/batch", files=files, headers=h).json()["job_id"]
    j = client.get(f"/cv/batch/{job}", headers=h).json()
    assert called == [] and j["status"] == "done_with_errors" and j["done"] == 2
    assert all("budget" in i["error"] for i in j["items"])


def test_dashboard_shows_budgets_scoped_by_role():
    a = client.get("/dashboard/data", headers=auth("alice")).json()["budgets"]
    adm = client.get("/dashboard/data", headers=auth("admin")).json()["budgets"]
    assert [b["user"] for b in a] == ["alice"]
    assert {b["user"] for b in adm} == {"admin", "alice", "bob", "carol"}
