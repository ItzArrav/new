import json
import threading

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.config import settings
from app.db import SessionLocal
from app.llm import FakeChatModel
from app.main import app
from app.models import UsageLog, User
from app.usage import UsageLogger


class FailingModel(FakeChatModel):
    def _generate(self, *a, **k):
        raise RuntimeError("boom")


def _uid(name: str) -> int:
    with SessionLocal() as db:
        return db.scalar(select(User.id).where(User.username == name))


def _jsonl():
    with open(settings.usage_jsonl_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_success_call_is_logged_with_response_tokens():
    FakeChatModel(reply="one two three").invoke(
        "hello there world", config={"callbacks": [UsageLogger(_uid("alice"), "cv_agent")]}
    )
    with SessionLocal() as db:
        row = db.scalars(select(UsageLog)).one()
    assert (row.input_tokens, row.output_tokens) == (3, 3)
    assert row.status == "success" and row.agent == "cv_agent" and row.latency_ms >= 0
    rec = _jsonl()[0]
    assert rec["run_id"] == row.run_id and rec["raw"]["provider_usage"]["input_tokens"] == 3
    assert rec["ts"] == row.ts.isoformat() + "Z"  # the raw log line carries the same call time as the database row


def test_error_call_is_logged():
    try:
        FailingModel().invoke("x", config={"callbacks": [UsageLogger(_uid("bob"), "sentiment_agent")]})
    except RuntimeError:
        pass
    with SessionLocal() as db:
        row = db.scalars(select(UsageLog)).one()
    assert row.status == "error" and "boom" in row.error


def test_two_users_at_once_do_not_mix():
    def work(name, n):
        uid = _uid(name)
        for _ in range(n):
            FakeChatModel().invoke("a b c d", config={"callbacks": [UsageLogger(uid, "cv_agent")]})

    threads = [threading.Thread(target=work, args=(u, 8)) for u in ("alice", "bob")]
    [t.start() for t in threads]
    [t.join() for t in threads]

    with SessionLocal() as db:
        counts = dict(db.execute(select(UsageLog.user_id, func.count()).group_by(UsageLog.user_id)).all())
    assert counts == {_uid("alice"): 8, _uid("bob"): 8}
    assert len(_jsonl()) == 16


def test_api_scoping_user_vs_admin():
    for name, n in (("alice", 2), ("bob", 3)):
        for _ in range(n):
            FakeChatModel().invoke("a b", config={"callbacks": [UsageLogger(_uid(name), "cv_agent")]})

    client = TestClient(app)

    def token(name):
        r = client.post("/auth/login", data={"username": name, "password": "test-pass"})
        assert r.status_code == 200
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    assert client.get("/usage/summary").status_code == 401  # no token -> rejected
    assert client.get("/usage/summary", headers=token("alice")).json()["calls"] == 2
    assert client.get("/usage/summary", headers=token("bob")).json()["calls"] == 3
    assert client.get("/usage/summary", headers=token("admin")).json()["calls"] == 5
    assert client.get("/admin/users", headers=token("alice")).status_code == 403
    bad = client.post("/auth/login", data={"username": "alice", "password": "wrong"})
    assert bad.status_code == 401
