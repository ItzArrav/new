import json

from fastapi.testclient import TestClient
from sqlalchemy import select

from app import routes_sentiment
from app.db import SessionLocal
from app.main import app
from app.models import SentimentResult, User
from app.routes_sentiment import parse_feedback_file


def _hdr(c, name="alice"):
    tok = c.post("/auth/login", data={"username": name, "password": "test-pass"}).json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


def _ok(user, text):
    label = "negative" if "slow" in text else "positive"
    return {"ok": True, "attempts": 1, "error": None,
            "result": {"sentiment": label, "confidence": None, "key_phrases": [text[:10]], "rationale": "r",
                       "injection_detected": False}}


# ---------- reading feedback files ----------

def test_txt_gives_one_feedback_per_non_blank_line():
    data = "Great support team.\n\n  The app is slow   on login. \r\nOkay overall.\n".encode()
    assert parse_feedback_file("f.txt", data) == ["Great support team.", "The app is slow on login.", "Okay overall."]


def test_csv_uses_the_feedback_column_by_name_and_skips_the_header():
    data = 'id,Comment,date\n1,"Loved it, very clear",2026-10-01\n2,"Too slow\nand buggy",2026-10-02\n3,,2026-10-03\n'.encode()
    assert parse_feedback_file("f.csv", data) == ["Loved it, very clear", "Too slow and buggy"]


def test_csv_without_a_known_header_uses_the_first_column_and_keeps_every_row():
    data = "Fast and friendly,5\nNever again,1\n".encode()
    assert parse_feedback_file("f.csv", data) == ["Fast and friendly", "Never again"]


def test_excel_byte_order_mark_is_ignored():
    data = "﻿feedback\nGood service\n".encode("utf-8")
    assert parse_feedback_file("f.csv", data) == ["Good service"]


def test_parse_route_returns_lines_for_review_without_any_llm_call(monkeypatch):
    monkeypatch.setattr(routes_sentiment, "analyze_feedback", lambda u, t: (_ for _ in ()).throw(AssertionError("no LLM")))
    c = TestClient(app)
    h = _hdr(c)
    lines = "\n".join(f"Feedback number {i}" for i in range(60)).encode()
    r = c.post("/sentiment/parse", files={"file": ("f.txt", lines, "text/plain")}, headers=h)
    body = r.json()
    assert r.status_code == 200 and len(body["texts"]) == 50 and body["total"] == 60 and body["truncated"] is True
    assert c.post("/sentiment/parse", files={"file": ("f.docx", b"x", "application/octet-stream")}, headers=h).status_code == 415
    assert c.post("/sentiment/parse", files={"file": ("f.txt", b"\n \n", "text/plain")}, headers=h).status_code == 422
    big = b"x" * (routes_sentiment.MAX_FILE_BYTES + 1)
    assert c.post("/sentiment/parse", files={"file": ("f.txt", big, "text/plain")}, headers=h).status_code == 413
    assert c.post("/sentiment/parse", files={"file": ("f.txt", b"a", "text/plain")}).status_code == 401


# ---------- real feedback vs AI sample feedback ----------

def test_pasted_feedback_is_stored_as_real_and_listed_per_user(monkeypatch):
    monkeypatch.setattr(routes_sentiment, "analyze_feedback", _ok)
    c = TestClient(app)
    h = _hdr(c)
    r = c.post("/sentiment/analyze", json={"texts": ["Great service", "The app is slow"]}, headers=h)
    assert r.status_code == 200 and all(x["ok"] for x in r.json()["results"])
    with SessionLocal() as db:
        assert {x.source for x in db.scalars(select(SentimentResult))} == {"user"}
    mine = c.get("/sentiment/results?source=user", headers=h).json()
    assert {x["text"] for x in mine} == {"Great service", "The app is slow"} and all(x["source"] == "user" for x in mine)
    assert c.get("/sentiment/results?source=ai_sample", headers=h).json() == []
    assert c.get("/sentiment/results?source=user", headers=_hdr(c, "bob")).json() == []  # bob sees only his own
    assert len(c.get("/sentiment/results?source=user", headers=_hdr(c, "admin")).json()) == 2  # admin sees all
    assert c.get("/sentiment/results?source=fake", headers=h).status_code == 422


def _add(username, source, n):
    with SessionLocal() as db:
        uid = db.scalar(select(User.id).where(User.username == username))
        for i in range(n):
            db.add(SentimentResult(user_id=uid, text=f"t{i}", result_json=json.dumps({"sentiment": "positive"}), source=source))
        db.commit()


def test_time_saved_counts_only_real_feedback():
    _add("alice", "user", 4)
    _add("alice", "ai_sample", 10)
    c = TestClient(app)
    ts = c.get("/dashboard/data", headers=_hdr(c)).json()["time_saved"]
    assert ts["basis"]["sentiment_ok"] == 4 and ts["basis"]["ai_sample_feedback_not_counted"] == 10
    assert ts["sentiment_hours"] == round(4 * (1.0 - 0.25) / 60, 3)  # default assumptions: 1 min by hand, 0.25 review


def test_old_database_gets_the_source_column_and_ai_feedback_is_labelled():
    from sqlalchemy import inspect, text

    from app.db import _add_missing_columns, engine
    with SessionLocal() as db:
        uid = db.scalar(select(User.id).where(User.username == "alice"))
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE sentiment_results DROP COLUMN source"))
        conn.execute(text("INSERT INTO recommendations (user_id, filename, result_json, feedback_json, ts) "
                          "VALUES (:u, 'cv.txt', '{}', :f, CURRENT_TIMESTAMP)"),
                     {"u": uid, "f": json.dumps({"feedback": "AI wrote this about the CV."})})
        for t in ("AI wrote this about the CV.", "A real customer comment."):
            conn.execute(text("INSERT INTO sentiment_results (user_id, text, result_json, ts) VALUES (:u, :t, '{}', CURRENT_TIMESTAMP)"),
                         {"u": uid, "t": t})
    engine.dispose()
    assert "source" not in {col["name"] for col in inspect(engine).get_columns("sentiment_results")}
    _add_missing_columns()
    with SessionLocal() as db:
        got = {r.text: r.source for r in db.scalars(select(SentimentResult))}
    assert got == {"AI wrote this about the CV.": "ai_sample", "A real customer comment.": "user"}
