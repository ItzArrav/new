import json

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db import SessionLocal
from app.main import app
from app.models import Recommendation


def _login(c, name="alice"):
    tok = c.post("/auth/login", data={"username": name, "password": "test-pass"}).json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


def test_classify_stores_cv_text_but_never_returns_it(monkeypatch):
    from app import routes_cv
    result = {"primary_category": "Other", "rationale": "r", "evidence": [], "secondary_categories": [],
              "scores": [], "injection_detected": False}
    monkeypatch.setattr(routes_cv, "classify_cv", lambda u, n, t: {"ok": True, "result": result, "error": None})
    c = TestClient(app)
    h = _login(c)
    r = c.post("/cv/classify", files={"file": ("cv.txt", b"Led a team of five Rust engineers.", "text/plain")}, headers=h)
    assert r.status_code == 200
    with SessionLocal() as db:
        assert db.scalars(select(Recommendation)).one().cv_text == "Led a team of five Rust engineers."
    listing = c.get("/cv/recommendations", headers=h).json()
    # the stored text itself is never exposed (quality examples may quote single lines, which is intended)
    assert "cv_text" not in json.dumps(listing) and "Led a team of five Rust engineers." not in listing[0]["result"].values()


def test_missing_cv_text_column_is_added_to_an_old_database():
    from sqlalchemy import inspect, text

    from app.db import _add_missing_columns, engine
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE recommendations DROP COLUMN cv_text"))
    engine.dispose()  # pooled SQLite connections can keep a stale view of the table after DROP COLUMN
    assert "cv_text" not in {c["name"] for c in inspect(engine).get_columns("recommendations")}
    _add_missing_columns()
    assert "cv_text" in {c["name"] for c in inspect(engine).get_columns("recommendations")}
