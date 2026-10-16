import json
from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import select

from app import routes_batch
from app.db import SessionLocal
from app.main import app
from app.models import Recommendation, UsageLog, User

client = TestClient(app)
RESULT = {"primary_category": "DCS", "secondary_categories": [], "scores": [], "rationale": "r",
          "evidence": [], "injection_detected": False}


def uid(name):
    with SessionLocal() as db:
        return db.scalar(select(User.id).where(User.username == name))


def auth(name):
    r = client.post("/auth/login", data={"username": name, "password": "test-pass"})
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def make_rec(owner):
    with SessionLocal() as db:
        rec = Recommendation(user_id=uid(owner), filename="x.txt", result_json=json.dumps(RESULT))
        db.add(rec)
        db.commit()
        return rec.id


def test_decision_rules():
    rid, h = make_rec("alice"), auth("alice")
    url = f"/cv/recommendations/{rid}/decision"
    assert client.post(url, json={"decision": "reject"}, headers=h).status_code == 422
    assert client.post(url, json={"decision": "modify", "reason": "x"}, headers=h).status_code == 422
    assert client.post(url, json={"decision": "modify", "final_categories": ["PSS"]}, headers=h).status_code == 422
    assert client.post(url, json={"decision": "modify", "final_categories": ["PSS"], "reason": "better fit"}, headers=h).status_code == 200
    assert client.post(url, json={"decision": "accept"}, headers=h).status_code == 200
    assert client.post(url, json={"decision": "accept"}, headers=auth("bob")).status_code == 404  # not bob's
    assert len(client.get("/cv/decisions", headers=h).json()) == 2  # append-only history


def test_dashboard_scoping_and_acceptance():
    ra, rb = make_rec("alice"), make_rec("bob")
    client.post(f"/cv/recommendations/{ra}/decision", json={"decision": "accept"}, headers=auth("alice"))
    client.post(f"/cv/recommendations/{rb}/decision", json={"decision": "reject", "reason": "no"}, headers=auth("bob"))
    with SessionLocal() as db:
        for name, tok in (("alice", 10), ("bob", 20)):
            db.add(UsageLog(run_id=f"r-{name}", user_id=uid(name), agent="cv_agent", model="m", input_tokens=tok,
                            output_tokens=1, latency_ms=5, status="success", cost_usd=0.0,
                            ts=datetime.utcnow() - timedelta(minutes=5)))
        db.commit()
    a = client.get("/dashboard/data", headers=auth("alice")).json()
    adm = client.get("/dashboard/data", headers=auth("admin")).json()
    assert a["usage"]["totals"]["input_tokens"] == 10 and set(a["usage"]["by_user"]) == {"alice"}
    assert adm["usage"]["totals"]["input_tokens"] == 30 - 0 and set(adm["usage"]["by_user"]) == {"alice", "bob"}
    assert a["cv_acceptance"]["DCS"]["acceptance_rate"] == 1.0
    assert adm["cv_acceptance"]["DCS"]["acceptance_rate"] == 0.5


def test_import_is_admin_only_and_shows_as_separate_source():
    f = {"file": ("other.jsonl", b'{"ts":"2026-10-03T10:00:00","user":"u","agent":"a","model":"m","input_tokens":5,"output_tokens":2,"latency_ms":9,"status":"success"}\n')}
    assert client.post("/dashboard/import", data={"source": "other"}, files=f, headers=auth("alice")).status_code == 403
    r = client.post("/dashboard/import", data={"source": "other"}, files=f, headers=auth("admin"))
    assert r.status_code == 200 and r.json()["imported"] == 1
    assert "other" in client.get("/dashboard/sources", headers=auth("admin")).json()["sources"]
    assert client.get("/dashboard/data?source=other", headers=auth("admin")).json()["usage"]["totals"]["input_tokens"] == 5
    assert client.get("/dashboard/data?source=other", headers=auth("alice")).status_code == 403


def test_batch_progress_and_per_file_errors(monkeypatch):
    monkeypatch.setattr(routes_batch, "classify_cv", lambda user, name, text: {"ok": True, "result": RESULT, "error": None})
    files = [("files", ("a.txt", b"cv a", "text/plain")), ("files", ("b.md", b"cv b", "text/plain")),
             ("files", ("c.pdf", b"%PDF", "application/pdf"))]
    h = auth("alice")
    job = client.post("/cv/batch", files=files, headers=h).json()["job_id"]  # background task runs before this returns
    j = client.get(f"/cv/batch/{job}", headers=h).json()
    assert j["total"] == 3 and j["done"] == 3 and j["percent"] == 100 and j["status"] == "done_with_errors"
    assert sorted(i["status"] for i in j["items"]) == ["done", "done", "error"]
    assert client.get(f"/cv/batch/{job}", headers=auth("bob")).status_code == 404


def test_import_accepts_pasted_text_and_detects_column_names():
    pasted = ('{"timestamp":"2026-10-03T10:00:00Z","agent_name":"triage","prompt_tokens":7,"completion_tokens":3,"outcome":"success"}\n'
              '{"timestamp":"2026-10-03T11:00:00Z","agent_name":"triage","prompt_tokens":5,"completion_tokens":2,"outcome":"error"}')
    r = client.post("/dashboard/import", data={"source": "pasted", "text": pasted}, headers=auth("admin"))
    assert r.status_code == 200 and r.json()["imported"] == 2
    d = client.get("/dashboard/data?source=pasted", headers=auth("admin")).json()["usage"]
    assert d["totals"]["input_tokens"] == 12 and d["agent_health"]["triage"]["errors"] == 1
    csv_text = "time,model,input_tokens,output_tokens\n2026-10-03T10:00:00,m,4,1\n"
    assert client.post("/dashboard/import", data={"source": "csvpaste", "text": csv_text}, headers=auth("admin")).json()["imported"] == 1


def test_import_explains_mistakes_instead_of_failing_with_a_parser_error():
    log_lines = '{"ts":"2026-10-03T10:00:00","input_tokens":5}\n{"ts":"2026-10-03T11:00:00","input_tokens":6}'
    admin = auth("admin")
    # log text pasted into the mapping box
    r = client.post("/dashboard/import", data={"source": "x", "mapping": log_lines, "text": log_lines}, headers=admin)
    assert r.status_code == 400 and "column mapping" in r.json()["detail"] and "Paste log text" in r.json()["detail"]
    # a single log line is valid JSON but not a mapping
    r = client.post("/dashboard/import", data={"source": "x", "mapping": '{"run_id": "abc", "input_tokens": 5}', "text": log_lines}, headers=admin)
    assert r.status_code == 400 and "does not look like a column mapping" in r.json()["detail"]  # numbers are not column names
    r = client.post("/dashboard/import", data={"source": "x", "mapping": '{"run_id": "abc"}', "text": log_lines}, headers=admin)
    assert r.status_code == 400 and "Unknown field" in r.json()["detail"] and "run_id" in r.json()["detail"]
    # neither a file nor text
    r = client.post("/dashboard/import", data={"source": "x"}, headers=admin)
    assert r.status_code == 400 and "paste" in r.json()["detail"].lower()
    # no timestamp column
    r = client.post("/dashboard/import", data={"source": "x", "text": '{"agent":"a","input_tokens":1}'}, headers=admin)
    assert r.status_code == 400 and "timestamp" in r.json()["detail"] and "Columns found in your log: agent, input_tokens" in r.json()["detail"]
    # source name rules still apply, and nothing was imported along the way
    assert client.post("/dashboard/import", data={"source": "platform", "text": log_lines}, headers=admin).status_code == 400
    assert "x" not in client.get("/dashboard/sources", headers=admin).json()["sources"]


def test_import_warns_when_it_could_not_recognise_the_columns():
    """A log with unusual names still imports, but must not report a silent success with empty values."""
    odd = '{"records":[{"time":"2026-10-05T04:10:00Z","who":"erin","bot":"x","llm":"m","tokens":{"in":1200,"out":310},"ms":9,"result":"ok"}]}'
    admin = auth("admin")
    r = client.post("/dashboard/import", data={"source": "odd"}, headers=admin,
                    files={"file": ("log.json", odd.encode(), "application/json")}).json()
    assert r["imported"] == 1
    assert r["warning"] and "input_tokens" in r["warning"] and "user" in r["warning"]
    assert "tokens" in r["warning"] and "who" in r["warning"]  # it lists the columns it did find
    d = client.get("/dashboard/data?source=odd", headers=admin).json()["usage"]
    assert d["totals"]["input_tokens"] == 0  # the warning is telling the truth

    mapping = '{"ts":"time","user":"who","agent":"bot","model":"llm","input_tokens":"tokens.in","output_tokens":"tokens.out","latency_ms":"ms","status":"result"}'
    r2 = client.post("/dashboard/import", data={"source": "odd", "mapping": mapping}, headers=admin,
                     files={"file": ("log.json", odd.encode(), "application/json")}).json()
    assert r2["imported"] == 1 and r2["warning"] is None
    d2 = client.get("/dashboard/data?source=odd", headers=admin).json()["usage"]
    assert d2["totals"]["input_tokens"] == 1200 and list(d2["by_user"]) == ["erin"]


def test_a_well_named_log_imports_with_no_warning():
    good = '{"ts":"2026-10-03T10:00:00","user":"u","agent":"a","model":"m","input_tokens":5,"output_tokens":2,"latency_ms":9,"status":"success"}'
    r = client.post("/dashboard/import", data={"source": "clean", "text": good}, headers=auth("admin")).json()
    assert r["imported"] == 1 and r["warning"] is None
