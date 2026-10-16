from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db import SessionLocal
from app.main import app
from app.models import AdminRequest, User

c = TestClient(app)


def hdr(name, password="test-pass"):
    r = c.post("/auth/login", data={"username": name, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def role_of(name):
    with SessionLocal() as db:
        return db.scalar(select(User.role).where(User.username == name))


def test_admin_creates_a_user_who_can_log_in_as_a_plain_user():
    r = c.post("/admin/users", json={"username": "Dana", "password": "a-good-pass"}, headers=hdr("admin"))
    assert r.status_code == 201 and r.json()["username"] == "dana" and r.json()["role"] == "user"
    h = hdr("dana", "a-good-pass")
    assert c.get("/me", headers=h).json()["role"] == "user"
    assert c.get("/admin/users", headers=h).status_code == 403


def test_only_admins_can_create_users_and_input_is_validated():
    body = {"username": "eve", "password": "a-good-pass"}
    assert c.post("/admin/users", json=body).status_code == 401
    assert c.post("/admin/users", json=body, headers=hdr("alice")).status_code == 403
    a = hdr("admin")
    assert c.post("/admin/users", json={"username": "alice", "password": "a-good-pass"}, headers=a).status_code == 409
    assert c.post("/admin/users", json={"username": "x", "password": "a-good-pass"}, headers=a).status_code == 422
    assert c.post("/admin/users", json={"username": "bad name!", "password": "a-good-pass"}, headers=a).status_code == 422
    assert c.post("/admin/users", json={"username": "frank", "password": "short"}, headers=a).status_code == 422
    assert c.post("/admin/users", json={"username": "frank", "password": "a-good-pass", "role": "root"}, headers=a).status_code == 422
    assert role_of("frank") is None


def test_an_admin_can_create_another_admin_but_a_user_cannot():
    a = hdr("admin")
    assert c.post("/admin/users", json={"username": "boss2", "password": "a-good-pass", "role": "admin"}, headers=a).status_code == 201
    assert c.get("/admin/users", headers=hdr("boss2", "a-good-pass")).status_code == 200
    # alice cannot make herself or anyone else an admin
    assert c.post("/admin/users", json={"username": "sneaky", "password": "a-good-pass", "role": "admin"}, headers=hdr("alice")).status_code == 403
    assert role_of("sneaky") is None


def test_request_approve_flow_promotes_the_user_immediately():
    h = hdr("bob")
    assert c.post("/me/admin-request", json={"reason": "short"}, headers=h).status_code == 422  # reason too short
    r = c.post("/me/admin-request", json={"reason": "I run the HR onboarding team"}, headers=h)
    assert r.status_code == 201
    assert c.post("/me/admin-request", json={"reason": "asking again, please"}, headers=h).status_code == 409  # one pending at a time
    assert c.get("/me/admin-request", headers=h).json()["request"]["status"] == "pending"
    assert role_of("bob") == "user" and c.get("/admin/users", headers=h).status_code == 403  # still no access

    rid = r.json()["id"]
    assert c.post(f"/admin/admin-requests/{rid}/approve", headers=h).status_code == 403  # bob cannot approve himself
    assert c.post(f"/admin/admin-requests/{rid}/approve", json={"note": "welcome"}, headers=hdr("admin")).status_code == 200
    assert role_of("bob") == "admin"
    assert c.get("/admin/users", headers=h).status_code == 200  # same token, new rights
    got = c.get("/me/admin-request", headers=h).json()
    assert got["is_admin"] is True and got["request"]["status"] == "approved" and got["request"]["decided_by"] == "admin"
    assert c.post(f"/admin/admin-requests/{rid}/approve", headers=hdr("admin")).status_code == 409  # cannot decide twice
    assert c.post("/me/admin-request", json={"reason": "I am already admin now"}, headers=h).status_code == 400


def test_denied_request_leaves_the_role_alone_and_can_be_retried():
    h = hdr("carol")
    rid = c.post("/me/admin-request", json={"reason": "I would like admin rights"}, headers=h).json()["id"]
    assert c.post(f"/admin/admin-requests/{rid}/deny", json={"note": "not needed"}, headers=hdr("admin")).status_code == 200
    assert role_of("carol") == "user"
    assert c.get("/me/admin-request", headers=h).json()["request"]["decision_note"] == "not needed"
    assert c.post("/me/admin-request", json={"reason": "second try with more detail"}, headers=h).status_code == 201


def test_only_admins_see_the_request_list_and_admins_cannot_request():
    c.post("/me/admin-request", json={"reason": "please make me admin"}, headers=hdr("alice"))
    assert c.get("/admin/admin-requests", headers=hdr("bob")).status_code == 403
    lst = c.get("/admin/admin-requests", headers=hdr("admin")).json()
    assert [x["username"] for x in lst] == ["alice"] and lst[0]["status"] == "pending"
    assert c.post("/me/admin-request", json={"reason": "I am the admin already"}, headers=hdr("admin")).status_code == 400
    assert c.post("/admin/admin-requests/999/approve", headers=hdr("admin")).status_code == 404
    with SessionLocal() as db:
        assert db.scalars(select(AdminRequest)).one().status == "pending"


def test_signup_creates_a_plain_user_and_logs_them_in():
    r = c.post("/auth/signup", json={"username": "Newbie", "password": "a-good-pass"})
    assert r.status_code == 201 and r.json()["role"] == "user" and r.json()["username"] == "newbie"
    h = {"Authorization": f"Bearer {r.json()['access_token']}"}
    assert c.get("/me", headers=h).json() == {"id": c.get("/me", headers=h).json()["id"], "username": "newbie", "role": "user"}
    assert c.get("/admin/users", headers=h).status_code == 403
    assert hdr("newbie", "a-good-pass")  # the password works for a normal login too


def test_signup_can_never_create_an_admin():
    r = c.post("/auth/signup", json={"username": "climber", "password": "a-good-pass", "role": "admin", "is_admin": True})
    assert r.status_code == 201 and r.json()["role"] == "user"
    assert role_of("climber") == "user"


def test_signup_validates_and_rejects_duplicates():
    assert c.post("/auth/signup", json={"username": "alice", "password": "a-good-pass"}).status_code == 409
    assert c.post("/auth/signup", json={"username": "ALICE", "password": "a-good-pass"}).status_code == 409  # case-insensitive
    assert c.post("/auth/signup", json={"username": "ok-name", "password": "short"}).status_code == 422
    assert c.post("/auth/signup", json={"username": "no spaces!", "password": "a-good-pass"}).status_code == 422


def test_signup_can_be_turned_off_and_is_capped(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "allow_signup", False)
    assert c.post("/auth/signup", json={"username": "locked", "password": "a-good-pass"}).status_code == 403
    monkeypatch.setattr(settings, "allow_signup", True)
    monkeypatch.setattr(settings, "signup_max_per_hour", 4)  # the 4 seeded accounts already count
    assert c.post("/auth/signup", json={"username": "toomany", "password": "a-good-pass"}).status_code == 429
    assert role_of("toomany") is None


def test_a_self_registered_user_still_needs_admin_approval_to_become_admin():
    tok = c.post("/auth/signup", json={"username": "hopeful", "password": "a-good-pass"}).json()["access_token"]
    h = {"Authorization": f"Bearer {tok}"}
    rid = c.post("/me/admin-request", json={"reason": "I will manage the HR team"}, headers=h).json()["id"]
    assert c.post(f"/admin/admin-requests/{rid}/approve", headers=h).status_code == 403
    assert role_of("hopeful") == "user"
    assert c.post(f"/admin/admin-requests/{rid}/approve", headers=hdr("admin")).status_code == 200
    assert role_of("hopeful") == "admin"


def test_login_ignores_username_case_and_spaces_but_not_password_case():
    c.post("/auth/signup", json={"username": "Mixed.Case", "password": "Right-Pass-1"})
    for typed in ("Mixed.Case", "mixed.case", "MIXED.CASE", "  mixed.case "):
        r = c.post("/auth/login", data={"username": typed, "password": "Right-Pass-1"})
        assert r.status_code == 200, typed
    assert c.post("/auth/login", data={"username": "Mixed.Case", "password": "right-pass-1"}).status_code == 401  # password stays case-sensitive
    assert c.post("/auth/login", data={"username": "Mixed.Case", "password": "wrong-password"}).status_code == 401
