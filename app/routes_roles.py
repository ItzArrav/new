"""Job roles to compare CVs with: the 8 built-in roles plus jobs that users add.

Adding a job is two steps so that a person always checks the checklist before it is used:
  1. POST /roles/draft: the AI reads a job description (pasted text or a .txt/.md/.pdf file) and drafts a checklist.
     Nothing is saved.
  2. POST /roles: the (possibly edited) checklist is validated and saved. It can also be written by hand.
Jobs are shared by every user; the person who added one or an admin can delete it."""
import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .agents.cv_extract import SUPPORTED, ExtractError, extract_text
from .agents.job_roles import MAX_JD_CHARS, MIN_JD_CHARS, all_roles, check_role, draft_role, find_role, suggestions
from .db import get_db
from .limits import enforce_budget, enforce_rate_limit
from .models import JobRole, User
from .security import get_current_user

router = APIRouter(prefix="/roles", tags=["job roles"])

MAX_UPLOAD_BYTES = 5_000_000


def _view(spec: dict, user: User, names: dict[int, str]) -> dict:
    out = {**spec}
    if not spec["builtin"]:
        out["created_by_name"] = names.get(spec["created_by"], f"user #{spec['created_by']}")
        out["can_delete"] = user.role == "admin" or spec["created_by"] == user.id
    return out


def _names(db: Session) -> dict[int, str]:
    return {u.id: u.username for u in db.scalars(select(User))}


@router.get("")
def list_roles(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    names = _names(db)
    return [_view(r, user, names) for r in all_roles(db)]


@router.get("/find")
def find(q: str = Query(min_length=1, max_length=80), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Is this job in the list? If not, the UI asks whether to add it, and shows close names it might have meant."""
    spec = find_role(db, q)
    return {"query": q, "found": _view(spec, user, _names(db)) if spec else None,
            "suggestions": [] if spec else suggestions(db, q)}


@router.post("/draft")
def draft(title: str = Form(""), text: str = Form(""), file: UploadFile | None = File(None),
          user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Draft a checklist from a job description. Returns it for review; nothing is saved."""
    source = text
    if file is not None and file.filename:
        if not file.filename.lower().endswith(SUPPORTED):
            raise HTTPException(415, f"Use a {', '.join(SUPPORTED)} file")
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large (5 MB at most)")
        try:
            source = extract_text(file.filename, data)
        except ExtractError as e:
            raise HTTPException(422, str(e))
    if len(source.strip()) < MIN_JD_CHARS:
        raise HTTPException(422, f"Give a job description of at least {MIN_JD_CHARS} characters: paste it or upload a file")
    enforce_budget(db, user)
    enforce_rate_limit(db, user)
    out = draft_role(user, source, title)
    if not out["ok"]:
        raise HTTPException(502, f"Could not draft the checklist: {out['error']}")
    if "not_a_job" in out["result"]:
        raise HTTPException(422, f"This does not look like a job description: {out['result']['not_a_job']}")
    return {"draft": out["result"], "truncated": len(source) > MAX_JD_CHARS, "exists": find_role(db, out["result"]["name"]) is not None}


class RoleIn(BaseModel):
    name: str = Field(max_length=80)
    definition: str = Field(max_length=600)
    must_have: list[str] = Field(max_length=20)
    nice_to_have: list[str] = Field(default_factory=list, max_length=20)


@router.post("", status_code=201)
def create(body: RoleIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    clean, errors = check_role(db, body.name, body.definition, body.must_have, body.nice_to_have)
    if errors:
        raise HTTPException(422, "; ".join(errors))
    row = JobRole(name=clean["name"], definition=clean["definition"], requirements_json=json.dumps(clean["requirements"]),
                  created_by=user.id)
    db.add(row)
    db.commit()
    return _view(find_role(db, row.name), user, _names(db))


@router.delete("/{role_id}")
def delete(role_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    row = db.get(JobRole, role_id)
    if row is None:
        raise HTTPException(404, "Job not found")
    if user.role != "admin" and row.created_by != user.id:
        raise HTTPException(403, "Only the person who added this job or an admin can delete it")
    db.delete(row)
    db.commit()
    return {"deleted": row.name}
