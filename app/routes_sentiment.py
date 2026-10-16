"""Feedback sentiment on REAL feedback that a person pastes or uploads (stored with source "user").
AI-written sample feedback about CVs goes through routes_feedback and is stored with source "ai_sample"."""
import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .agents.sentiment_agent import analyze_feedback
from .db import SessionLocal, get_db
from .limits import budget_block_message, enforce_budget, enforce_rate_limit
from .models import SentimentResult, User
from .security import get_current_user, scope_to_user

router = APIRouter(prefix="/sentiment", tags=["sentiment"])

MAX_ITEMS = 50  # per analysis request; also the most lines one file load returns
MAX_FILE_BYTES = 1_000_000
TEXT_COLUMNS = ("feedback", "text", "comment", "comments", "review", "message", "response")


class AnalyzeIn(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_ITEMS)


@router.post("/analyze")
def analyze(body: AnalyzeIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    enforce_budget(db, user)
    enforce_rate_limit(db, user, weight=len(body.texts))

    def one(text: str) -> dict:
        text = text.strip()[:2000]
        if not text:
            return {"text": text, "ok": False, "error": "empty text"}
        blocked = budget_block_message(user)
        try:
            out = {"ok": False, "error": blocked} if blocked else analyze_feedback(user, text)
        except Exception as e:
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        if out["ok"]:
            with SessionLocal() as db:
                db.add(SentimentResult(user_id=user.id, text=text, result_json=json.dumps(out["result"]), source="user"))
                db.commit()
        return {"text": text, **out}

    with ThreadPoolExecutor(max_workers=4) as pool:
        return {"results": list(pool.map(one, body.texts))}


def parse_feedback_file(filename: str, data: bytes) -> list[str]:
    """Feedback lines from a .txt file (one per line) or a .csv file (the column named feedback, text, comment,
    review, message or response; otherwise the first column). Blank lines are dropped, whitespace is collapsed."""
    content = data.decode("utf-8-sig", errors="replace")  # utf-8-sig: Excel writes a BOM
    if filename.lower().endswith(".csv"):
        rows = [r for r in csv.reader(io.StringIO(content)) if any(c.strip() for c in r)]
        if not rows:
            return []
        header = [h.strip().lower() for h in rows[0]]
        col = next((header.index(name) for name in TEXT_COLUMNS if name in header), None)
        body = rows[1:] if col is not None else rows
        col = col or 0
        items = [r[col] for r in body if len(r) > col]
    else:
        items = content.splitlines()
    return [" ".join(i.split()) for i in items if i.strip()]


@router.post("/parse")
async def parse(file: UploadFile = File(...), user: User = Depends(get_current_user)):
    """Read a feedback file and return its lines for the user to review. No LLM call, nothing stored."""
    name = file.filename or ""
    if not name.lower().endswith((".txt", ".csv")):
        raise HTTPException(415, "Upload a .txt file (one feedback per line) or a .csv file")
    data = await file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, "File too large (max 1 MB)")
    items = parse_feedback_file(name, data)
    if not items:
        raise HTTPException(422, "No feedback lines found in the file")
    return {"texts": items[:MAX_ITEMS], "total": len(items), "truncated": len(items) > MAX_ITEMS, "max": MAX_ITEMS}


@router.get("/results")
def results(source: Literal["user", "ai_sample"] | None = None, user: User = Depends(get_current_user),
            db: Session = Depends(get_db)):
    stmt = scope_to_user(select(SentimentResult), SentimentResult.user_id, user)
    if source:
        stmt = stmt.where(SentimentResult.source == source)
    return [
        dict(id=r.id, user_id=r.user_id, text=r.text, result=json.loads(r.result_json), source=r.source, ts=r.ts.isoformat())
        for r in db.scalars(stmt.order_by(SentimentResult.id.desc()).limit(200))
    ]
