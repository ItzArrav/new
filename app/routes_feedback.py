"""One AI-written feedback comment per stored CV, with its sentiment. Regenerating replaces the previous comment."""
import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .agents.feedback_generator import generate_feedback
from .agents.sentiment_agent import analyze_feedback
from .db import get_db
from .limits import budget_block_message, enforce_budget, enforce_rate_limit
from .models import Recommendation, SentimentResult, User
from .security import get_current_user

router = APIRouter(prefix="/feedback", tags=["feedback"])


class GenerateIn(BaseModel):
    recommendation_id: int


@router.post("/generate")
def generate(body: GenerateIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rec = db.get(Recommendation, body.recommendation_id)
    if rec is None or (user.role != "admin" and rec.user_id != user.id):
        raise HTTPException(404, "Recommendation not found")
    if not (rec.cv_text or "").strip():
        raise HTTPException(409, "This CV was classified before its text was stored. Upload it again to get feedback.")

    enforce_budget(db, user)
    enforce_rate_limit(db, user, weight=2)  # one call to write the feedback, one to label its sentiment
    try:
        out = generate_feedback(user, rec.cv_text)
    except Exception as e:
        raise HTTPException(502, f"Feedback generator failed: {type(e).__name__}: {e}")
    if not out["ok"]:
        raise HTTPException(502, f"Feedback rejected: {out['error']}")

    feedback = out["result"]["feedback"]
    sentiment, sentiment_error = None, None
    blocked = budget_block_message(user)
    try:
        s = {"ok": False, "error": blocked} if blocked else analyze_feedback(user, feedback)
        sentiment, sentiment_error = (s["result"], None) if s["ok"] else (None, s.get("error"))
    except Exception as e:  # the feedback itself is still worth keeping
        sentiment_error = f"{type(e).__name__}: {e}"
    if sentiment:
        db.add(SentimentResult(user_id=user.id, text=feedback, result_json=json.dumps(sentiment), source="ai_sample"))

    saved = {"feedback": feedback, "evidence": out["result"]["evidence"], "sentiment": sentiment,
             "sentiment_error": sentiment_error, "ts": datetime.now(timezone.utc).isoformat()}
    rec.feedback_json = json.dumps(saved)
    db.commit()
    return {"recommendation_id": rec.id, "filename": rec.filename, "feedback": saved}
