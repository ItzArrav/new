import json

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from .agents.cv_agent import classify_cv
from .agents.cv_checker import classify_with_quality
from .agents.cv_extract import SUPPORTED, ExtractError, extract_text
from .agents.cv_schema import CVClassification
from .db import get_db
from .email_extract import extract_candidate_email, extract_candidate_name
from .limits import enforce_budget, enforce_rate_limit
from .models import Recommendation, SentEmail, User
from .security import get_current_user, scope_to_user

router = APIRouter(prefix="/cv", tags=["cv"])

MAX_UPLOAD_BYTES = 5_000_000
MAX_STORED_TEXT = 30_000  # characters of CV text kept for later use
ALLOWED_SUFFIXES = SUPPORTED


@router.get("/schema")
def schema():
    """The documented JSON Schema of the agent's output."""
    return CVClassification.model_json_schema()


@router.post("/classify")
def classify(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    name = file.filename or "cv.txt"
    if not name.lower().endswith(ALLOWED_SUFFIXES):
        raise HTTPException(415, f"Only {', '.join(ALLOWED_SUFFIXES)} files for now")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large")
    try:
        text = extract_text(name, data)
    except ExtractError as e:
        raise HTTPException(422, str(e))

    enforce_budget(db, user)
    enforce_rate_limit(db, user)
    # classification and the quality check run together; every LLM call is usage-logged
    outcome, quality, role_fit = classify_with_quality(lambda: classify_cv(user, name, text), user, text)
    rec_id = None
    if outcome["ok"]:
        rec = Recommendation(user_id=user.id, filename=name, result_json=json.dumps(outcome["result"]),
                             cv_text=text[:MAX_STORED_TEXT], quality_json=json.dumps(quality) if quality else None,
                             role_fit_json=json.dumps(role_fit) if role_fit else None)
        db.add(rec)
        db.commit()
        rec_id = rec.id
    return {"recommendation_id": rec_id, "quality": quality if outcome["ok"] else None,
            "role_fit": role_fit if outcome["ok"] else None, **outcome}


@router.get("/recommendations")
def my_recommendations(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    stmt = scope_to_user(select(Recommendation), Recommendation.user_id, user)
    rows = db.scalars(stmt.order_by(Recommendation.id.desc()).limit(100)).all()
    rec_ids = [r.id for r in rows]
    sent_map = {}
    if rec_ids:
        sent_rows = db.scalars(select(SentEmail).where(SentEmail.recommendation_id.in_(rec_ids))).all()
        for s in sent_rows:
            sent_map.setdefault(s.recommendation_id, []).append({
                "id": s.id,
                "recipient_email": s.recipient_email,
                "status": s.status,
                "template_type": s.template_type,
                "subject": s.subject,
                "ts": s.ts.isoformat(),
            })

    return [
        dict(id=r.id, user_id=r.user_id, filename=r.filename, ts=r.ts.isoformat(), result=json.loads(r.result_json),
             quality=json.loads(r.quality_json) if r.quality_json else None,
             feedback=json.loads(r.feedback_json) if r.feedback_json else None,
             role_fit=json.loads(r.role_fit_json) if r.role_fit_json else None,
             candidate_email=extract_candidate_email(r.cv_text),
             candidate_name=extract_candidate_name(r.cv_text, r.filename),
             sent_emails=sent_map.get(r.id, []))
        for r in rows
    ]

