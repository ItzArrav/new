"""Step 5a: user decisions on CV recommendations (Accept / Modify / Reject)."""
import json
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from .agents.cv_schema import Category
from .db import get_db
from .models import Decision, Recommendation, User
from .security import get_current_user, scope_to_user

router = APIRouter(prefix="/cv", tags=["decisions"])


class DecisionIn(BaseModel):
    decision: Literal["accept", "modify", "reject"]
    final_categories: list[Category] | None = None  # first item = primary
    reason: str | None = None

    @model_validator(mode="after")
    def _rules(self):
        reason = (self.reason or "").strip()
        if self.decision in ("modify", "reject") and not reason:
            raise ValueError("reason is mandatory for modify and reject")
        if self.decision == "modify":
            cats = self.final_categories or []
            if not cats:
                raise ValueError("modify needs final_categories (first item is the primary)")
            if len(set(cats)) != len(cats):
                raise ValueError("final_categories contains duplicates")
        return self


@router.post("/recommendations/{rec_id}/decision")
def decide(
    rec_id: int,
    body: DecisionIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rec = db.get(Recommendation, rec_id)
    if rec is None or rec.user_id != user.id:  # only the owner decides on their results
        raise HTTPException(404, "Recommendation not found")
    original = json.loads(rec.result_json)

    if body.decision == "accept":
        final = [original["primary_category"], *original["secondary_categories"]]
    elif body.decision == "modify":
        final = body.final_categories
    else:
        final = None

    row = Decision(
        recommendation_id=rec.id,
        user_id=user.id,
        original_json=rec.result_json,
        decision=body.decision,
        final_categories_json=json.dumps(final) if final else None,
        reason=(body.reason or "").strip() or None,
    )
    db.add(row)  # append-only: every decision is kept; the latest one counts
    db.commit()
    return {"decision_id": row.id, "decision": row.decision, "final_categories": final, "ts": row.ts.isoformat()}


@router.get("/decisions")
def list_decisions(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    stmt = scope_to_user(select(Decision), Decision.user_id, user)
    return [
        dict(
            id=d.id, recommendation_id=d.recommendation_id, user_id=d.user_id,
            original=json.loads(d.original_json), decision=d.decision,
            final_categories=json.loads(d.final_categories_json) if d.final_categories_json else None,
            reason=d.reason, ts=d.ts.isoformat(),
        )
        for d in db.scalars(stmt.order_by(Decision.id.desc()).limit(500))
    ]
