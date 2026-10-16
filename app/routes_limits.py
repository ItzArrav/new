from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db
from .limits import budget_status, effective_budget
from .models import User, UserLimit
from .security import get_current_user, require_admin

router = APIRouter(tags=["limits"])


@router.get("/me/budget")
def my_budget(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return {**budget_status(db, user), "rate_limit_per_minute": settings.rate_limit_per_minute}


@router.get("/admin/budgets")
def all_budgets(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    return [{"user_id": u.id, "username": u.username, **budget_status(db, u)} for u in db.scalars(select(User))]


class BudgetIn(BaseModel):
    monthly_token_budget: int | None = Field(
        default=None, ge=0, description="Tokens per month. 0 = unlimited, null = back to the default."
    )


@router.put("/admin/users/{user_id}/budget")
def set_budget(user_id: int, body: BudgetIn, _: User = Depends(require_admin), db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    row = db.get(UserLimit, user_id) or UserLimit(user_id=user_id)
    row.monthly_token_budget = body.monthly_token_budget
    db.add(row)
    db.commit()
    return {"user_id": user_id, "effective_limit_tokens": effective_budget(db, user)}
