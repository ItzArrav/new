"""API routes for candidate email communication (individual and batch)."""
import json
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db
from .email_extract import extract_candidate_email, extract_candidate_name
from .email_service import DEFAULT_TEMPLATES, render_template, send_email_message
from .models import Decision, Recommendation, SentEmail, User
from .security import get_current_user, scope_to_user

router = APIRouter(prefix="/email", tags=["email"])


class SingleEmailRequest(BaseModel):
    recommendation_id: int | None = None
    recipient_email: str
    candidate_name: str | None = None
    role_name: str | None = None
    subject: str
    body: str
    template_type: str = "custom"


class BatchEmailRequest(BaseModel):
    recommendation_ids: list[int]
    template_type: Literal["interview", "shortlist", "reject", "custom"] = "interview"
    subject_template: str | None = None
    body_template: str | None = None
    company_name: str | None = None
    recruiter_name: str | None = None


@router.get("/templates")
def get_templates():
    """Returns available email templates with their default subjects and bodies."""
    return {
        "templates": DEFAULT_TEMPLATES,
        "default_company": settings.company_name,
        "default_recruiter": settings.recruiter_name,
    }


@router.get("/config-status")
def get_email_status():
    """Returns whether live SMTP is configured or running in simulated test mode."""
    configured = bool(settings.smtp_host and settings.smtp_user and settings.smtp_password)
    return {
        "is_configured": configured,
        "mode": "live" if configured else "simulated",
        "smtp_host": settings.smtp_host or "(not set)",
        "smtp_user": settings.smtp_user or "(not set)",
        "from_address": settings.smtp_from or settings.smtp_user or "simulated@localhost",
    }


@router.post("/send")
def send_individual_email(
    req: SingleEmailRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Sends an email to a single candidate (or records a simulated email)."""
    rec = None
    if req.recommendation_id:
        rec = db.get(Recommendation, req.recommendation_id)
        if rec and rec.user_id != user.id and user.role != "admin":
            raise HTTPException(403, "Not authorized to access this candidate recommendation")

    # Render placeholders if present
    replacements = {
        "name": req.candidate_name or (extract_candidate_name(rec.cv_text, rec.filename) if rec else "Candidate"),
        "role": req.role_name or (json.loads(rec.result_json).get("primary_category") if rec else "the position"),
        "company": settings.company_name,
        "recruiter": settings.recruiter_name,
    }
    final_subject = render_template(req.subject, replacements)
    final_body = render_template(req.body, replacements)

    try:
        res = send_email_message(
            to_email=req.recipient_email,
            subject=final_subject,
            body=final_body,
            db=db,
            user_id=user.id,
            candidate_name=replacements["name"],
            recommendation_id=req.recommendation_id,
            template_type=req.template_type,
        )
        return {"ok": True, **res}
    except Exception as e:
        raise HTTPException(400, f"Could not send email: {e}")


@router.post("/batch")
def send_batch_emails(
    req: BatchEmailRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Sends reply emails to multiple candidates in batch."""
    if not req.recommendation_ids:
        raise HTTPException(400, "No recommendations selected for batch email")

    # Fetch recommendations
    stmt = select(Recommendation).where(Recommendation.id.in_(req.recommendation_ids))
    if user.role != "admin":
        stmt = stmt.where(Recommendation.user_id == user.id)
    recs = db.scalars(stmt).all()
    if not recs:
        raise HTTPException(404, "No matching candidate recommendations found")

    # Select templates
    tpl = DEFAULT_TEMPLATES.get(req.template_type, DEFAULT_TEMPLATES["interview"])
    raw_subject = req.subject_template or tpl["subject"]
    raw_body = req.body_template or tpl["body"]
    company = req.company_name or settings.company_name
    recruiter = req.recruiter_name or settings.recruiter_name

    results = []
    sent_count = 0
    simulated_count = 0
    skipped_count = 0
    failed_count = 0

    for r in recs:
        # Detect email from CV text or stored data
        cand_email = extract_candidate_email(r.cv_text)
        cand_name = extract_candidate_name(r.cv_text, r.filename)
        res_data = json.loads(r.result_json) if r.result_json else {}
        primary_cat = res_data.get("primary_category", "Specialist")

        if not cand_email:
            skipped_count += 1
            results.append({
                "recommendation_id": r.id,
                "filename": r.filename,
                "candidate_name": cand_name,
                "recipient_email": None,
                "status": "skipped_no_email",
                "error": "No email address found in CV text",
            })
            continue

        replacements = {
            "name": cand_name,
            "role": primary_cat,
            "company": company,
            "recruiter": recruiter,
        }
        subj = render_template(raw_subject, replacements)
        body = render_template(raw_body, replacements)

        try:
            res = send_email_message(
                to_email=cand_email,
                subject=subj,
                body=body,
                db=db,
                user_id=user.id,
                candidate_name=cand_name,
                recommendation_id=r.id,
                template_type=req.template_type,
            )
            if res["status"] == "sent":
                sent_count += 1
            elif res["status"] == "simulated":
                simulated_count += 1
            else:
                failed_count += 1

            results.append({
                "recommendation_id": r.id,
                "filename": r.filename,
                "candidate_name": cand_name,
                "recipient_email": cand_email,
                "status": res["status"],
                "error": res["error"],
            })
        except Exception as err:
            failed_count += 1
            results.append({
                "recommendation_id": r.id,
                "filename": r.filename,
                "candidate_name": cand_name,
                "recipient_email": cand_email,
                "status": "failed",
                "error": str(err),
            })

    return {
        "ok": True,
        "total": len(recs),
        "sent": sent_count,
        "simulated": simulated_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "results": results,
    }


@router.get("/history")
def get_email_history(
    recommendation_id: int | None = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieves log of sent emails."""
    stmt = scope_to_user(select(SentEmail), SentEmail.user_id, user)
    if recommendation_id:
        stmt = stmt.where(SentEmail.recommendation_id == recommendation_id)
    rows = db.scalars(stmt.order_by(SentEmail.id.desc()).limit(100)).all()

    return [
        {
            "id": r.id,
            "user_id": r.user_id,
            "recommendation_id": r.recommendation_id,
            "recipient_email": r.recipient_email,
            "candidate_name": r.candidate_name,
            "subject": r.subject,
            "body": r.body,
            "template_type": r.template_type,
            "status": r.status,
            "error": r.error,
            "ts": r.ts.isoformat(),
        }
        for r in rows
    ]
