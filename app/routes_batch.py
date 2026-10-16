"""Step 5b: batch CV upload with progress. Work runs in a background thread pool; progress
lives in the database, so any uvicorn worker can answer the progress request."""
import json
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .agents.cv_agent import classify_cv
from .agents.cv_checker import classify_with_quality
from .agents.cv_extract import SUPPORTED, ExtractError, extract_text
from .db import SessionLocal, get_db
from .limits import budget_block_message, enforce_budget, enforce_rate_limit
from .models import Job, JobItem, Recommendation, User
from .routes_cv import MAX_STORED_TEXT
from .security import get_current_user

router = APIRouter(prefix="/cv", tags=["batch"])

MAX_FILES = 50
MAX_BYTES = 5_000_000
ALLOWED = SUPPORTED
WORKERS = 4


@router.post("/batch")
def start_batch(
    background: BackgroundTasks,
    files: list[UploadFile] = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not files or len(files) > MAX_FILES:
        raise HTTPException(400, f"Upload 1-{MAX_FILES} files")
    enforce_budget(db, user)
    enforce_rate_limit(db, user, weight=len(files))

    job = Job(user_id=user.id, status="running", total=len(files), done=0)
    db.add(job)
    db.flush()

    payloads, rejected = [], 0
    for f in files:
        name = f.filename or "cv.txt"
        data = f.file.read(MAX_BYTES + 1)  # must be read now; the request body is gone later
        problem, text = None, ""
        if not name.lower().endswith(ALLOWED):
            problem = f"Only {', '.join(ALLOWED)} files"
        elif len(data) > MAX_BYTES:
            problem = "File too large"
        else:
            try:
                text = extract_text(name, data)  # PDFs are parsed here, while the bytes are in hand
            except ExtractError as e:
                problem = str(e)
        item = JobItem(job_id=job.id, filename=name, status="error" if problem else "pending", error=problem)
        db.add(item)
        db.flush()
        if problem:
            rejected += 1
        else:
            payloads.append((item.id, name, text))
    job.done = rejected
    if not payloads:
        job.status = "done_with_errors"
    db.commit()

    if payloads:
        background.add_task(_run_batch, job.id, user.id, payloads)
    return {"job_id": job.id, "total": job.total, "rejected_at_upload": rejected}


def _run_batch(job_id: int, user_id: int, payloads: list) -> None:
    with SessionLocal() as db:
        user = db.get(User, user_id)

    def work(p):
        item_id, name, text = p
        blocked = budget_block_message(user)  # checked per file, so a batch stops spending once the budget is gone
        quality = role_fit = None
        try:
            if blocked:
                out = {"ok": False, "error": blocked}
            else:  # classification and the quality check run together
                out, quality, role_fit = classify_with_quality(lambda: classify_cv(user, name, text), user, text)
        except Exception as e:  # one bad CV must not stop the batch
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        with SessionLocal() as db:
            item = db.get(JobItem, item_id)
            if out["ok"]:
                rec = Recommendation(user_id=user_id, filename=name, result_json=json.dumps(out["result"]),
                                     cv_text=text[:MAX_STORED_TEXT],
                                     quality_json=json.dumps(quality) if quality else None,
                                     role_fit_json=json.dumps(role_fit) if role_fit else None)
                db.add(rec)
                db.flush()
                item.status, item.recommendation_id = "done", rec.id
            else:
                item.status, item.error = "error", str(out["error"])[:500]
            db.execute(update(Job).where(Job.id == job_id).values(done=Job.done + 1))  # atomic +1
            db.commit()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(work, payloads))

    with SessionLocal() as db:
        any_error = db.scalar(select(JobItem.id).where(JobItem.job_id == job_id, JobItem.status == "error"))
        job = db.get(Job, job_id)
        job.status = "done_with_errors" if any_error else "done"
        db.commit()


@router.get("/batch/{job_id}")
def batch_progress(job_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None or (user.role != "admin" and job.user_id != user.id):
        raise HTTPException(404, "Job not found")
    items = db.scalars(select(JobItem).where(JobItem.job_id == job_id).order_by(JobItem.id)).all()
    return {
        "job_id": job.id, "status": job.status, "total": job.total, "done": job.done,
        "percent": round(100 * job.done / job.total) if job.total else 100,
        "items": [
            dict(filename=i.filename, status=i.status, recommendation_id=i.recommendation_id, error=i.error)
            for i in items
        ],
    }
