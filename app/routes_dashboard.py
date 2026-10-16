"""Step 7: dashboard data (role-scoped) and importing another agent's log as a separate tab."""
import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from .config import estimate_cost, settings
from .db import get_db
from .limits import budget_status
from .metrics import CANON, acceptance, aggregate, parse_external_rows, found_columns, sniff_kind, time_saved
from .models import Decision, ExternalUsage, Recommendation, SentimentResult, UsageLog, User
from .security import get_current_user, require_admin, scope_to_user

dev_router = APIRouter(prefix="/dev", tags=["dev"])


@dev_router.get("/config")
def dev_config(_: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Which provider is configured and which one answered last, for the status badge in the UI.
    Logged-in users only: it queries the usage log and reveals the provider setup."""
    # Which provider answered the most recent successful call?
    last_success = db.scalar(
        select(UsageLog.model).where(UsageLog.status == "success").order_by(UsageLog.ts.desc()).limit(1)
    )
    current = None
    if last_success:
        # Extract provider from model name: "gpt-4o-mini-2024-07-18" → "openai", "claude-haiku-4-5..." → "anthropic"
        if last_success.startswith(("gpt-", "text-davinci")):
            current = "openai"
        elif last_success.startswith("claude-"):
            current = "anthropic"
        elif last_success.startswith("openai/"):
            current = "openrouter"
    return {
        "llm_provider": settings.llm_provider,
        "current_provider": current,  # which one actually answered the last call
    }


router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _assumptions() -> dict:
    return {
        "manual_min_per_cv": settings.manual_min_per_cv,
        "review_min_per_cv": settings.review_min_per_cv,
        "manual_min_per_feedback": settings.manual_min_per_feedback,
        "review_min_per_feedback": settings.review_min_per_feedback,
        "modify_credit": settings.modify_credit,
    }


@router.get("/sources")
def sources(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    ext = [] if user.role != "admin" else list(db.scalars(select(ExternalUsage.source).distinct()))
    return {"sources": ["platform", *ext]}


@router.get("/data")
def data(source: str = "platform", user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if source != "platform":
        if user.role != "admin":
            raise HTTPException(403, "Imported logs are visible to admins only")
        recs = db.scalars(select(ExternalUsage).where(ExternalUsage.source == source)).all()
        if not recs:
            raise HTTPException(404, "Unknown source")
        rows = [dict(ts=r.ts, user=r.user, agent=r.agent, model=r.model, input_tokens=r.input_tokens,
                     output_tokens=r.output_tokens, latency_ms=r.latency_ms, status=r.status,
                     cost_usd=r.cost_usd) for r in recs]
        return {"source": source, "scope": "imported log", "usage": aggregate(rows)}

    # ---- platform data: every query goes through scope_to_user ----
    stmt = scope_to_user(
        select(UsageLog, User.username).join(User, User.id == UsageLog.user_id), UsageLog.user_id, user
    )
    pairs = db.execute(stmt).all()
    rows = [dict(ts=l.ts, user=name, agent=l.agent, model=l.model, input_tokens=l.input_tokens,
                 output_tokens=l.output_tokens, latency_ms=l.latency_ms, status=l.status,
                 cost_usd=l.cost_usd) for l, name in pairs]
    unreported = sum(1 for l, _ in pairs if l.status == "success" and not l.usage_reported)

    # latest decision per recommendation (decisions are append-only)
    rec_rows = db.scalars(scope_to_user(select(Recommendation), Recommendation.user_id, user)).all()
    latest: dict[int, str] = {}
    for d in db.scalars(scope_to_user(select(Decision), Decision.user_id, user).order_by(Decision.id)):
        latest[d.recommendation_id] = d.decision
    cv_recs = [
        dict(category=json.loads(r.result_json)["primary_category"], decision=latest.get(r.id))
        for r in rec_rows
    ]
    acc = acceptance(cv_recs)
    overall = acc.get("ALL", {})
    def sentiment_count(source: str) -> int:
        return db.scalar(scope_to_user(select(func.count(SentimentResult.id)), SentimentResult.user_id, user)
                         .where(SentimentResult.source == source)) or 0

    sentiment_ok = sentiment_count("user")  # only real feedback saves anyone time
    ai_samples = sentiment_count("ai_sample")
    saved = time_saved(overall.get("accepted", 0), overall.get("modified", 0), overall.get("rejected", 0),
                       sentiment_ok, _assumptions())
    saved["basis"]["ai_sample_feedback_not_counted"] = ai_samples

    return {
        "source": "platform",
        "scope": "all users" if user.role == "admin" else user.username,
        "usage": aggregate(rows),
        "usage_not_reported": unreported,
        "budgets": [
            {"user": u.username, **budget_status(db, u)}
            for u in (db.scalars(select(User).order_by(User.id)).all() if user.role == "admin" else [user])
        ],
        "cv_acceptance": acc,
        "time_saved": saved,
        "definitions": {
            "run": "one logged LLM call (a retry counts as another call)",
            "time_saved_sentiment": "only feedback people pasted or uploaded; AI-written sample feedback is not counted",
            "acceptance_rate": "accepted / (accepted + modified + rejected); pending excluded",
            "latency": "average over all calls, errors included",
        },
    }


MAPPING_HELP = ('The column mapping is optional and must be a short JSON object that maps our field names to theirs, for example '
                '{"input_tokens": "usage.prompt_tokens", "ts": "timestamp"}. Leave it as {} to detect common column names '
                'automatically. To import log lines, choose a file or use the "Paste log text" box instead.')


def _parse_mapping(raw: str) -> dict:
    try:
        mp = json.loads(raw or "{}")
    except ValueError:
        raise HTTPException(400, "That is not a valid column mapping. " + MAPPING_HELP)
    if not isinstance(mp, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mp.items()):
        raise HTTPException(400, "That does not look like a column mapping. " + MAPPING_HELP)
    unknown = sorted(set(mp) - set(CANON))
    if unknown:
        raise HTTPException(400, f"Unknown field(s) in the column mapping: {', '.join(unknown)}. Allowed: {', '.join(CANON)}. " + MAPPING_HELP)
    return mp


@router.post("/import")
async def import_log(
    source: str = Form(...),
    mapping: str = Form("{}"),
    text: str = Form(""),
    file: UploadFile | None = File(None),
    _: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Import another agent's log (JSONL, JSON or CSV) from an uploaded file or pasted text. `mapping` is JSON: our
    field -> their column, e.g. {"input_tokens": "usage.prompt_tokens", "ts": "timestamp"}; common names are
    detected automatically when it is empty."""
    source = source.strip()
    if not source or source == "platform":
        raise HTTPException(400, "Choose a source name other than 'platform'")
    mp = _parse_mapping(mapping)
    if file is not None and file.filename:
        name = file.filename.lower()
        kind = "csv" if name.endswith(".csv") else "json" if name.endswith(".json") else "jsonl"
        content = (await file.read()).decode("utf-8-sig", errors="replace")
    elif text.strip():
        content = text.strip()
        kind = sniff_kind(content)
    else:
        raise HTTPException(400, "Choose a log file or paste the log text.")
    try:
        rows, skipped = parse_external_rows(content, kind, mp, cost_fn=estimate_cost)
    except ValueError as e:
        raise HTTPException(400, f"Could not read the log as {kind.upper()}: {e}")
    if not rows:
        cols = ", ".join(found_columns(content, kind)) or "none"
        raise HTTPException(400, f"No usable rows ({skipped} skipped): none has a timestamp the importer recognises. "
                                 f"Columns found in your log: {cols}. If one of them is the time of each call, say which one "
                                 "in the mapping box, for example {\"ts\": \"that_column_name\"}.")

    # A log whose columns have unusual names still imports, but with empty values. Say so instead of
    # reporting a silent success: without this, "imported 6 rows" can mean 6 rows of zeros.
    missing = [
        field for field, empty in (
            ("input_tokens / output_tokens", all(r["input_tokens"] == 0 and r["output_tokens"] == 0 for r in rows)),
            ("user", all(r["user"] == "unknown" for r in rows)),
            ("agent", all(r["agent"] == "external" for r in rows)),
            ("model", all(r["model"] == "unknown" for r in rows)),
            ("latency_ms", all(r["latency_ms"] == 0 for r in rows)),
        ) if empty and field not in mp
    ]
    warning = None
    if missing:
        warning = (f"Imported, but no column was recognised for: {', '.join(missing)}, so those are empty. "
                   f"Columns found in your log: {', '.join(found_columns(content, kind)) or 'none'}. "
                   "Map them if you need them, for example {\"input_tokens\": \"tokens.in\"}.")

    db.execute(delete(ExternalUsage).where(ExternalUsage.source == source))  # re-import replaces
    db.add_all(ExternalUsage(source=source, **r) for r in rows)
    db.commit()
    return {"source": source, "imported": len(rows), "skipped": skipped, "warning": warning}
