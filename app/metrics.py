"""Stdlib-only metric maths for the dashboard (so it can be tested without the web stack)."""
import csv
import io
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone

WINDOWS = {"24h": timedelta(hours=24), "7d": timedelta(days=7), "30d": timedelta(days=30)}
CANON = ["ts", "user", "agent", "model", "input_tokens", "output_tokens", "latency_ms", "status", "cost_usd"]
# Column names other logs commonly use. When no mapping is given for a field, the first of these that exists is used.
ALIASES = {
    "ts": ["ts", "timestamp", "time", "created_at", "datetime", "date"],
    "user": ["user", "username", "user_name", "user_id"],
    "agent": ["agent", "agent_name"],
    "model": ["model", "model_name"],
    "input_tokens": ["input_tokens", "prompt_tokens", "usage.input_tokens", "usage.prompt_tokens"],
    "output_tokens": ["output_tokens", "completion_tokens", "usage.output_tokens", "usage.completion_tokens"],
    "latency_ms": ["latency_ms", "duration_ms", "latency", "elapsed_ms"],
    "status": ["status", "outcome", "result"],
    "cost_usd": ["cost_usd", "cost", "price"],
}


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def month_start(now: datetime | None = None) -> datetime:
    now = now or _now()
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def next_month_start(now: datetime | None = None) -> datetime:
    s = month_start(now)
    return s.replace(year=s.year + 1, month=1) if s.month == 12 else s.replace(month=s.month + 1)


def retry_after_seconds(oldest: datetime, now: datetime, window: int = 60) -> int:
    """Seconds until the oldest event in the window expires (at least 1)."""
    return max(1, math.ceil(window - (now - oldest).total_seconds()))


def _blank() -> dict:
    return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


def aggregate(rows: list[dict], now: datetime | None = None) -> dict:
    """rows: dicts with ts (naive UTC datetime), user, agent, input_tokens, output_tokens,
    latency_ms, status, cost_usd. A 'run' here is one logged LLM call (retries included)."""
    now = now or _now()
    totals, by_user, by_agent = _blank(), defaultdict(_blank), defaultdict(_blank)
    by_model = defaultdict(lambda: {**_blank(), "errors": 0})  # shows which provider answered and how many attempts failed
    runs = defaultdict(lambda: {k: 0 for k in WINDOWS})
    stat = defaultdict(lambda: {"calls": 0, "errors": 0, "latency": 0})

    for r in rows:
        for bucket in (totals, by_user[r["user"]], by_agent[r["agent"]]):
            bucket["calls"] += 1
            bucket["input_tokens"] += r["input_tokens"]
            bucket["output_tokens"] += r["output_tokens"]
            bucket["cost_usd"] += r["cost_usd"]
        m = by_model[r.get("model") or "unknown"]
        m["calls"] += 1
        m["input_tokens"] += r["input_tokens"]
        m["output_tokens"] += r["output_tokens"]
        m["cost_usd"] += r["cost_usd"]
        m["errors"] += 0 if r["status"] == "success" else 1
        age = now - r["ts"]
        for name, width in WINDOWS.items():
            if age <= width:
                runs[r["agent"]][name] += 1
        s = stat[r["agent"]]
        s["calls"] += 1
        s["errors"] += 0 if r["status"] == "success" else 1
        s["latency"] += r["latency_ms"]

    def fin(b):
        return {**b, "cost_usd": round(b["cost_usd"], 6)}

    return {
        "totals": fin(totals),
        "by_user": {k: fin(v) for k, v in by_user.items()},
        "by_agent": {k: fin(v) for k, v in by_agent.items()},
        "by_model": {k: fin(v) for k, v in by_model.items()},
        "runs": dict(runs),
        "agent_health": {
            a: {
                "calls": s["calls"],
                "errors": s["errors"],
                "error_rate": s["errors"] / s["calls"],
                "avg_latency_ms": round(s["latency"] / s["calls"], 1),
            }
            for a, s in stat.items()
        },
    }


def acceptance(recs: list[dict]) -> dict:
    """recs: {category, decision} where decision is accept|modify|reject|None (pending).
    Rate = accepted / decided; pending items are not counted as either."""
    cats = defaultdict(lambda: {"recommendations": 0, "accepted": 0, "modified": 0, "rejected": 0, "pending": 0})
    key = {"accept": "accepted", "modify": "modified", "reject": "rejected", None: "pending"}
    for r in recs:
        for bucket in (cats[r["category"]], cats["ALL"]):
            bucket["recommendations"] += 1
            bucket[key[r["decision"]]] += 1
    for v in cats.values():
        decided = v["accepted"] + v["modified"] + v["rejected"]
        v["acceptance_rate"] = round(v["accepted"] / decided, 4) if decided else None
    return dict(cats)


def time_saved(accepted: int, modified: int, rejected: int, sentiment_ok: int, cfg: dict) -> dict:
    """Assumption-based estimate in person-hours. Pending CV results count for nothing.
      accept -> saves (manual - review); modify -> that saving x modify_credit;
      reject -> the user still did the work by hand AND spent review time: -review.
      sentiment results -> (manual - review) each (there is no accept/reject step)."""
    gain = cfg["manual_min_per_cv"] - cfg["review_min_per_cv"]
    cv = accepted * gain + modified * gain * cfg["modify_credit"] - rejected * cfg["review_min_per_cv"]
    sent = sentiment_ok * (cfg["manual_min_per_feedback"] - cfg["review_min_per_feedback"])
    return {
        "cv_hours": round(cv / 60, 3),
        "sentiment_hours": round(sent / 60, 3),
        "total_hours": round((cv + sent) / 60, 3),
        "basis": {"accepted": accepted, "modified": modified, "rejected": rejected, "sentiment_ok": sentiment_ok},
        "assumptions_minutes": cfg,
        "note": "Estimate from stated assumptions, not measured time.",
    }


# ---------- importing another agent's log file ----------

def _get(rec: dict, key: str):
    if key in rec:
        return rec[key]
    cur = rec
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _parse_ts(v):
    if isinstance(v, (int, float)):
        v = v / 1000 if v > 1e11 else v
        return datetime.fromtimestamp(v, timezone.utc).replace(tzinfo=None)
    if isinstance(v, str) and v.strip():
        try:
            d = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
        except ValueError:
            try:
                return datetime.fromtimestamp(float(v), timezone.utc).replace(tzinfo=None)
            except ValueError:
                return None
        return d.astimezone(timezone.utc).replace(tzinfo=None) if d.tzinfo else d
    return None


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def sniff_kind(text: str) -> str:
    """Guess the format of pasted log text: a JSON array/object, JSON lines, or CSV."""
    s = text.strip()
    try:
        json.loads(s)
        return "json"
    except ValueError:
        pass
    return "jsonl" if s.startswith("{") else "csv"


def found_columns(content: str, kind: str) -> list[str]:
    """Top-level column names in the first records of a log, for error messages."""
    try:
        if kind == "csv":
            return [h for h in (next(csv.reader(io.StringIO(content)), []))][:30]
        if kind == "jsonl":
            recs = [json.loads(l) for l in content.splitlines()[:5] if l.strip()]
        else:
            data = json.loads(content)
            recs = data if isinstance(data, list) else data["records"] if isinstance(data.get("records"), list) else [data]
        return sorted({k for r in recs[:5] if isinstance(r, dict) for k in r})[:30]
    except (ValueError, AttributeError, KeyError):
        return []


def parse_external_rows(content: str, kind: str, mapping: dict | None = None, cost_fn=None):
    """Read a JSONL / JSON / CSV log into canonical rows. `mapping` maps our field name to
    their column name (dotted paths allowed, e.g. {"input_tokens": "usage.prompt_tokens"}).
    Returns (rows, skipped_count). Rows without a parseable timestamp are skipped."""
    if kind == "csv":
        records = list(csv.DictReader(io.StringIO(content)))
    elif kind == "jsonl":
        records = []
        for line in content.splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    records.append({})
    elif kind == "json":
        data = json.loads(content)
        records = data if isinstance(data, list) else data["records"] if isinstance(data.get("records"), list) else [data]
    else:
        raise ValueError(f"unsupported kind {kind!r}")

    # explicit mapping wins; otherwise use the first common column name that appears in the first records
    sample = [r for r in records[:50] if isinstance(r, dict)]
    mp = {}
    for field in CANON:
        if mapping and mapping.get(field):
            mp[field] = mapping[field]
        else:
            mp[field] = next((a for a in ALIASES[field] if any(_get(r, a) is not None for r in sample)), field)

    rows, skipped = [], 0
    for rec in records:
        ts = _parse_ts(_get(rec, mp["ts"])) if isinstance(rec, dict) else None
        if ts is None:
            skipped += 1
            continue
        inp, out = int(_num(_get(rec, mp["input_tokens"]))), int(_num(_get(rec, mp["output_tokens"])))
        model = str(_get(rec, mp["model"]) or "unknown")
        raw_status = _get(rec, mp["status"])
        ok = raw_status is None or str(raw_status).strip().lower() in ("success", "ok", "200", "true", "completed")
        cost = _get(rec, mp["cost_usd"])
        rows.append(
            {
                "ts": ts,
                "user": str(_get(rec, mp["user"]) or "unknown"),
                "agent": str(_get(rec, mp["agent"]) or "external"),
                "model": model,
                "input_tokens": inp,
                "output_tokens": out,
                "latency_ms": int(_num(_get(rec, mp["latency_ms"]))),
                "status": "success" if ok else "error",
                "cost_usd": _num(cost) if cost not in (None, "") else (cost_fn(model, inp, out) if cost_fn else 0.0),
            }
        )
    return rows, skipped
