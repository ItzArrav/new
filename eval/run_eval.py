"""CV classification evaluation: precision / recall per category + prompt-injection test.

    python -m eval.make_dataset          # once: easy set (45 normal + 5 injection CVs)
    python -m eval.make_hard_dataset     # once: hard set (40 CVs, neutral titles, mixed profiles, near-miss Other)
    python -m eval.run_eval              # easy set (uses the real model from .env)
    python -m eval.run_eval --set hard   # hard set
    python -m eval.run_eval --limit 10   # quick trial

Runs the same agent code the API uses, as user 'admin', so these calls also show up in the usage log and dashboard.
Writes eval/out/results[_hard].json and eval/out/report[_hard].md.
"""
import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.agents.cv_agent import classify_cv
from app.agents.cv_checks import CATEGORIES
from app.config import settings
from app.db import SessionLocal
from app.models import UsageLog, User
from app.seed import seed
from eval.metrics import confusion, multilabel_pr, precision_recall

OUT = Path("eval/out")
SETS = {"easy": (Path("eval/data"), ""), "hard": (Path("eval/data/hard"), "_hard")}


def pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), default="easy")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    data, suffix = SETS[args.set]

    seed()
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.username == "admin"))
    with open(data / "labels.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    started = datetime.now(timezone.utc).replace(tzinfo=None)

    def run(row):
        text = (data / "cvs" / row["filename"]).read_text(encoding="utf-8")
        try:
            out = classify_cv(user, row["filename"], text)
        except Exception as e:  # keep going; count as rejected
            out = {"ok": False, "result": None, "error": f"{type(e).__name__}: {e}", "attempts": 0, "injection_suspected": []}
        res = out["result"] if out["ok"] else None
        pred = res["primary_category"] if res else "REJECTED"
        return {**row, "predicted": pred, "predicted_secondary": res["secondary_categories"] if res else [],
                "ok": out["ok"], "attempts": out["attempts"], "error": out["error"],
                "injection_flagged": bool(res and res["injection_detected"]) or bool(out["injection_suspected"]),
                "confidence": next((s["confidence"] for s in res["scores"] if s["category"] == pred), None) if res else None}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, rows))

    # which models really answered (the provider chain may fall back), read from the usage log, not from settings
    with SessionLocal() as db:
        logged = db.scalars(select(UsageLog).where(UsageLog.user_id == user.id, UsageLog.agent == "cv_agent",
                                                   UsageLog.ts >= started)).all()
    models = Counter(r.model for r in logged if r.status == "success")
    failed_attempts = sum(1 for r in logged if r.status != "success")
    tokens = (sum(r.input_tokens for r in logged), sum(r.output_tokens for r in logged), sum(r.cost_usd for r in logged))

    normal = [r for r in results if r["kind"] == "normal"]
    y_true, y_pred = [r["label"] for r in normal], [r["predicted"] for r in normal]
    m = precision_recall(y_true, y_pred, CATEGORIES)
    has_secondary = any(r.get("secondary") for r in normal)
    true_sets = [{r["label"], *([r["secondary"]] if r.get("secondary") else [])} for r in normal]
    pred_sets = [({r["predicted"], *r["predicted_secondary"]} if r["ok"] else set()) for r in normal]
    ml = multilabel_pr(true_sets, pred_sets, CATEGORIES)
    inj = [r for r in results if r["kind"] == "injection"]
    hijacked = [r for r in inj if r["predicted"] == r["injection_target"]]
    correct_inj = [r for r in inj if r["predicted"] == r["label"]]
    right = [r["confidence"] for r in normal if r["predicted"] == r["label"] and r["confidence"] is not None]
    wrong = [r["confidence"] for r in normal if r["predicted"] != r["label"] and r["confidence"] is not None]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"results{suffix}.json").write_text(json.dumps(
        {"set": args.set, "metrics": m, "multilabel": ml, "confusion": confusion(y_true, y_pred), "models": dict(models),
         "rows": results}, indent=2))

    L = [f"# CV classification evaluation ({args.set} set)", "",
         f"- Models that answered (from the usage log): {', '.join(f'`{k}` x{v}' for k, v in models.items()) or 'none'}"
         + (f"; {failed_attempts} failed provider attempts were retried on the next provider" if failed_attempts else ""),
         f"- Configured provider chain: `{settings.llm_provider}`",
         f"- Run: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · tokens {tokens[0]:,} in / {tokens[1]:,} out · cost ${tokens[2]:.4f}",
         f"- Normal CVs: {len(normal)} (synthetic, labelled by construction)",
         f"- Primary category: accuracy {pct(m['accuracy'])} · macro precision {pct(m['macro_precision'])} · macro recall {pct(m['macro_recall'])}",
         f"- Rejected after validation (counted as wrong): {sum(1 for r in normal if not r['ok'])}",
         f"- Average confidence the model gave its primary category: {pct(sum(right) / len(right)) if right else 'n/a'} when right, "
         f"{pct(sum(wrong) / len(wrong)) if wrong else 'n/a (no wrong answers)'} when wrong", "",
         "## Primary category, per category", "",
         "| Category | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for c in CATEGORIES:
        v = m["per_label"][c]
        L.append(f"| {c} | {pct(v['precision'])} | {pct(v['recall'])} | {pct(v['f1'])} | {v['support']} |")
    if has_secondary:
        L += ["", "## All categories (primary + secondary), per category", "",
              "A category counts if it is the primary or a secondary one, both in the label and in the answer.", "",
              f"Micro precision {pct(ml['micro_precision'])} · micro recall {pct(ml['micro_recall'])}", "",
              "| Category | Precision | Recall | Support |", "|---|---|---|---|"]
        for c in CATEGORIES:
            v = ml["per_label"][c]
            L.append(f"| {c} | {pct(v['precision'])} | {pct(v['recall'])} | {v['support']} |")
        mixed = [r for r in normal if r.get("secondary")]
        found = [r for r in mixed if r["secondary"] in r["predicted_secondary"]]
        L += ["", f"Mixed CVs: the expected secondary category was returned for {len(found)} of {len(mixed)}."]
    L += ["", "## Misclassified (primary)", ""]
    wrong_rows = [r for r in normal if r["predicted"] != r["label"]]
    L += [f"- `{r['filename']}`: expected {r['label']}, got {r['predicted']}" + (f" ({r['error']})" if not r["ok"] else "")
          for r in wrong_rows] or ["- none"]
    if inj:
        L += ["", "## Prompt-injection CVs", "",
              f"- Total: {len(inj)} · hijacked (predicted the attacker's target): **{len(hijacked)}** · classified correctly: {len(correct_inj)}",
              "", "| File | True label | Attacker target | Predicted | Flagged |", "|---|---|---|---|---|"]
        L += [f"| {r['filename']} | {r['label']} | {r['injection_target']} | {r['predicted']} | {r['injection_flagged']} |" for r in inj]
    (OUT / f"report{suffix}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
