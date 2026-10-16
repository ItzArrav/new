"""Logging-accuracy check: compare the raw JSONL (written at call time, straight from the
API response) against what the database / dashboard reports.

    python -m scripts.reconcile            # every call
    python -m scripts.reconcile --sample 20
Exit code 0 = everything matches, 1 = mismatch.
"""
import argparse
import json
import random
import sys

from sqlalchemy import func, select

from app.config import settings
from app.db import SessionLocal
from app.models import UsageLog
from app.usage import tokens_from


def raw_tokens(rec: dict) -> tuple[int, int]:
    raw = rec.get("raw") or {}
    return tokens_from(raw.get("provider_usage") or raw.get("usage_metadata"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    with open(settings.usage_jsonl_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    ok_records = [r for r in records if r["status"] == "success"]
    if args.sample:
        ok_records = random.sample(ok_records, min(args.sample, len(ok_records)))

    mismatches = 0
    with SessionLocal() as db:
        print(f"{'run_id':10} {'raw in/out':>14} {'db in/out':>14}  match")
        for r in ok_records:
            row = db.scalar(select(UsageLog).where(UsageLog.run_id == r["run_id"]))
            raw = raw_tokens(r)
            got = (row.input_tokens, row.output_tokens) if row else None
            match = got == raw
            mismatches += 0 if match else 1
            print(f"{r['run_id'][:8]:10} {str(raw):>14} {str(got):>14}  {'OK' if match else 'MISMATCH'}")

        run_ids = [r["run_id"] for r in ok_records]
        db_in, db_out = db.execute(
            select(func.coalesce(func.sum(UsageLog.input_tokens), 0), func.coalesce(func.sum(UsageLog.output_tokens), 0))
            .where(UsageLog.run_id.in_(run_ids))
        ).one()

    raw_in = sum(raw_tokens(r)[0] for r in ok_records)
    raw_out = sum(raw_tokens(r)[1] for r in ok_records)
    print(f"\nTOTAL raw API usage : input={raw_in} output={raw_out}")
    print(f"TOTAL database      : input={db_in} output={db_out}")
    passed = mismatches == 0 and (raw_in, raw_out) == (db_in, db_out)
    print("RESULT:", "PASS" if passed else f"FAIL ({mismatches} row mismatches)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
