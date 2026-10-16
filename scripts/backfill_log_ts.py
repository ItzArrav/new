"""Add the missing "ts" to old raw usage log lines (written before the log carried a timestamp).

    python -m scripts.backfill_log_ts

The timestamp is taken from the database row with the same run_id, so it is the real call time. Lines that already
have a "ts", and lines whose run_id is not in the database, are left exactly as they are. The original file is kept
as <file>.bak."""
import json
import shutil
import sys
from pathlib import Path

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.models import UsageLog


def main() -> int:
    path = Path(settings.usage_jsonl_path)
    if not path.exists():
        print(f"No log file at {path}")
        return 1
    with SessionLocal() as db:
        ts_by_run = {run_id: ts for run_id, ts in db.execute(select(UsageLog.run_id, UsageLog.ts))}
    out, added, unmatched = [], 0, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "ts" not in rec:
            ts = ts_by_run.get(rec.get("run_id"))
            if ts is None:
                unmatched += 1
            else:
                rec = {"ts": ts.isoformat() + "Z", **rec}
                added += 1
            line = json.dumps(rec, default=str)
        out.append(line)
    if added:
        shutil.copyfile(path, str(path) + ".bak")
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{len(out)} lines: added ts to {added}, {unmatched} had no matching database row (left unchanged). "
          + (f"Backup: {path}.bak" if added else "Nothing changed."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
