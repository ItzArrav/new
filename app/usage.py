"""Usage logging: ONE callback handler that records every LLM call.

Token counts are read from the provider's response (never estimated):
  - usage_metadata       -> LangChain's normalised usage on the AIMessage
  - response_metadata    -> the provider's raw usage block (e.g. Anthropic "usage")
Both are stored, so the dashboard can be reconciled against the raw figures.
"""
import json
import logging
import threading
import time
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from .config import estimate_cost, settings
from .db import SessionLocal
from .models import UsageLog, User, utcnow

log = logging.getLogger(__name__)
_jsonl_lock = threading.Lock()


def _append_jsonl(record: dict) -> None:
    line = json.dumps(record, default=str) + "\n"
    with _jsonl_lock:
        # one write() per line: appends of short lines are atomic on POSIX,
        # so two uvicorn workers can share the file
        with open(settings.usage_jsonl_path, "a", encoding="utf-8") as f:
            f.write(line)


def tokens_from(usage: dict | None) -> tuple[int, int]:
    """Read (input, output) tokens from either LangChain or provider-style usage dicts."""
    usage = usage or {}
    inp = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    out = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    return int(inp), int(out)


def _reported_cost(usage: dict | None) -> float | None:
    """The provider's own cost figure in USD, if its usage block has a sane numeric 'cost' (OpenRouter does)."""
    c = (usage or {}).get("cost")
    return float(c) if isinstance(c, (int, float)) and not isinstance(c, bool) and c >= 0 else None


class UsageLogger(BaseCallbackHandler):
    """Create one per request: UsageLogger(user.id, "cv_agent").

    Because it is built per request from the authenticated user, two users running
    at the same time can never mix up their logs.
    """

    def __init__(self, user_id: int, agent: str):
        self.user_id = user_id
        self.agent = agent
        self._started: dict[UUID, tuple[float, dict]] = {}

    # --- start -----------------------------------------------------------
    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._started[run_id] = (time.perf_counter(), kwargs.get("invocation_params") or {})

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._started[run_id] = (time.perf_counter(), kwargs.get("invocation_params") or {})

    # --- success ---------------------------------------------------------
    def on_llm_end(self, response, *, run_id, **kwargs):
        t0, params = self._started.pop(run_id, (time.perf_counter(), {}))
        latency_ms = int((time.perf_counter() - t0) * 1000)

        gen = response.generations[0][0]
        msg = getattr(gen, "message", None)
        usage_metadata = dict(getattr(msg, "usage_metadata", None) or {})
        meta = dict(getattr(msg, "response_metadata", None) or {})
        provider_usage = meta.get("usage") or meta.get("token_usage") or (
            (response.llm_output or {}).get("token_usage") if response.llm_output else None
        )
        model = (
            meta.get("model_name")
            or meta.get("model")
            or params.get("model")
            or params.get("model_name")
            or settings.model_name
        )

        # prefer the provider's raw block; fall back to LangChain's normalised one
        source = provider_usage or usage_metadata
        in_tok, out_tok = tokens_from(source)
        self._record(
            run_id, model, in_tok, out_tok, latency_ms, "success", None,
            usage_reported=bool(source),
            raw={"usage_metadata": usage_metadata, "provider_usage": provider_usage},
            reported_cost=_reported_cost(provider_usage),
        )

    # --- error -----------------------------------------------------------
    def on_llm_error(self, error, *, run_id, **kwargs):
        t0, params = self._started.pop(run_id, (time.perf_counter(), {}))
        latency_ms = int((time.perf_counter() - t0) * 1000)
        model = params.get("model") or params.get("model_name") or settings.model_name
        self._record(
            run_id, model, 0, 0, latency_ms, "error", f"{type(error).__name__}: {error}",
            usage_reported=False, raw=None,
        )

    # --- storage ---------------------------------------------------------
    def _record(self, run_id, model, in_tok, out_tok, latency_ms, status, error, usage_reported, raw, reported_cost=None):
        # a cost the provider itself reported (OpenRouter does) beats our price-table estimate
        cost = reported_cost if reported_cost is not None else estimate_cost(model, in_tok, out_tok)
        ts = utcnow()  # one timestamp for the database row and the raw log line
        try:
            with SessionLocal() as db:
                db.add(
                    UsageLog(
                        ts=ts, run_id=str(run_id), user_id=self.user_id, agent=self.agent, model=model,
                        input_tokens=in_tok, output_tokens=out_tok, latency_ms=latency_ms,
                        status=status, error=error, cost_usd=cost,
                        usage_reported=usage_reported, raw_usage=json.dumps(raw) if raw else None,
                    )
                )
                db.commit()
            _append_jsonl(
                {
                    "ts": ts.isoformat() + "Z", "run_id": str(run_id), "user_id": self.user_id, "agent": self.agent,
                    "model": model, "status": status, "latency_ms": latency_ms,
                    "input_tokens": in_tok, "output_tokens": out_tok, "cost_usd": cost,
                    "raw": raw,
                }
            )
        except Exception:
            log.exception("FAILED to record usage for run %s", run_id)


def usage_config(user: User, agent: str) -> dict:
    """Pass this as `config=` to every graph/model call so nothing goes unlogged."""
    return {"callbacks": [UsageLogger(user.id, agent)]}
