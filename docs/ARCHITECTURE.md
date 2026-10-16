# Architecture

A multi-user web platform where authenticated users run AI agents (CV classification, feedback sentiment) and a dashboard
reports token use, cost, usage, errors, acceptance and time saved. Design rule: **the agents are the feature, the
logging is the product.**

## 1. System overview

```
Browser (single HTML page, Chart.js)
   |  Authorization: Bearer <JWT> on every request
   v
FastAPI (app/main.py)
   |-- security.py   token -> user loaded from the DB; require_admin; scope_to_user
   |-- limits.py     monthly token budget (from usage_logs) + per-minute rate limit (rate_events table)
   |-- routes_*.py   cv, batch, decisions, sentiment, feedback, rolefit, dashboard, limits, admin
   |        |
   |        v
   |   agents/       cv_agent, cv_checker (HR review), role_fit, sentiment_agent, feedback_generator
   |        |        every call carries config=usage_config(user, "<agent>")
   |        v
   |   llm.py        provider chain with fallback (LLM_PROVIDER, comma separated)
   |        |
   |        v
   |   usage.py      UsageLogger callback, fires on every LLM call (success or error)
   |        |
   |        +--> usage_logs table  (read by the dashboard)
   |        +--> data/raw_usage.jsonl  (append-only, provider's raw usage block, same timestamp)
   |
   +-- metrics.py     pure functions: totals, windows, error rate, latency, acceptance, time saved
   +-- routes_dashboard.py  role-scoped data + external-log import

SQLite (WAL) via SQLAlchemy:  users, usage_logs, recommendations, decisions, jobs, job_items,
sentiment_results, external_usage, admin_requests, user_limits, rate_events
```

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + Pydantic | typed request/response, auto docs at `/docs` |
| Auth | bcrypt + JWT (HS256, 60 min) | standard, stateless; user re-read from DB each request |
| Agents | LangGraph graphs / validated single calls | explicit steps: prepare, classify, validate, retry once |
| LLM access | LangChain chat models, provider chain | one callback point for logging across providers |
| Storage | SQLite (WAL) | zero setup for a demo; SQLAlchemy makes PostgreSQL a connection-string change |
| UI | one static HTML page | no build step; easy to run locally |

## 2. Identity and access

- `get_current_user` (security.py) is the **only** source of identity: it decodes the JWT, takes the user ID from the
  token, and loads the user from the database. No user ID is ever read from the URL or request body.
- The role is read from the database on every request, so a role change applies immediately.
- `scope_to_user(stmt, column, user)` wraps every per-user query: admin gets all rows, user gets `WHERE user_id = me`.
- The server refuses to start if `JWT_SECRET` is empty, a placeholder, or under 32 characters (anyone who knew it could
  sign an admin token).
- Self sign-up creates plain users only; admin rights need an existing admin's approval (`admin_requests`).

## 3. CV classification flow

1. `POST /cv/classify` (single) or `POST /cv/batch` (multiple, progress at `GET /cv/batch/{id}`, one bad file does not
   stop the batch). Accepts .pdf, .txt, .md; max 5 MB; PDFs via pypdf; scanned PDFs rejected with a clear message.
2. `enforce_budget` and `enforce_rate_limit` run before any LLM call (HTTP 429 when exceeded).
3. LangGraph: **prepare** (strip invisible characters, wrap CV in a random per-request tag, regex pre-scan) ->
   **classify** (LLM call, logged) -> **validate** -> retry once with the validation errors fed back -> **finalize**.
4. Validation: valid JSON against the Pydantic schema (`GET /cv/schema`); all 9 category scores present; primary has the
   top score; every evidence quote is found in the CV and is not steering text.
5. HR review and role fit run alongside; their failure never breaks the classification.
6. Result stored in `recommendations`. The user then Accepts, Modifies or Rejects; a reason is mandatory for Modify and
   Reject (server-enforced). Decisions are append-only with user, original recommendation, final categories, reason,
   timestamp. The latest decision per recommendation counts.

**Prompt-injection defence (layers):** cleaned text + random delimiter tag; "CV is data" system prompt; evidence-must-
exist-in-CV check (the real barrier: a hijacked answer cannot cite genuine evidence); pattern scan + model flag
(`injection_detected`).

## 4. Sentiment flow

Input is pasted or uploaded (.txt/.csv) and shown for review before any tokens are spent (max 50 lines). Each line is
analysed **5 times in parallel**; confidence is the share of runs agreeing on the winning label (measured agreement, not
self-reported). Key phrases must appear verbatim in the text. Results carry `source`: `user` (real) or `ai_sample`
(synthetic, never counted in time saved).

## 5. Usage logging (the core)

- One `UsageLogger` callback per request, built from the authenticated user, so concurrent users cannot mix logs.
- On success it reads tokens from the provider's response `usage` block (fallback: LangChain's normalised
  `usage_metadata`); never estimated. If a successful response has no usage, the row is flagged `usage_reported=false`.
- On error it records an error row with 0 tokens. Retries and fallback attempts are real calls and are logged.
- Fields: timestamp, run_id, user_id, agent, model, input/output tokens, latency (ms), status, error, cost_usd, raw usage.
- Cost: the provider-reported cost if present (OpenRouter), otherwise tokens x price table (an estimate).
- Two writes from the same callback with the same timestamp: the database row and the raw JSONL line. This is what makes
  reconciliation possible.
- `scripts/reconcile.py` compares raw JSONL vs database call by call and in total; current run: 281 calls, 149,936 input
  / 50,241 output tokens, PASS. Limit: both sides come from our logger, so this proves the pipeline preserves the
  provider's numbers, not the provider's invoice.

## 6. Provider fallback (llm.py)

`LLM_PROVIDER` is one provider or an ordered chain. Providers without a key are skipped. On any exception the next
provider is tried; the failed attempt is logged as an error row. A provider that just failed is tried last for 60
seconds (never excluded). Each provider has its own model setting; one provider configured uses `MODEL_NAME`.

## 7. Dashboard (metrics.py, routes_dashboard.py)

Total input/output tokens and cost; by user and by agent; runs in last 24 h / 7 d / 30 d; error rate and average latency
per agent; acceptance rate per category = accepted / (accepted + modified + rejected), pending excluded; time saved
from configurable assumptions (10 min by hand vs 2 to review per CV; 1 vs 0.25 per real feedback; Accept full gain,
Modify half, Reject costs the review time). Admin sees all users; a user sees only their own.
Admins can import another agent's log (JSONL/JSON/CSV); columns are auto-detected and it appears as its own tab.

## 8. Cost controls

Monthly token budget per user (default 500,000; admins unlimited; per-user override) computed from `usage_logs`; rate
limit per user per rolling minute stored in the database so every worker sees the same counts.

## 9. Evaluation and verification

- `eval/run_eval.py`: precision/recall per category. Easy set 45 CVs (100%); hard set 40 CVs (39/40; secondary category
  found for 5 of 8 mixed CVs). Both sets are synthetic.
- `scripts/demo_multiuser.py`: alice and bob concurrently, then carol.
- `pytest -q`: 190 tests, all offline (a fake model), about 7 minutes.

## 10. Known limits (say these first)

- Evaluation data is synthetic; all results are from one model (gpt-4o-mini).
- Dashboard "runs" are LLM calls, so retries, 5 sentiment votes and fallback attempts inflate runs and the error rate.
- Time saved is assumed, not measured; CV confidences are the model's own and not calibrated.
- SQLite, one machine, shared demo password, no login throttling; stored CV text is personal data (needs a retention
  policy).
- `/dev/config` is no longer used by the UI.
