# Multi-user agent platform with usage analytics

Logged-in users run AI agents on CVs and feedback; every LLM call is logged from the provider's own usage figures, and a
dashboard shows tokens, cost, usage, errors, acceptance and time saved. The write-up with all results is in
[REPORT.md](REPORT.md).

- **Required agents:** CV classification (9 categories, multi-label, evidence-backed) and feedback sentiment.
- **Additional agents:** HR review, role fit and a sample-feedback generator (see REPORT.md section 3).
- **Stack:** FastAPI, LangGraph / LangChain, SQLite (WAL), one HTML page for the UI. Model: `gpt-4o-mini` through a
  provider fallback chain.

## Run locally
```
pip install -r requirements.txt
cp .env.example .env        # then fill in at least one API key and JWT_SECRET (see below)
python -m app.seed          # creates tables and the demo users admin, alice, bob, carol
uvicorn app.main:app
```
- UI: http://localhost:8000 · API docs: http://localhost:8000/docs
- Demo users log in with the `SEED_PASSWORD` from `.env` (demo only; change it for anything shared).
- **`JWT_SECRET` is required.** The server refuses to start if it is empty, the example placeholder, or shorter than 32
  characters. Generate one: `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
- Keys and settings are read once at startup: restart the server after editing `.env`.
- `python -m scripts.set_key openai|anthropic|openrouter` stores a key in `.env` from a hidden prompt and tests it.

Docker: `docker compose up --build` (reads `.env` at run time; `.dockerignore` keeps keys and data out of the image).
The image has not been built or run on this machine, so treat the Dockerfile as untested.

## Reproduce the results in REPORT.md
```
python -m eval.make_dataset          # easy set: 45 CVs + 5 prompt-injection CVs (synthetic)
python -m eval.make_hard_dataset     # hard set: 40 CVs, neutral titles, mixed profiles, near-miss "Other"
python -m eval.run_eval              # -> eval/out/report.md
python -m eval.run_eval --set hard   # -> eval/out/report_hard.md
uvicorn app.main:app                 # in another terminal
python -m scripts.demo_multiuser     # alice + bob at the same time, then carol -> both agents, decisions
python -m scripts.reconcile          # dashboard numbers vs raw API usage, call by call -> PASS / FAIL
```
`data/app.db` and `data/raw_usage.jsonl` are git-ignored. `eval/out/` holds the evidence from the last clean run,
including a raw log sample (`raw_usage_sample.jsonl`) and the admin dashboard data (`dashboard_admin.json`).

## Choosing the LLM provider
`LLM_PROVIDER` is one provider or a fallback order, for example `openai,anthropic,openrouter`.

| Provider | Key | Model setting (in a chain) |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | `OPENAI_MODEL` (default `gpt-4o-mini`) |
| `anthropic` | `ANTHROPIC_API_KEY` | `ANTHROPIC_MODEL` |
| `openrouter` | `OPENROUTER_API_KEY` | `OPENROUTER_MODEL` (default `openai/gpt-4o-mini`) |
| `fake` | none | offline tests only; its token counts are fake |

- With one provider, `MODEL_NAME` is used. In a chain the next provider is tried on any failure (bad key, no credits,
  outage); providers without a key are skipped; a provider that just failed is tried last for 60 seconds.
- Every attempt is logged: a failed attempt is an error row with 0 tokens, so it counts in the dashboard error rate.
- OpenRouter returns the real cost of each call, which is stored instead of the price-table estimate.
- The header badge in the UI shows which provider answered the last call.

## Layout
```
app/main.py                login, /me, usage endpoints, startup check of JWT_SECRET
app/config.py              settings, price table, provider chain settings
app/security.py            bcrypt, signed tokens, get_current_user, require_admin, scope_to_user
app/usage.py               UsageLogger: one callback logs every LLM call to usage_logs + data/raw_usage.jsonl
app/llm.py                 model factory and provider fallback chain
app/limits.py              monthly token budgets and per-minute rate limits
app/metrics.py             dashboard maths, acceptance rate, time saved, external log import
app/agents/cv_agent.py     CV classification (LangGraph: prepare -> classify -> validate -> retry once -> finalize)
app/agents/cv_schema.py    the documented output schema (GET /cv/schema)
app/agents/cv_checks.py    categories, injection patterns, quote matching, answer validation
app/agents/sentiment_agent.py   feedback sentiment (same pattern; confidence = agreement of SENTIMENT_VOTES runs)
app/agents/cv_checker.py   HR review (one LLM call + 12 rule checks in cv_rules.py), runs with every classification
app/agents/role_fit.py     role-fit checklists and scoring (quote-backed, double-checked), automatic for the CV's own role
app/agents/job_roles.py    jobs users add for role fit: lookup, validation, AI draft of a checklist from a job description
app/agents/feedback_generator.py   synthetic sample feedback about one CV
app/routes_cv.py           POST /cv/classify, GET /cv/recommendations, GET /cv/schema
app/routes_batch.py        POST /cv/batch, GET /cv/batch/{id} (progress)
app/routes_decisions.py    accept / modify / reject (reason mandatory for modify and reject)
app/routes_sentiment.py    POST /sentiment/parse (.txt/.csv, no LLM), POST /sentiment/analyze, GET /sentiment/results
app/routes_rolefit.py      POST /cv/recommendations/{id}/role-fit {role | all}; role can be an added job
app/routes_roles.py        GET /roles, GET /roles/find?q=, POST /roles/draft (job description -> checklist, not saved),
                           POST /roles (save, after review), DELETE /roles/{id} (who added it, or an admin)
app/routes_feedback.py     POST /feedback/generate {recommendation_id}
app/routes_dashboard.py    GET /dashboard/data, /dashboard/sources, POST /dashboard/import (admin)
app/routes_admin.py        sign-up, admin-created users, admin-access requests
app/routes_limits.py       budgets: GET /me/budget, GET /admin/budgets, PUT /admin/users/{id}/budget
app/static/index.html      the UI (Liquid Glass theme, light/dark, three.js 3D wallpaper and charts; works without WebGL)
eval/                      datasets, evaluation runner, metrics; eval/out = evidence
scripts/                   demo_multiuser, reconcile, backfill_log_ts, set_key
tests/                     offline tests (fake model, no API key needed)
```

## Design rules
- Every LLM call gets `config=usage_config(user, "agent_name")`; nothing else logs.
- User identity comes only from the token (`get_current_user`), never from the URL or body.
- Every per-user query goes through `scope_to_user(...)`.
- Token counts come from the provider response. A successful call without usage data is flagged
  `usage_reported=false`, never estimated.
- Every quote an agent shows must be found in the CV, and what is shown is the CV's own wording.

## Tests
```
pytest -q
```
The tests run offline with a fake model (its token counts are fake); they check logic, scoping, logging, validation
and the API. The results in REPORT.md come from real API calls.

## Cost controls
- Monthly token budget per user (`DEFAULT_MONTHLY_TOKEN_BUDGET`, 500,000; admins can set per-user values). Over budget:
  HTTP 429 with the reset date.
- Rate limit per user (`RATE_LIMIT_PER_MINUTE` items: one CV or one feedback line = 1). Over the limit: HTTP 429.
- Batches and feedback lists re-check the budget per item. Calls already in flight can overshoot by a few.
- Extra calls per CV: the HR review (1 call) and role fit (2 calls) run automatically; `AUTO_ROLE_FIT=false` turns
  role fit off. Sentiment uses `SENTIMENT_VOTES` calls per feedback (default 5; 1 = no confidence is claimed).

## PDF input
PDFs are read with pypdf (first 15 pages). Hidden text (for example white on white) is extracted and goes through the
same defences as any CV text. Scanned PDFs have no text layer and are rejected (they would need OCR). .docx is not
supported.
