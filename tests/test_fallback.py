import json

import pytest
from sqlalchemy import select

from app import llm
from app.agents import json_call
from app.config import settings
from app.db import SessionLocal
from app.llm import AllProvidersFailed, FakeChatModel, get_chat_model
from app.models import UsageLog, User
from app.usage import UsageLogger


class Down(FakeChatModel):
    """A provider that fails (out of credits, bad key, outage ...)."""
    def _generate(self, *a, **k):
        raise RuntimeError("insufficient_quota: no credits remaining")


def good(name):
    return FakeChatModel(model_name=name, reply="hello from " + name)


@pytest.fixture(autouse=True)
def chain(monkeypatch):
    """OpenAI -> Anthropic -> OpenRouter, all 'configured'; clean cool-down state; calls recorded in `calls`."""
    monkeypatch.setattr(settings, "llm_provider", "openai,anthropic,openrouter")
    monkeypatch.setattr(llm, "_has_key", lambda name: True)
    monkeypatch.setattr(llm, "_down_until", {})
    clock = {"t": 1000.0}
    monkeypatch.setattr(llm, "_now", lambda: clock["t"])
    state = {"calls": [], "models": {"openai": Down(model_name="gpt-4o-mini"), "anthropic": good("claude-haiku"),
                                      "openrouter": good("openai/gpt-4o-mini")}, "clock": clock}

    def build(name, single):
        state["calls"].append(name)
        return state["models"][name]

    monkeypatch.setattr(llm, "_build_one", build)
    return state


def _alice_id():
    with SessionLocal() as db:
        return db.scalar(select(User.id).where(User.username == "alice"))


def _invoke():
    return get_chat_model().invoke("hi there", config={"callbacks": [UsageLogger(_alice_id(), "cv_agent")]})


def test_falls_back_to_the_next_provider_and_logs_both_attempts(chain):
    reply = _invoke()
    assert reply.content == "hello from claude-haiku" and chain["calls"] == ["openai", "anthropic"]
    with SessionLocal() as db:
        rows = db.scalars(select(UsageLog).order_by(UsageLog.id)).all()
    assert [(r.model, r.status) for r in rows] == [("gpt-4o-mini", "error"), ("claude-haiku", "success")]
    assert rows[0].input_tokens == 0 and "insufficient_quota" in rows[0].error  # the failure is visible, not hidden
    assert rows[1].input_tokens > 0  # tokens of the answer that was actually used come from its own usage block


def test_a_failed_provider_goes_last_for_a_while_but_is_tried_again_later(chain):
    _invoke()
    chain["calls"].clear()
    _invoke()  # openai failed a moment ago: the next call goes straight to anthropic
    assert chain["calls"] == ["anthropic"]
    chain["calls"].clear()
    chain["clock"]["t"] += llm.COOLDOWN_SECONDS + 1  # cool-down over: openai is tried first again
    chain["models"]["openai"] = good("gpt-4o-mini")  # ... and it has recovered
    assert _invoke().content == "hello from gpt-4o-mini" and chain["calls"] == ["openai"]


def test_providers_in_cooldown_are_still_used_as_a_last_resort(chain):
    chain["models"]["anthropic"] = Down(model_name="claude-haiku")
    reply = _invoke()  # openai down, anthropic down, openrouter answers
    assert reply.content == "hello from openai/gpt-4o-mini" and chain["calls"] == ["openai", "anthropic", "openrouter"]
    chain["calls"].clear()
    chain["models"]["openrouter"] = Down(model_name="openai/gpt-4o-mini")
    chain["models"]["anthropic"] = good("claude-haiku")  # anthropic recovered while still in its cool-down window
    assert _invoke().content == "hello from claude-haiku"  # fresh providers are all down, so the cooling ones get a turn
    assert set(chain["calls"]) == {"openrouter", "openai", "anthropic"}


def test_when_every_provider_fails_the_error_names_each_one(chain):
    for k in chain["models"]:
        chain["models"][k] = Down(model_name=k)
    with pytest.raises(AllProvidersFailed) as e:
        _invoke()
    msg = str(e.value)
    assert all(p in msg for p in ("openai:", "anthropic:", "openrouter:")) and "insufficient_quota" in msg
    with SessionLocal() as db:
        assert [r.status for r in db.scalars(select(UsageLog))] == ["error", "error", "error"]


def test_providers_without_a_key_are_skipped(monkeypatch, chain):
    monkeypatch.setattr(llm, "_has_key", lambda name: name == "openrouter")
    assert _invoke().content == "hello from openai/gpt-4o-mini" and chain["calls"] == ["openrouter"]
    monkeypatch.setattr(llm, "_has_key", lambda name: False)
    with pytest.raises(AllProvidersFailed, match="No API key"):
        _invoke()


def test_an_agent_keeps_working_when_its_first_provider_is_down(monkeypatch, chain):
    from app.agents.cv_checker import run_llm_part
    chain["models"]["anthropic"] = FakeChatModel(model_name="claude-haiku", reply=json.dumps(
        {"summary": "A short neutral summary.", "evidence_strength": "moderate", "reasons_to_hire": [],
         "points_to_consider": [], "concerns": [], "interview_questions": [], "soft_skills": [], "hard_skills": ["Python"]}))
    monkeypatch.setattr(json_call, "get_chat_model", get_chat_model)
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.username == "alice"))
    out = run_llm_part(user, "Backend engineer. Built APIs with Python.")
    assert out["ok"] and out["result"]["hard_skills"] == ["Python"]
    with SessionLocal() as db:
        assert [(r.agent, r.status) for r in db.scalars(select(UsageLog).order_by(UsageLog.id))] == [
            ("cv_checker", "error"), ("cv_checker", "success")]


def test_unknown_provider_names_are_rejected(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai,gemini")
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_chat_model()


def test_single_provider_behaves_exactly_as_before(monkeypatch):
    monkeypatch.undo()  # drop the chain fixture's patches
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "model_name", "gpt-4o-mini")
    model = get_chat_model()
    assert type(model).__name__ == "ChatOpenAI" and model.model_name == "gpt-4o-mini"


def test_chain_builds_each_provider_with_its_own_model_and_fails_fast(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(settings, "model_name", "ignored-in-a-chain")
    monkeypatch.setattr(settings, "openai_api_key", "sk-a")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-b")
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-c")
    o, a, r = (llm._build_one(n, single=False) for n in ("openai", "anthropic", "openrouter"))
    assert (type(o).__name__, o.model_name) == ("ChatOpenAI", settings.openai_model)
    assert type(a).__name__ == "ChatAnthropic" and a.model == settings.anthropic_model
    assert r.model_name == settings.openrouter_model and "openrouter.ai" in str(r.openai_api_base)
    assert o.max_retries == 1 and r.max_retries == 1  # fail fast so the next provider can take over
