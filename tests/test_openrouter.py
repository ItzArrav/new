import logging

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from sqlalchemy import select

from app.config import estimate_cost, settings
from app.db import SessionLocal
from app.llm import FakeChatModel, get_chat_model
from app.models import UsageLog, User
from app.usage import UsageLogger


def test_openrouter_provider_builds_an_openai_compatible_client(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-test")
    monkeypatch.setattr(settings, "model_name", "openai/gpt-4o-mini")
    model = get_chat_model()
    assert type(model).__name__ == "ChatOpenAI"
    assert "openrouter.ai" in str(model.openai_api_base) and model.model_name == "openai/gpt-4o-mini"
    assert model.openai_api_key.get_secret_value() == "sk-or-test"  # the OpenRouter key, never the OpenAI one


def test_openai_provider_still_uses_openais_own_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-openai-test")
    monkeypatch.setattr(settings, "model_name", "gpt-4o-mini")
    assert "openrouter" not in str(get_chat_model().openai_api_base or "")


def test_free_models_cost_nothing_and_known_ones_use_the_price_table(caplog):
    with caplog.at_level(logging.WARNING):
        assert estimate_cost("meta-llama/llama-3.3-70b-instruct:free", 5000, 2000) == 0.0
    assert "No price configured" not in caplog.text
    assert estimate_cost("openai/gpt-4o-mini", 1_000_000, 1_000_000) == 0.75  # 0.15 + 0.60


class _OpenRouterLike(FakeChatModel):
    """Returns a message shaped like an OpenRouter reply: real token counts and (optionally) a real cost."""
    cost: float | None = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        usage = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}
        if self.cost is not None:
            usage["cost"] = self.cost
        msg = AIMessage(content="ok", usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
                        response_metadata={"model_name": "openai/gpt-4o-mini", "token_usage": usage})
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _uid():
    with SessionLocal() as db:
        return db.scalar(select(User.id).where(User.username == "alice"))


def test_cost_reported_by_the_provider_is_recorded_as_is():
    _OpenRouterLike(cost=0.000123).invoke("hi", config={"callbacks": [UsageLogger(_uid(), "cv_agent")]})
    with SessionLocal() as db:
        row = db.scalars(select(UsageLog)).one()
    assert (row.input_tokens, row.output_tokens) == (120, 30)  # tokens still come straight from the usage block
    assert row.cost_usd == 0.000123 and row.model == "openai/gpt-4o-mini"


def test_without_a_reported_cost_the_price_table_estimate_is_used():
    _OpenRouterLike().invoke("hi", config={"callbacks": [UsageLogger(_uid(), "cv_agent")]})
    with SessionLocal() as db:
        row = db.scalars(select(UsageLog)).one()
    assert round(row.cost_usd, 8) == round((120 * 0.15 + 30 * 0.60) / 1_000_000, 8)


def test_a_nonsense_reported_cost_is_ignored():
    from app.usage import _reported_cost
    for bad in (-1.0, True, "0.5", None, "free"):
        assert _reported_cost({"cost": bad}) is None, bad
    assert _reported_cost(None) is None and _reported_cost({}) is None
    assert _reported_cost({"cost": 0}) == 0.0 and _reported_cost({"cost": 0.25}) == 0.25  # zero is a valid real cost

    # end to end: a negative reported cost falls back to the price-table estimate
    _OpenRouterLike(cost=-1.0).invoke("hi", config={"callbacks": [UsageLogger(_uid(), "cv_agent")]})
    with SessionLocal() as db:
        cost = db.scalars(select(UsageLog)).one().cost_usd
    assert 0 < cost < 0.001
