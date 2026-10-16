import logging

from dotenv import dotenv_values
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./data/app.db"
    usage_jsonl_path: str = "./data/raw_usage.jsonl"

    jwt_secret: str = ""  # required; see jwt_secret_problem()
    jwt_minutes: int = 60

    # One provider, or a fallback order such as "openai,anthropic,openrouter": the next one is tried when the previous
    # fails for any reason. Providers without a key are skipped.
    llm_provider: str = "anthropic"  # anthropic | openai | openrouter | fake (comma separated for a chain)
    model_name: str = "claude-haiku-4-5-20251001"  # the model when ONE provider is configured
    # Models used per provider when LLM_PROVIDER is a chain (each provider has its own model names):
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    openrouter_model: str = "openai/gpt-4o-mini"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    # OpenRouter speaks the OpenAI API, so it reuses the OpenAI client with a different base URL.
    # MODEL_NAME then uses OpenRouter's "vendor/model" names, e.g. "openai/gpt-4o-mini".
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Sentiment confidence = share of independent runs that agree on the label (a measured score, not the model's own
    # guess). Each feedback is analysed this many times in parallel, so it costs this many times the tokens. 1 = off.
    sentiment_votes: int = 5

    # Check the role fit for the CV's own category automatically after classification (2 extra LLM calls per CV).
    auto_role_fit: bool = True

    seed_password: str = "demo-pass-123"

    # Self sign-up (POST /auth/signup). New accounts are always plain users; admin rights need an admin's approval.
    allow_signup: bool = True
    signup_max_per_hour: int = 20  # all new accounts platform-wide per rolling hour: limits abuse of the LLM budget

    # Cost controls. 0 = unlimited. Admins are unlimited unless given an explicit override.
    default_monthly_token_budget: int = 500_000  # input + output tokens per user per calendar month (UTC)
    rate_limit_per_minute: int = 60              # items per user per rolling minute (a CV or a feedback text = 1)

    # Time-saved ASSUMPTIONS (minutes). Not measured: state them in your report.
    manual_min_per_cv: float = 10.0
    review_min_per_cv: float = 2.0
    manual_min_per_feedback: float = 1.0
    review_min_per_feedback: float = 0.25
    modify_credit: float = 0.5  # share of the saving kept when the user modifies a result


settings = Settings()

# Only the API keys: a key written in .env beats a stale machine-wide variable.
# Everything else keeps the normal order (environment variable beats .env), which the tests rely on.
for _field, _name in (("openai_api_key", "OPENAI_API_KEY"), ("anthropic_api_key", "ANTHROPIC_API_KEY"),
                      ("openrouter_api_key", "OPENROUTER_API_KEY")):
    _val = (dotenv_values(".env").get(_name) or "").strip()
    if _val:
        setattr(settings, _field, _val)

MIN_SECRET_LENGTH = 32
SECRET_HELP = ('Generate one with:  python -c "import secrets; print(secrets.token_urlsafe(48))"  '
               "and put it in .env as JWT_SECRET=<that value>.")


def jwt_secret_problem(secret: str) -> str | None:
    """Why this login-token signing secret is unsafe, or None. Anyone who knows the secret can sign a token for any
    user, including an admin, so an empty, example or short secret is refused."""
    if not secret or secret.lower().startswith("change-me") or secret in {"secret", "changeme", "your-secret-here"}:
        return "JWT_SECRET is missing or is still the example placeholder"
    if len(secret) < MIN_SECRET_LENGTH:
        return f"JWT_SECRET is too short ({len(secret)} characters; at least {MIN_SECRET_LENGTH} are needed)"
    return None


# USD per 1M tokens: (input, output).
# These are ESTIMATES. Verify against Anthropic's pricing page before reporting cost.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5-5": (3.00, 15.00),  # placeholder: verify
    "gpt-4o-mini": (0.15, 0.60),  # OpenAI: verify on their pricing page
    "gpt-4.1-mini": (0.40, 1.60),  # OpenAI: verify
    # OpenRouter uses "vendor/model" names. Prices there change often and can differ per upstream provider: verify.
    # When OpenRouter reports the real cost in the response, that figure is used instead (see usage.py).
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "fake-model": (0.0, 0.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model.endswith(":free"):  # OpenRouter's free models cost nothing (they have strict rate limits instead)
        return 0.0
    price = PRICES_PER_MTOK.get(model)
    if price is None:  # tolerate dated suffixes
        for name, p in PRICES_PER_MTOK.items():
            if model.startswith(name):
                price = p
                break
    if price is None:
        log.warning("No price configured for model %r; cost recorded as 0", model)
        return 0.0
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
