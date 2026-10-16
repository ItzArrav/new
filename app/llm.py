import logging
import threading
import time

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .config import settings

log = logging.getLogger(__name__)


class FakeChatModel(BaseChatModel):
    """Offline stand-in used by tests. Its token counts are NOT real API usage;
    it exists only to exercise the logging pipeline without a network or API key."""

    model_name: str = "fake-model"
    reply: str = "pong"

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        in_tok = sum(len(str(m.content).split()) for m in messages)
        out_tok = len(self.reply.split())
        msg = AIMessage(
            content=self.reply,
            usage_metadata={"input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": in_tok + out_tok},
            response_metadata={
                "model_name": self.model_name,
                "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
            },
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


PROVIDERS = ("openai", "anthropic", "openrouter")
COOLDOWN_SECONDS = 60  # a provider that just failed is tried last for this long (never excluded completely)
_now = time.monotonic
_lock = threading.Lock()
_down_until: dict[str, float] = {}  # provider -> monotonic time until which it is tried last


class AllProvidersFailed(RuntimeError):
    """Every configured provider failed. The message lists each provider and its error."""


def provider_chain() -> list[str]:
    """LLM_PROVIDER is one provider ('openai') or a fallback order ('openai,anthropic,openrouter')."""
    names = [n.strip().lower() for n in settings.llm_provider.split(",") if n.strip()]
    for n in names:
        if n not in (*PROVIDERS, "fake"):
            raise ValueError(f"Unknown LLM provider {n!r}. Use one or more of: {', '.join(PROVIDERS)} (comma separated)")
    return names or ["openai"]


def _api_key(name: str) -> str | None:
    return {"openai": settings.openai_api_key, "anthropic": settings.anthropic_api_key,
            "openrouter": settings.openrouter_api_key}.get(name)


def _has_key(name: str) -> bool:
    return bool((_api_key(name) or "").strip())


def _build_one(name: str, single: bool) -> BaseChatModel:
    """One provider's client. `single` = the only provider configured: uses MODEL_NAME and the SDK's normal retries.
    In a fallback chain each provider has its own model setting and fails fast, so the next one can take over."""
    extra = {} if single else {"max_retries": 1}
    if name == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=settings.model_name if single else settings.anthropic_model, temperature=0,
                             max_tokens=1024, api_key=settings.anthropic_api_key, **extra)
    from langchain_openai import ChatOpenAI

    if name == "openrouter":  # OpenRouter speaks the OpenAI API
        return ChatOpenAI(model=settings.model_name if single else settings.openrouter_model, temperature=0,
                          max_tokens=1024, api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url, **extra)
    return ChatOpenAI(model=settings.model_name if single else settings.openai_model, temperature=0,
                      max_tokens=1024, api_key=settings.openai_api_key, **extra)


class FallbackChatModel:
    """Tries the providers in order and moves on to the next one on ANY failure (quota, bad key, rate limit, outage,
    timeout ...). Each attempt goes through LangChain with the caller's callbacks, so the usage logger records every
    failed attempt as an error row and the successful one as a normal row, with the real model name."""

    def __init__(self, names: list[str]):
        self.names = names

    def invoke(self, input, config=None, **kwargs):
        usable = [n for n in self.names if _has_key(n)]
        if not usable:
            raise AllProvidersFailed("No API key is set for any provider in LLM_PROVIDER=" + ",".join(self.names))
        now = _now()
        with _lock:
            fresh = [n for n in usable if _down_until.get(n, 0) <= now]
        order = fresh + [n for n in usable if n not in fresh]  # providers that failed recently go last, not away
        errors = []
        for name in order:
            try:
                reply = _build_one(name, single=False).invoke(input, config=config, **kwargs)
            except Exception as e:  # deliberately any exception: "whatever the reason", try the next provider
                with _lock:
                    _down_until[name] = _now() + COOLDOWN_SECONDS
                errors.append(f"{name}: {type(e).__name__}: {str(e)[:160]}")
                log.warning("LLM provider %s failed (%s); trying the next one", name, type(e).__name__)
                continue
            with _lock:
                _down_until.pop(name, None)
            return reply
        raise AllProvidersFailed("All LLM providers failed. " + " | ".join(errors))


def get_chat_model():
    names = provider_chain()
    if names == ["fake"]:
        return FakeChatModel()
    if len(names) == 1:
        return _build_one(names[0], single=True)  # exactly the old single-provider behaviour
    return FallbackChatModel(names)
