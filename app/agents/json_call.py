"""Shared by the small single-call agents: ask the model for a JSON object, validate it,
retry once with the reason it was rejected. Every call is usage-logged through `usage_config`."""
import json
import re
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage

from ..llm import get_chat_model
from ..usage import usage_config

MAX_ATTEMPTS = 2


def extract_json(text: str) -> str:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    a, b = text.find("{"), text.rfind("}")
    return text[a : b + 1] if a != -1 and b > a else text


def run_json_agent(user, agent: str, messages: list, validate: Callable[[dict], tuple[dict | None, str | None]]) -> dict:
    """validate(obj) -> (clean_result, None) or (None, problem). Returns {ok, result, error, attempts}."""
    config = usage_config(user, agent)
    msgs, error = list(messages), "no attempt"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        raw = str(get_chat_model().invoke(msgs, config=config).content)
        try:
            obj = json.loads(extract_json(raw))
            if not isinstance(obj, dict):
                raise ValueError("answer must be a JSON object")
            result, problem = validate(obj)
        except ValueError as e:
            result, problem = None, f"invalid JSON: {str(e)[:200]}"
        if result is not None:
            return {"ok": True, "result": result, "error": None, "attempts": attempt}
        error = problem or "rejected"
        msgs += [AIMessage(content=raw), HumanMessage(content=f"Your previous answer was rejected: {error}. Return ONLY a corrected JSON object.")]
    return {"ok": False, "result": None, "error": error, "attempts": MAX_ATTEMPTS}
