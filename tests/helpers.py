import json

from app.agents.cv_checks import CATEGORIES
from app.llm import FakeChatModel


class ScriptedChatModel(FakeChatModel):
    """Returns queued replies, one per call (fake token counts, offline)."""

    replies: list[str] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.reply = self.replies.pop(0)
        return super()._generate(messages, stop, run_manager, **kwargs)


def make_reply(primary, secondary=(), evidence=(), injection=False, top=0.9):
    scores = [{"category": c, "confidence": top if c == primary else 0.1} for c in CATEGORIES]
    return json.dumps(
        {
            "primary_category": primary,
            "secondary_categories": list(secondary),
            "scores": scores,
            "rationale": "Based on the experience quoted in the CV.",
            "evidence": list(evidence),
            "injection_detected": injection,
        }
    )
