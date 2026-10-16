"""Pure-Python (stdlib only) helpers for the CV agent: categories, text cleaning,
injection detection and the semantic checks applied to every model answer."""
import re
from difflib import SequenceMatcher

CATEGORIES = [
    "PSS", "DCS", "Front-End Developer", "Back-End Developer", "Web Automation",
    "Mobile Automation", "AI Engineer", "Chatbot Application Tester", "Other",
]

DEFINITIONS = {
    "PSS": "Experience with airline reservation, inventory, ticketing, or PSS (Passenger Service System) platforms",
    "DCS": "Experience with check-in, boarding, load control, or DCS (Departure Control System) platforms",
    "Front-End Developer": "Angular, React, Ionic, front-end, Tailwind CSS, Next.js, TanStack Query, Vite, Git, etc.",
    "Back-End Developer": "JavaScript/TypeScript with Node.js (Express, NestJS, Fastify), Python (Django, FastAPI, Flask), Java or Kotlin (Spring Boot), etc.",
    "Web Automation": "Web UI test automation using tools such as Selenium, Playwright, or Cypress",
    "Mobile Automation": "Mobile app test automation using tools such as Appium, Espresso, or XCUITest",
    "AI Engineer": "Building ML or LLM-based systems, including RAG, agents, fine-tuning, or model deployment",
    "Chatbot Application Tester": "Testing conversational applications: intent coverage, dialogue flows, LLM output quality",
    "Other": "Profiles that do not clearly fit any category above",
}

MAX_CV_CHARS = 20_000

INJECTION_RE = re.compile(
    "|".join(
        [
            r"ignore (all |any |the )*(previous|prior|above|earlier) (instructions?|prompts?|rules?)",
            r"disregard (all |any |the )*(previous|prior|above|earlier)",
            r"forget (all |any |the )*(previous|prior|above|earlier)",
            r"you are now\b",
            r"system prompt",
            r"new instructions?\s*:",
            r"classify (this|me|the candidate|this candidate|this cv) as",
            r"(set|give|assign|output|return) (the )?(confidence|score)",
            r"</?\s*(system|assistant|user|cv\w*)\s*>",
            r"do not (follow|obey)",
            r"\bjailbreak\b",
        ]
    ),
    re.IGNORECASE,
)

_INVISIBLE = re.compile(r"[\u200b-\u200f\u2060\u202a-\u202e\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]")
_DELIMITER = re.compile(r"</?\s*cv_[a-z0-9]+\s*>", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Drop invisible/control characters (a common place to hide instructions) and cap length."""
    return _INVISIBLE.sub("", text)[:MAX_CV_CHARS]


def neutralize(text: str) -> str:
    """Stop CV text from imitating our delimiter tags."""
    return _DELIMITER.sub("[removed-tag]", text)


def find_injection(text: str) -> list[str]:
    return [m.group(0) for m in INJECTION_RE.finditer(text)][:5]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


_WORD = re.compile(r"\S+")


def _key(word: str) -> str:
    """'CI/CD', 'CI_CD' and 'ci-cd,' all become 'cicd', so punctuation and case differences do not matter."""
    return re.sub(r"[^a-z0-9]", "", word.lower())


def snap_quote(quote: str, cv_text: str, min_ratio: float = 0.85) -> str | None:
    """Find where `quote` really is in the CV and return the CV's OWN wording of that passage, or None if the quote
    is not in the CV. Real CVs (especially PDFs) rarely match a model's quote character for character: '/' vs '_',
    '&' vs 'and', line breaks inside a sentence, cut-off words. Matching is on words (ignoring punctuation and case),
    and what is returned is cut from the CV text, so a quote that is shown is always genuine CV text.
    Short quotes (under 3 words) must match exactly; anything that is mostly not in the CV is rejected."""
    q_words = [k for k in (_key(w) for w in quote.split()) if k]
    if not q_words:
        return None
    cv = [(m.start(), m.end(), _key(m.group(0))) for m in _WORD.finditer(cv_text)]
    cv = [t for t in cv if t[2]]
    if not cv:
        return None
    cv_keys = [t[2] for t in cv]
    vocab = set(cv_keys)
    by_prefix: dict[str, list[str]] = {}
    for k in vocab:
        by_prefix.setdefault(k[:4], []).append(k)

    def canon(k: str) -> str:  # a cut-off word ('deplo') counts as the CV word it starts ('deployment')
        if k in vocab or len(k) < 4:
            return k
        for c in by_prefix.get(k[:4], []):
            if c.startswith(k) or k.startswith(c):
                return c
        return k

    q = [canon(k) for k in q_words]
    n = len(q)
    need = 1.0 if n < 3 else min_ratio
    qset = set(q)
    member = [1 if k in qset else 0 for k in cv_keys]
    prefix = [0]
    for m in member:
        prefix.append(prefix[-1] + m)
    best, best_span = 0.0, None
    for s, k in enumerate(cv_keys):
        if k not in qset:  # a matching passage can only start at one of the quote's own words
            continue
        for length in range(max(1, n - 2), n + 3):
            e = s + length
            if e > len(cv_keys) or prefix[e] - prefix[s] < 0.6 * min(n, length):
                continue
            ratio = SequenceMatcher(None, q, cv_keys[s:e], autojunk=False).ratio()
            if ratio > best:
                best, best_span = ratio, (s, e)
    if best_span is None or best < need:
        return None
    s, e = best_span
    return re.sub(r"\s+", " ", cv_text[cv[s][0]:cv[e - 1][1]]).strip()[:300]


def check_result(r, cv_text: str) -> list[str]:
    """Return a list of problems with a parsed answer (empty list = valid).
    `r` needs: primary_category, secondary_categories, scores[.category/.confidence],
    rationale, evidence."""
    errors: list[str] = []

    cats = [s.category for s in r.scores]
    if sorted(cats) != sorted(CATEGORIES):
        errors.append("scores must contain each of the 9 categories exactly once")
    else:
        conf = {s.category: s.confidence for s in r.scores}
        if conf[r.primary_category] < max(conf.values()):
            errors.append("primary_category must have the highest confidence")
        for c in r.secondary_categories:
            if conf[c] <= 0:
                errors.append(f"secondary category {c} has zero confidence")

    if r.primary_category in r.secondary_categories:
        errors.append("primary_category must not repeat in secondary_categories")
    if len(set(r.secondary_categories)) != len(r.secondary_categories):
        errors.append("secondary_categories contains duplicates")
    if "Other" in r.secondary_categories:
        errors.append("'Other' cannot be a secondary category")
    if r.primary_category == "Other" and r.secondary_categories:
        errors.append("secondary_categories must be empty when primary is 'Other'")

    if not r.rationale.strip() or len(r.rationale) > 1500:
        errors.append("rationale must be 1-1500 characters")

    if r.primary_category != "Other" and not r.evidence:
        errors.append("evidence must contain at least one quote from the CV")
    if len(r.evidence) > 8:
        errors.append("at most 8 evidence quotes")

    # Each quote is matched to the passage it points at (see snap_quote) and replaced by the CV's own wording, so a
    # quote with a different '/', '&', line break or cut-off last word still counts, and what is shown is real CV text.
    checked: list[str] = []
    for q in r.evidence:
        if len(q.strip()) < 8:
            errors.append(f"evidence quote too short: {q!r}")
            continue
        real = snap_quote(q, cv_text)
        if real is None:
            errors.append(f"evidence quote not found in the CV: {q[:60]!r}")
        elif INJECTION_RE.search(q) or INJECTION_RE.search(real):
            errors.append(f"evidence quote is injected steering text, not experience: {q[:60]!r}")
        elif real not in checked:
            checked.append(real)
    if not errors:
        r.evidence[:] = checked
    return errors
