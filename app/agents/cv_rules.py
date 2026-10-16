"""CV quality checks that need no LLM (stdlib only). Deterministic, so the same CV always gets the same result.

Each check returns {id, category, label, status, weight, detail, examples}; status is pass | warn | fail.
Categories: Impact, Brevity, Style. (Soft Skills comes from the LLM part, see cv_checker.py.)
These are heuristics about writing quality, NOT a judgement of the candidate.
"""
import re

BULLET_RE = re.compile(r"^\s*(?:[-*•●▪–—>]|\d{1,2}[.)])\s+(.*\S)")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?:\+?\d[\s().-]?){8,}")
METRIC_RE = re.compile(r"\d|%|\$|€|£|\b(?:million|thousand|billion|dozens?|hundreds?)\b", re.I)
PRONOUN_RE = re.compile(r"\b(?:I|me|my|myself)\b")
WEAK_RE = re.compile(
    r"\b(?:responsible for|worked on|helped(?: with| to)?|duties (?:included|include)|assisted(?: with| in)?|"
    r"tasked with|in charge of|participated in|involved in|exposure to)\b", re.I)
FILLER_RE = re.compile(r"\b(?:very|really|various|several|numerous|extremely|basically|quite|just|etc)\b", re.I)
BUZZWORDS = ["team player", "hard-working", "hardworking", "results-driven", "go-getter", "synergy",
             "think outside the box", "detail-oriented", "self-motivated", "dynamic", "passionate",
             "proven track record", "fast learner", "people person", "highly motivated"]

BASE_VERBS = set("""achieve analyze architect automate build collaborate coordinate create cut deliver deploy design develop
direct drive establish evaluate execute generate implement improve increase integrate introduce launch lead maintain manage
migrate mentor monitor negotiate optimize organize own plan present reduce refactor research resolve scale ship simplify
spearhead streamline test train transform write configure debug document investigate publish secure validate verify
administer support onboard review select audit prototype""".split())
WEAK_STARTERS = {"worked", "helped", "assisted", "handled", "used", "was", "were", "had", "did", "made", "got", "responsible",
                 "involved", "participated", "tasked", "duties", "experience", "knowledge", "familiar"}

SECTION_WORDS = {
    "experience": ("experience", "employment", "work history", "career history"),
    "education": ("education", "academic", "qualifications"),
    "skills": ("skills", "technologies", "tech stack", "competencies", "expertise"),
}
MONTH_ABBR = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?"
MONTH_FULL = r"(?:january|february|march|april|june|july|august|september|october|november|december)"
DATE_FORMS = {
    "Jan 2020": re.compile(rf"\b{MONTH_ABBR}\s+(?:19|20)\d\d\b", re.I),
    "January 2020": re.compile(rf"\b{MONTH_FULL}\s+(?:19|20)\d\d\b", re.I),
    "01/2020": re.compile(r"\b(?:0?[1-9]|1[0-2])[/.-](?:19|20)\d\d\b"),
}


def _words(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9'+#./%$-]+", s)


def _is_strong_start(first: str) -> bool:
    w = first.lower().strip(".,:;")
    if w in WEAK_STARTERS:
        return False
    if w in BASE_VERBS or w[:-1] in BASE_VERBS or w[:-2] in BASE_VERBS or w == "led":
        return True
    return w.endswith("ed") and len(w) > 4  # other past-tense verbs: counted as action verbs


def split_statements(text: str) -> tuple[list[str], bool]:
    """Return (the CV's bullet/achievement lines, whether real bullet markers were found).
    PDFs often lose bullet markers, so without markers every sentence-like line of 6+ words is used."""
    lines = [l.rstrip() for l in text.splitlines() if l.strip() and not _is_skill_list(l)]
    marked = [m.group(1).strip() for l in lines if (m := BULLET_RE.match(l)) and not _is_skill_list(m.group(1))]
    if len(marked) >= 3:
        return marked, True
    plain = [l.strip() for l in lines if len(_words(l)) >= 6 and not l.strip().endswith(":")]
    return plain, False


SKILL_LIST_RE = re.compile(r"^\s*(?:[-*•]\s*)?[A-Za-z][A-Za-z &/+().-]{1,40}:\s*[^:]*,[^:]*,")


def _is_skill_list(line: str) -> bool:
    """'Label: a, b, c' lines (tool and skill lists) are not achievement bullets, so they are not judged as such."""
    return bool(SKILL_LIST_RE.match(line))


def _is_heading_for(line: str, words_: tuple[str, ...]) -> bool:
    """A short line that names a section, e.g. 'Experience', 'Professional Experience', 'Technical Skills:'."""
    s = line.strip().rstrip(":").strip().lower()
    return bool(s) and len(s.split()) <= 5 and not s.endswith(".") and any(w in s for w in words_)


def _check(cid, category, label, status, weight, detail, examples=()):
    return {"id": cid, "category": category, "label": label, "status": status, "weight": weight,
            "detail": detail, "examples": list(examples)[:3]}


def _ratio_status(ratio: float, good: float, ok: float) -> str:
    return "pass" if ratio >= good else "warn" if ratio >= ok else "fail"


def run_rules(text: str) -> dict:
    words = _words(text)
    n_words = len(words)
    stmts, has_markers = split_statements(text)
    checks: list[dict] = []

    # ---------------- Impact ----------------
    if stmts:
        with_metric = [s for s in stmts if METRIC_RE.search(s)]
        r = len(with_metric) / len(stmts)
        no_metric = [s[:110] for s in stmts if not METRIC_RE.search(s)]
        checks.append(_check("metrics", "Impact", "Bullets with measurable results (numbers, %, $)",
                             _ratio_status(r, 0.4, 0.2), 3,
                             f"{len(with_metric)} of {len(stmts)} lines contain a number or metric (aim for 40%+).", no_metric))
        strong = [s for s in stmts if _is_strong_start(s.split()[0])]
        r = len(strong) / len(stmts)
        weak_start = [s[:110] for s in stmts if not _is_strong_start(s.split()[0])]
        checks.append(_check("action_verbs", "Impact", "Bullets that start with an action verb",
                             _ratio_status(r, 0.7, 0.4), 2,
                             f"{len(strong)} of {len(stmts)} lines start with a strong action verb.", weak_start))
        weak = [m.group(0) for s in stmts if (m := WEAK_RE.search(s))]
        weak_lines = [s[:110] for s in stmts if WEAK_RE.search(s)]
        checks.append(_check("weak_phrases", "Impact", "Weak phrases (\"responsible for\", \"worked on\" ...)",
                             "pass" if not weak else "warn" if len(weak) <= 2 else "fail", 2,
                             "None found." if not weak else f"{len(weak)} weak phrase(s): {', '.join(sorted(set(w.lower() for w in weak)))}.",
                             weak_lines))
    else:
        checks.append(_check("bullets", "Impact", "Achievement bullets", "fail", 3,
                             "No bullet-style achievement lines found, so impact cannot be assessed."))

    # ---------------- Brevity ----------------
    pages = n_words / 500
    status = "pass" if 250 <= n_words <= 800 else "warn" if (150 <= n_words < 250 or 800 < n_words <= 1100) else "fail"
    checks.append(_check("length", "Brevity", "Overall length", status, 3,
                         f"{n_words} words (about {pages:.1f} pages). Typical target: 250-800 words, at most 2 pages."))
    long_stmts = [s for s in stmts if len(_words(s)) > 35]
    checks.append(_check("bullet_length", "Brevity", "Bullets of 35 words or fewer",
                         "pass" if not long_stmts else "warn" if len(long_stmts) <= 2 else "fail", 2,
                         "No overlong lines." if not long_stmts else f"{len(long_stmts)} line(s) longer than 35 words.",
                         [s[:110] + "..." for s in long_stmts]))
    fillers = [m.group(0).lower() for m in FILLER_RE.finditer(text)]
    checks.append(_check("filler", "Brevity", "Filler words (very, really, various, several ...)",
                         "pass" if len(fillers) <= 2 else "warn" if len(fillers) <= 6 else "fail", 1,
                         "Few or none." if len(fillers) <= 2 else f"{len(fillers)} filler words: {', '.join(sorted(set(fillers)))}."))

    # ---------------- Style ----------------
    pron = [m.group(0) for m in PRONOUN_RE.finditer(text)]
    checks.append(_check("pronouns", "Style", "No personal pronouns (I, me, my)",
                         "pass" if not pron else "warn" if len(pron) <= 3 else "fail", 1,
                         "None found." if not pron else f"{len(pron)} personal pronoun(s) found."))
    lower = text.lower()
    found_buzz = [b for b in BUZZWORDS if b in lower]
    checks.append(_check("buzzwords", "Style", "Clichés and buzzwords", "pass" if not found_buzz else
                         "warn" if len(found_buzz) <= 2 else "fail", 2,
                         "None found." if not found_buzz else f"Found: {', '.join(found_buzz)}. Show it with evidence instead."))
    head = "\n".join(text.splitlines()[:15])
    has_email, has_phone = bool(EMAIL_RE.search(head)), bool(PHONE_RE.search(head))
    checks.append(_check("contact", "Style", "Contact details (email and phone near the top)",
                         "pass" if has_email and has_phone else "warn" if has_email or has_phone else "fail", 2,
                         f"Email: {'yes' if has_email else 'missing'}, phone: {'yes' if has_phone else 'missing'}."))
    present = {key: any(_is_heading_for(l, words_) for l in text.splitlines())
               for key, words_ in SECTION_WORDS.items()}
    missing = [k for k, v in present.items() if not v]
    checks.append(_check("sections", "Style", "Standard sections (Experience, Education, Skills)",
                         "pass" if not missing else "warn" if len(missing) == 1 else "fail", 2,
                         "All three found." if not missing else f"Heading not found for: {', '.join(missing)}."))
    forms = [name for name, rx in DATE_FORMS.items() if rx.search(text)]
    checks.append(_check("dates", "Style", "Consistent date format",
                         "pass" if len(forms) <= 1 else "warn", 1,
                         "Consistent." if len(forms) <= 1 else f"Mixed date styles: {', '.join(forms)}."))
    starts: dict[str, int] = {}
    for s in stmts:
        k = s.split()[0].lower()
        starts[k] = starts.get(k, 0) + 1
    repeated = {k: v for k, v in starts.items() if v >= 3 and _is_strong_start(k)}
    checks.append(_check("repetition", "Style", "Varied opening verbs",
                         "pass" if not repeated else "warn", 1,
                         "No verb opens 3 or more bullets." if not repeated else
                         "Repeated: " + ", ".join(f"'{k}' x{v}" for k, v in repeated.items()) + "."))

    return {"checks": checks, "stats": {"words": n_words, "pages": round(pages, 1), "statements": len(stmts),
                                         "bullet_markers_found": has_markers}}


STATUS_POINTS = {"pass": 1.0, "warn": 0.5, "fail": 0.0}


def category_scores(checks: list[dict], soft_skill_count: int | None) -> dict:
    """Weighted share of points per category, 0-100. Soft Skills = distinct evidenced skills, 5 or more = 100.
    Overall = plain average of the categories that could be computed."""
    out: dict[str, int | None] = {}
    for cat in ("Impact", "Brevity", "Style"):
        cs = [c for c in checks if c["category"] == cat]
        total = sum(c["weight"] for c in cs)
        out[cat] = round(100 * sum(c["weight"] * STATUS_POINTS[c["status"]] for c in cs) / total) if total else None
    out["Soft Skills"] = None if soft_skill_count is None else min(100, round(100 * soft_skill_count / 5))
    vals = [v for v in out.values() if v is not None]
    out["Overall"] = round(sum(vals) / len(vals)) if vals else 0
    return out
