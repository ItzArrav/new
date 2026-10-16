"""Extraction helpers for candidate contact information (email & name) from CV text."""
import re
from pathlib import Path

# Match standard email formats
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

# Non-name words to filter out if appearing as candidate name
NON_NAME_TOKENS = {
    "resume", "curriculum", "vitae", "cv", "summary", "experience", "education",
    "skills", "profile", "contact", "about", "objective", "phone", "email",
    "page", "address", "professional", "work", "history", "projects"
}


def extract_candidate_email(text: str | None) -> str | None:
    """Finds candidate email in the CV text.
    If a Gmail address is present, it is prioritized; otherwise returns the first detected email.
    """
    if not text:
        return None
    emails = EMAIL_REGEX.findall(text)
    if not emails:
        return None

    # Clean any trailing punctuation (like period or comma at end of line)
    cleaned = [e.rstrip(".,;:)") for e in emails if len(e) > 5]
    if not cleaned:
        return None

    # Prioritize gmail if found (e.g. user asks to reply to their gmail)
    gmails = [e for e in cleaned if "gmail" in e.lower()]
    if gmails:
        return gmails[0]

    return cleaned[0]


def extract_candidate_name(text: str | None, filename: str = "") -> str:
    """Extracts the candidate's full name from the top of the CV text.
    Falls back to a humanized version of the filename if text does not yield a clear name.
    """
    if text:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines[:6]:
            # Skip lines with contact markers, URLs, or long paragraphs
            if "@" in line or "http" in line or "www." in line or len(line) > 40:
                continue
            lower = line.lower()
            if any(token in lower for token in NON_NAME_TOKENS):
                continue
            # Names typically consist of 1 to 4 alphabetical words
            words = line.split()
            if 1 <= len(words) <= 4 and all(re.match(r"^[A-Za-z.'-]+$", w) for w in words):
                return line.title()

    # Fallback to file name stem
    if filename:
        stem = Path(filename).stem
        cleaned = re.sub(r"[_.-]+", " ", stem)
        # Remove numbers and generic tags like cv, resume, frontend, etc.
        cleaned = re.sub(r"\b\d+\b", "", cleaned).strip()
        if cleaned:
            return cleaned.title()

    return "Candidate"
