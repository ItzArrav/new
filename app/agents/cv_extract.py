"""Turn an uploaded CV file into plain text. Uses pypdf directly (no LangChain wrapper,
works on in-memory bytes). Text from here is untrusted and goes through the same
cleaning, delimiting and evidence checks as any other CV text."""
import io

from pypdf import PdfReader

MAX_PAGES = 15
MIN_TEXT_CHARS = 50
SUPPORTED = (".txt", ".md", ".pdf")


class ExtractError(ValueError):
    """The file cannot be turned into usable text (message is safe to show the user)."""


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="replace")
    if name.endswith(".pdf"):
        return _pdf_text(data)
    raise ExtractError(f"Unsupported file type. Use: {', '.join(SUPPORTED)}")


def _pdf_text(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise ExtractError("PDF is password-protected")
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:MAX_PAGES])
    except ExtractError:
        raise
    except Exception as e:  # pypdf raises several error types on damaged files
        raise ExtractError(f"Could not read PDF ({type(e).__name__})")
    if len(text.strip()) < MIN_TEXT_CHARS:
        raise ExtractError("No extractable text in this PDF (scanned image? it would need OCR)")
    return text
