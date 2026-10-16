import io
import sys

from scripts import set_key


def run(monkeypatch, tmp_path, provider, typed, content=None):
    env = tmp_path / ".env"
    if content is not None:
        env.write_text(content, encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", io.StringIO(typed))
    monkeypatch.setattr(sys, "argv", ["set_key", provider, "--env-file", str(env), "--no-check"])
    return set_key.main(), env


def test_replaces_the_active_line_and_keeps_everything_else(monkeypatch, tmp_path):
    original = 'LLM_PROVIDER=openai,openrouter\nOPENAI_API_KEY="sk-old"\nOPENROUTER_API_KEY=\n#OPENROUTER_API_KEY=sk-or-commented\nJWT_SECRET=abc\n'
    code, env = run(monkeypatch, tmp_path, "openai", "sk-newkey1234567890\n", original)
    assert code == 0
    assert env.read_text() == 'LLM_PROVIDER=openai,openrouter\nOPENAI_API_KEY=sk-newkey1234567890\nOPENROUTER_API_KEY=\n#OPENROUTER_API_KEY=sk-or-commented\nJWT_SECRET=abc\n'
    assert (tmp_path / ".env.bak").read_text() == original  # backup of the file as it was


def test_quotes_spaces_and_line_breaks_are_stripped(monkeypatch, tmp_path):
    code, env = run(monkeypatch, tmp_path, "openrouter", '  "sk-or-abc def\n', "X=1\n")
    assert code == 0 and "OPENROUTER_API_KEY=sk-or-abcdef\n" in env.read_text()


def test_duplicate_active_lines_collapse_to_one(monkeypatch, tmp_path):
    code, env = run(monkeypatch, tmp_path, "openai", "sk-fresh\n", "OPENAI_API_KEY=sk-a\nOPENAI_API_KEY=sk-b\n")
    assert env.read_text().count("OPENAI_API_KEY=") == 1 and "sk-fresh" in env.read_text()


def test_a_key_for_the_wrong_provider_is_refused_and_nothing_changes(monkeypatch, tmp_path):
    code, env = run(monkeypatch, tmp_path, "anthropic", "sk-proj-fake\n", "A=1\n")
    assert code == 1 and env.read_text() == "A=1\n" and not (tmp_path / ".env.bak").exists()


def test_empty_input_changes_nothing(monkeypatch, tmp_path):
    code, env = run(monkeypatch, tmp_path, "openai", "   \n", "A=1\n")
    assert code == 1 and env.read_text() == "A=1\n"


def test_creates_the_file_when_missing(monkeypatch, tmp_path):
    code, env = run(monkeypatch, tmp_path, "anthropic", "sk-ant-xyz123\n")
    assert code == 0 and env.read_text() == "ANTHROPIC_API_KEY=sk-ant-xyz123\n"
