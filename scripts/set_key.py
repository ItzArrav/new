"""Put an API key into .env safely, then test it with one tiny call.

    python -m scripts.set_key openai
    python -m scripts.set_key anthropic
    python -m scripts.set_key openrouter

The key is typed (or pasted) at a hidden prompt, so it never appears on screen, in shell history or in chat. The script
removes stray quotes, spaces and line breaks, replaces the existing line for that provider (or adds one), leaves everything
else in .env alone, and makes a backup copy first (.env.bak). Restart the server afterwards: keys are read once at startup.
"""
import argparse
import getpass
import re
import shutil
import sys
from pathlib import Path

NAMES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
PREFIXES = {"openai": ("sk-",), "anthropic": ("sk-ant-",), "openrouter": ("sk-or-",)}


def clean(raw: str) -> str:
    return re.sub(r"\s+", "", raw).strip("\"'")


def write_key(env_path: Path, var: str, value: str) -> None:
    """Replace every ACTIVE `VAR=...` line with one clean line (commented-out lines are left alone)."""
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    out, done = [], False
    for line in lines:
        if re.match(rf"^\s*{var}\s*=", line):
            if not done:
                out.append(f"{var}={value}")
                done = True
            continue  # drop duplicate active lines: the first one used to win silently
        out.append(line)
    if not done:
        out.append(f"{var}={value}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("provider", choices=sorted(NAMES))
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--no-check", action="store_true", help="do not make the test call")
    args = ap.parse_args()
    env_path, var = Path(args.env_file), NAMES[args.provider]

    raw = sys.stdin.readline() if not sys.stdin.isatty() else getpass.getpass(f"Paste your {args.provider} key (hidden), then press Enter: ")
    key = clean(raw)
    if not key:
        print("No key entered. Nothing was changed.")
        return 1
    if not key.startswith(PREFIXES[args.provider]):
        print(f"That does not look like a {args.provider} key (expected it to start with {' or '.join(PREFIXES[args.provider])}). Nothing was changed.")
        return 1

    if env_path.exists():
        shutil.copyfile(env_path, str(env_path) + ".bak")
    write_key(env_path, var, key)
    print(f"Saved {var} to {env_path} ({len(key)} characters, ends with ...{key[-4:]}). Backup: {env_path}.bak")

    if args.no_check:
        return 0
    try:
        from app import llm
        from app.config import settings

        setattr(settings, var.lower(), key)  # test the key just saved, whatever the process loaded before
        reply = llm._build_one(args.provider, single=False).invoke("Say hi")
        print("Live test OK:", str(reply.content)[:50])
    except Exception as e:
        print("Live test FAILED:", type(e).__name__, str(e)[:200])
        print("The key is saved, but the provider rejected it. Check the key and the account's credits.")
        return 2
    print("Now restart the server so it picks the key up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
