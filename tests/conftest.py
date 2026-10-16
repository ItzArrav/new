import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["USAGE_JSONL_PATH"] = f"{_tmp}/raw_usage.jsonl"
os.environ["LLM_PROVIDER"] = "fake"
os.environ["SEED_PASSWORD"] = "test-pass"
os.environ["JWT_SECRET"] = "test-only-secret-" + "x" * 40  # the app refuses placeholder or short secrets
os.environ["SENTIMENT_VOTES"] = "1"  # one call per feedback in tests; the voting tests set their own number
os.environ["AUTO_ROLE_FIT"] = "false"  # keeps scripted-model tests deterministic; the role-fit tests turn it on

import pytest  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import Base, engine  # noqa: E402
from app.seed import seed  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_state():
    Base.metadata.drop_all(engine)
    seed()
    open(settings.usage_jsonl_path, "w").close()
    yield
