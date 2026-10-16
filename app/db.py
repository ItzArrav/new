import os

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings


class Base(DeclarativeBase):
    pass


_is_sqlite = settings.database_url.startswith("sqlite")

if _is_sqlite:
    _path = settings.database_url.replace("sqlite:///", "", 1)
    os.makedirs(os.path.dirname(os.path.abspath(_path)), exist_ok=True)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False, "timeout": 30} if _is_sqlite else {},
)

if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")  # concurrent readers + one writer
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from . import models  # noqa: F401  (register tables)

    Base.metadata.create_all(engine)
    _add_missing_columns()


LATER_COLUMNS = {
    "recommendations": [("cv_text", "TEXT"), ("quality_json", "TEXT"), ("feedback_json", "TEXT"), ("role_fit_json", "TEXT")],
    "sentiment_results": [("source", "VARCHAR(16) NOT NULL DEFAULT 'user'")],
}


def _add_missing_columns() -> None:
    """create_all() never alters an existing table, so add columns that were introduced later."""
    from sqlalchemy import inspect, text
    from sqlalchemy.exc import OperationalError, ProgrammingError

    for table, columns in LATER_COLUMNS.items():
        existing = {c["name"] for c in inspect(engine).get_columns(table)}
        for column, ddl in columns:
            if column in existing:
                continue
            try:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
            except (OperationalError, ProgrammingError) as e:
                # two workers starting together can both try; "already exists" means the other one won
                if "duplicate column" not in str(e).lower() and "already exists" not in str(e).lower():
                    raise
                continue
            if (table, column) == ("sentiment_results", "source"):
                _label_ai_sample_sentiment()


def _label_ai_sample_sentiment() -> None:
    """Older rows have no source. Mark those whose text is a CV's stored AI feedback as "ai_sample".
    Best effort: AI feedback that was regenerated later is no longer stored, so its rows stay "user"."""
    import json

    from sqlalchemy import text

    with engine.begin() as conn:
        texts = {json.loads(f).get("feedback") for (f,) in
                 conn.execute(text("SELECT feedback_json FROM recommendations WHERE feedback_json IS NOT NULL"))}
        for t in texts - {None}:
            conn.execute(text("UPDATE sentiment_results SET source = 'ai_sample' WHERE text = :t"), {"t": t})
