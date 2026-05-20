"""
SQLite-backed conversation memory for the FPViber chat app.

Tables:
- sessions(id, title, created_at, updated_at)
- messages(id, session_id, role, content, context_json, metadata_json, created_at)

Session isolation
-----------------
Every read and write touching the `messages` table filters explicitly on
`session_id`. The schema enforces a foreign key from `messages.session_id`
to `sessions.id` with `ON DELETE CASCADE`. There is no API on this module
that returns messages without a session_id filter, so it is impossible for
a request scoped to session A to leak data from session B.
"""

import os
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone


# Database path can be overridden via env var (used by Docker to mount a
# volume at /app/chat.db).
DB_PATH = os.environ.get(
    "FPVIBER_DB_PATH",
    os.path.join(os.path.dirname(__file__), "chat.db"),
)

_local = threading.local()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_connection() -> sqlite3.Connection:
    """Return a per-thread sqlite connection."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        _local.conn = conn
    return conn


def init_db() -> None:
    """Create / migrate the tables on first run."""
    conn = get_connection()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT NOT NULL,
                role          TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content       TEXT NOT NULL,
                context_json  TEXT,
                metadata_json TEXT,
                created_at    TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session "
            "ON messages(session_id, id)"
        )

        # Lightweight migration: older databases may not have metadata_json.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        if "metadata_json" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN metadata_json TEXT")


# ==========================================================
# SESSIONS
# ==========================================================

def create_session(title: str = "New conversation") -> dict:
    session_id = uuid.uuid4().hex
    now = _utcnow_iso()
    conn = get_connection()
    with conn:
        conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, title, now, now),
        )
    return {
        "id": session_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
    }


def list_sessions() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, title, created_at, updated_at "
        "FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def get_session(session_id: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    return dict(row) if row else None


def update_session_title(session_id: str, title: str) -> None:
    conn = get_connection()
    with conn:
        conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, _utcnow_iso(), session_id),
        )


def touch_session(session_id: str) -> None:
    conn = get_connection()
    with conn:
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (_utcnow_iso(), session_id),
        )


def delete_session(session_id: str) -> None:
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


# ==========================================================
# MESSAGES
# ==========================================================

def add_message(
    session_id: str,
    role: str,
    content: str,
    context: list | None = None,
    metadata: dict | None = None,
) -> dict:
    if role not in ("user", "assistant"):
        raise ValueError(f"Invalid role: {role}")
    if not session_id:
        raise ValueError("session_id is required")

    now = _utcnow_iso()
    context_json = json.dumps(context) if context is not None else None
    metadata_json = json.dumps(metadata) if metadata is not None else None

    conn = get_connection()
    with conn:
        cur = conn.execute(
            "INSERT INTO messages "
            "(session_id, role, content, context_json, metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, content, context_json, metadata_json, now),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )

    return {
        "id": cur.lastrowid,
        "session_id": session_id,
        "role": role,
        "content": content,
        "context": context,
        "metadata": metadata,
        "created_at": now,
    }


def get_messages(session_id: str) -> list[dict]:
    if not session_id:
        return []
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, session_id, role, content, context_json, metadata_json, "
        "       created_at "
        "FROM messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    ).fetchall()

    messages = []
    for row in rows:
        item = dict(row)
        ctx_raw = item.pop("context_json", None)
        meta_raw = item.pop("metadata_json", None)
        item["context"] = json.loads(ctx_raw) if ctx_raw else None
        item["metadata"] = json.loads(meta_raw) if meta_raw else None
        messages.append(item)
    return messages


def get_history_for_llm(session_id: str, limit: int = 20) -> list[dict]:
    """
    Return the last `limit` messages for a SPECIFIC session, in the simple
    format Gemini expects: [{role: "user"|"assistant", content: "..."}].
    Strictly filtered by session_id.
    """
    if not session_id:
        return []
    conn = get_connection()
    rows = conn.execute(
        "SELECT role, content FROM messages "
        "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
