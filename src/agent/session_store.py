"""
Session Store
=============
Stores per-user conversation state so the agent can maintain context across
multiple API calls within a single session (e.g. navigating procedure steps,
remembering company type, holding last retrieved chunks, and persisting
multi-turn chat history).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from src.agent.intent_classifier import IntentType


@dataclass
class SessionMessage:
    """A single user or assistant turn in a conversation."""

    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    citations: list[dict[str, Any]] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)


def _auto_title(text: str, max_len: int = 48) -> str:
    """Build a short conversation title from the first user message (0 LLM calls)."""
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return "New Chat"
    if len(cleaned) <= max_len:
        return cleaned
    trimmed = cleaned[:max_len].rsplit(" ", 1)[0].strip()
    return f"{trimmed or cleaned[:max_len].strip()}…"


@dataclass
class SessionState:
    """Mutable state for a single user conversation session."""

    session_id: str
    title: str = ""
    messages: list[SessionMessage] = field(default_factory=list)
    intent_history: list[IntentType] = field(default_factory=list)
    active_process: str | None = None
    current_step: int = 0
    total_steps: int = 0
    confirmed_answers: dict[str, str] = field(default_factory=dict)
    last_chunks: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    _MAX_HISTORY: int = field(default=20, init=False, repr=False, compare=False)

    def push_intent(self, intent: IntentType) -> None:
        """Append *intent* to history, keeping at most ``_MAX_HISTORY`` items."""
        self.intent_history.append(intent)
        if len(self.intent_history) > self._MAX_HISTORY:
            self.intent_history = self.intent_history[-self._MAX_HISTORY :]
        self.updated_at = datetime.now(tz=timezone.utc)

    def add_message(
        self,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        steps: list[str] | None = None,
    ) -> SessionMessage:
        """Append a message turn and auto-title the conversation on the first user turn."""
        if role == "user" and not self.title:
            self.title = _auto_title(content)
        msg = SessionMessage(
            role=role,
            content=content,
            citations=list(citations or []),
            steps=list(steps or []),
        )
        self.messages.append(msg)
        self.updated_at = datetime.now(tz=timezone.utc)
        return msg

    def replace_last_assistant_message(
        self,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        steps: list[str] | None = None,
    ) -> SessionMessage:
        """Replace the latest assistant reply (used on regenerate) or append if none."""
        msg = SessionMessage(
            role="assistant",
            content=content,
            citations=list(citations or []),
            steps=list(steps or []),
        )
        if self.messages and self.messages[-1].role == "assistant":
            self.messages[-1] = msg
        else:
            self.messages.append(msg)
        self.updated_at = datetime.now(tz=timezone.utc)
        return msg

    @property
    def has_active_process(self) -> bool:
        """``True`` when the user is mid-procedure."""
        return self.active_process is not None

    @property
    def procedure_complete(self) -> bool:
        """``True`` when all steps have been walked through."""
        return (
            self.has_active_process
            and self.total_steps > 0
            and self.current_step > self.total_steps
        )


class SessionStore(Protocol):
    """Interface that every session store backend must satisfy."""

    def get_or_create(self, session_id: str) -> SessionState:
        """Return the existing session or create a new one."""
        ...

    def get(self, session_id: str) -> SessionState | None:
        """Return the existing session if present, else ``None``."""
        ...

    def save(self, session: SessionState) -> None:
        """Persist the updated session state."""
        ...

    def delete(self, session_id: str) -> bool:
        """Remove a session and return ``True`` if it existed."""
        ...

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """List conversation summaries ordered by most recently updated."""
        ...

    def rename(self, session_id: str, title: str) -> bool:
        """Rename a conversation and return ``True`` if it existed."""
        ...


class InMemorySessionStore:
    """Thread-safe in-memory session store for tests and ephemeral deployments."""

    def __init__(self, max_sessions: int = 1000) -> None:
        self._store: dict[str, SessionState] = {}
        self._max = max_sessions
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str = "") -> SessionState:
        """Return the session for *session_id*, or create a new one."""
        if not session_id:
            session_id = str(uuid.uuid4())

        with self._lock:
            if session_id not in self._store:
                if len(self._store) >= self._max:
                    oldest = min(self._store.values(), key=lambda s: s.created_at)
                    del self._store[oldest.session_id]

                self._store[session_id] = SessionState(session_id=session_id)

            return self._store[session_id]

    def get(self, session_id: str) -> SessionState | None:
        """Return the existing session if present, else ``None``."""
        with self._lock:
            return self._store.get(session_id)

    def save(self, session: SessionState) -> None:
        """Upsert *session* into the store."""
        with self._lock:
            session.updated_at = datetime.now(tz=timezone.utc)
            self._store[session.session_id] = session

    def delete(self, session_id: str) -> bool:
        """Remove *session_id* if it exists."""
        with self._lock:
            return self._store.pop(session_id, None) is not None

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return conversation summaries ordered newest-first."""
        safe_limit = max(1, min(int(limit), 200))
        with self._lock:
            ordered = sorted(self._store.values(), key=lambda s: s.updated_at, reverse=True)
            return [
                {
                    "session_id": s.session_id,
                    "title": s.title or "New Chat",
                    "created_at": s.created_at.isoformat(),
                    "updated_at": s.updated_at.isoformat(),
                    "message_count": len(s.messages),
                }
                for s in ordered[:safe_limit]
            ]

    def rename(self, session_id: str, title: str) -> bool:
        """Rename *session_id* if it exists."""
        with self._lock:
            session = self._store.get(session_id)
            if session is None:
                return False
            session.title = title.strip()
            session.updated_at = datetime.now(tz=timezone.utc)
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


class SQLiteSessionStore:
    """Zero-dependency SQLite-backed session and conversation history store."""

    def __init__(self, db_path: str = "data/chat_history.db") -> None:
        if db_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(db_path))
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '',
                    intent_history TEXT NOT NULL DEFAULT '[]',
                    active_process TEXT,
                    current_step INTEGER NOT NULL DEFAULT 0,
                    total_steps INTEGER NOT NULL DEFAULT 0,
                    confirmed_answers TEXT NOT NULL DEFAULT '{}',
                    last_chunks TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    citations TEXT NOT NULL DEFAULT '[]',
                    steps TEXT NOT NULL DEFAULT '[]',
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)"
            )

    def _row_to_session(self, row: sqlite3.Row, msg_rows: list[sqlite3.Row]) -> SessionState:
        intents: list[IntentType] = []
        for raw_intent in json.loads(row["intent_history"] or "[]"):
            try:
                intents.append(IntentType(raw_intent))
            except ValueError:
                continue

        messages = [
            SessionMessage(
                role=m["role"],
                content=m["content"],
                timestamp=m["timestamp"],
                citations=json.loads(m["citations"] or "[]"),
                steps=json.loads(m["steps"] or "[]"),
            )
            for m in msg_rows
        ]
        return SessionState(
            session_id=row["session_id"],
            title=row["title"] or "",
            messages=messages,
            intent_history=intents,
            active_process=row["active_process"],
            current_step=int(row["current_step"]),
            total_steps=int(row["total_steps"]),
            confirmed_answers=json.loads(row["confirmed_answers"] or "{}"),
            last_chunks=json.loads(row["last_chunks"] or "[]"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def get(self, session_id: str) -> SessionState | None:
        """Return the existing session if present, else ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None
            msg_rows = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,)
            ).fetchall()
            return self._row_to_session(row, msg_rows)

    def get_or_create(self, session_id: str = "") -> SessionState:
        """Return the session for *session_id*, or create a new one."""
        sid = session_id or str(uuid.uuid4())
        existing = self.get(sid)
        if existing is not None:
            return existing
        session = SessionState(session_id=sid)
        self.save(session)
        return session

    def save(self, session: SessionState) -> None:
        """Upsert *session* and its ordered messages into SQLite."""
        session.updated_at = datetime.now(tz=timezone.utc)
        intent_json = json.dumps([i.value for i in session.intent_history])
        answers_json = json.dumps(session.confirmed_answers, ensure_ascii=False)
        chunks_json = json.dumps(session.last_chunks, ensure_ascii=False)

        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO sessions (
                    session_id, title, intent_history, active_process,
                    current_step, total_steps, confirmed_answers, last_chunks,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    title = excluded.title,
                    intent_history = excluded.intent_history,
                    active_process = excluded.active_process,
                    current_step = excluded.current_step,
                    total_steps = excluded.total_steps,
                    confirmed_answers = excluded.confirmed_answers,
                    last_chunks = excluded.last_chunks,
                    updated_at = excluded.updated_at
                """,
                (
                    session.session_id,
                    session.title,
                    intent_json,
                    session.active_process,
                    session.current_step,
                    session.total_steps,
                    answers_json,
                    chunks_json,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                ),
            )
            self._conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session.session_id,)
            )
            self._conn.executemany(
                """
                INSERT INTO messages (session_id, role, content, timestamp, citations, steps)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        session.session_id,
                        m.role,
                        m.content,
                        m.timestamp,
                        json.dumps(m.citations, ensure_ascii=False),
                        json.dumps(m.steps, ensure_ascii=False),
                    )
                    for m in session.messages
                ],
            )

    def delete(self, session_id: str) -> bool:
        """Remove *session_id* and its messages; return ``True`` if it existed."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            return cur.rowcount > 0

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """List conversation summaries ordered by most recently updated."""
        safe_limit = max(1, min(int(limit), 200))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT s.session_id, s.title, s.created_at, s.updated_at,
                       COUNT(m.id) AS message_count
                FROM sessions s
                LEFT JOIN messages m ON m.session_id = s.session_id
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
            return [
                {
                    "session_id": r["session_id"],
                    "title": r["title"] or "New Chat",
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "message_count": int(r["message_count"]),
                }
                for r in rows
            ]

    def rename(self, session_id: str, title: str) -> bool:
        """Rename *session_id* persistently; return ``True`` if found."""
        cleaned = title.strip()
        if not cleaned:
            return False
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (cleaned, now, session_id),
            )
            return cur.rowcount > 0

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
            return int(row[0]) if row else 0

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()
