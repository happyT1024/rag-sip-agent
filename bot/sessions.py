"""Per-chat session storage (AD-4): the sole owner of chat state.

Keyed by raw ``chat_id: int``. Production store is stdlib sqlite3 with lazy
schema init (repo precedent: ``vectorstore/embed.py``); state file lives under
gitignored ``journal/``. ``MemorySessionStore`` serves tests and dev.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Protocol

from bot.port import HistoryItem

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SESSIONS_PATH = REPO_ROOT / "journal" / "sessions.sqlite3"


class SessionStore(Protocol):
    def get(self, chat_id: int) -> list[HistoryItem]:
        """Return that chat's history (oldest first); empty if unknown."""
        raise NotImplementedError

    def has(self, chat_id: int) -> bool:
        """True if the chat still has a session key."""
        raise NotImplementedError

    def append(self, chat_id: int, item: HistoryItem) -> None:
        """Add one turn to that chat's history, creating it if needed."""
        raise NotImplementedError

    def delete(self, chat_id: int) -> None:
        """Drop that chat's session key only; other chats are untouched."""
        raise NotImplementedError


class SqliteSessionStore:
    """sqlite-backed store; one row per chat, history as a JSON list."""

    def __init__(self, path: Path | str = DEFAULT_SESSIONS_PATH) -> None:
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS history ("
            "chat_id INTEGER PRIMARY KEY, items TEXT NOT NULL)"
        )
        return conn

    def get(self, chat_id: int) -> list[HistoryItem]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT items FROM history WHERE chat_id = ?", (chat_id,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return []
        try:
            entries = json.loads(row[0])
            return [HistoryItem(**entry) for entry in entries]
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            logger.warning(
                "chat_id=%s corrupt session row ignored; history reset", chat_id
            )
            return []

    def has(self, chat_id: int) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM history WHERE chat_id = ?", (chat_id,)
            ).fetchone()
        finally:
            conn.close()
        return row is not None

    def append(self, chat_id: int, item: HistoryItem) -> None:
        items = self.get(chat_id)
        items.append(item)
        payload = json.dumps(
            [{"role": i.role, "text": i.text} for i in items], ensure_ascii=False
        )
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO history (chat_id, items) VALUES (?, ?)",
                (chat_id, payload),
            )
            conn.commit()
        finally:
            conn.close()

    def delete(self, chat_id: int) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM history WHERE chat_id = ?", (chat_id,))
            conn.commit()
        finally:
            conn.close()


class MemorySessionStore:
    """In-memory store with identical semantics — tests and offline dev."""

    def __init__(self) -> None:
        self._chats: dict[int, list[HistoryItem]] = {}

    def get(self, chat_id: int) -> list[HistoryItem]:
        return list(self._chats.get(chat_id, ()))

    def has(self, chat_id: int) -> bool:
        return chat_id in self._chats

    def append(self, chat_id: int, item: HistoryItem) -> None:
        self._chats.setdefault(chat_id, []).append(item)

    def delete(self, chat_id: int) -> None:
        self._chats.pop(chat_id, None)
