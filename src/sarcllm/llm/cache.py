"""SQLite request cache + usage ledger — Dev Plan v2 §6 (llm/cache.py).

Cache key = SHA-256 of (model, messages, params) after NFC normalisation, so the same
request is paid for once, and a changed prompt always gives a new key.
The ledger records every *paid* (non-cached) call: it drives the budget cap and the
daily-limit guard, and it is the source of the cost figures in the paper appendix.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
import unicodedata
from pathlib import Path


def _nfc(obj):
    if isinstance(obj, str):
        return unicodedata.normalize("NFC", obj)
    if isinstance(obj, list):
        return [_nfc(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _nfc(v) for k, v in obj.items()}
    return obj


def request_key(model: str, messages: list[dict], params: dict) -> str:
    payload = {"model": model, "messages": _nfc(messages), "params": params}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def utc_day() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


class Store:
    """One SQLite file holding the response cache and the usage ledger."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY, record TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS ledger (
                ts TEXT NOT NULL, day TEXT NOT NULL, model TEXT, tag TEXT,
                tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL, key TEXT);
            CREATE INDEX IF NOT EXISTS ledger_day ON ledger(day);
        """)
        self.db.commit()

    # --- cache ---
    def get(self, key: str) -> dict | None:
        row = self.db.execute("SELECT record FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, record: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)",
                        (key, json.dumps(record, ensure_ascii=False), dt.datetime.now(dt.timezone.utc).isoformat()))
        self.db.commit()

    def n_cached(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM cache").fetchone()[0]

    # --- ledger ---
    def log_usage(self, model: str, tag: str, tokens_in: int, tokens_out: int, cost: float, key: str) -> None:
        now = dt.datetime.now(dt.timezone.utc)
        self.db.execute("INSERT INTO ledger VALUES (?,?,?,?,?,?,?,?)",
                        (now.isoformat(), now.strftime("%Y-%m-%d"), model, tag, tokens_in, tokens_out, cost, key))
        self.db.commit()

    def totals(self, day: str | None = None) -> dict:
        q = "SELECT COUNT(*), COALESCE(SUM(tokens_in),0), COALESCE(SUM(tokens_out),0), COALESCE(SUM(cost_usd),0) FROM ledger"
        row = self.db.execute(q + (" WHERE day=?" if day else ""), ((day,) if day else ())).fetchone()
        return {"requests": row[0], "tokens_in": row[1], "tokens_out": row[2], "cost_usd": row[3]}

    def by_tag(self):
        return self.db.execute(
            "SELECT tag, COUNT(*), SUM(tokens_in), SUM(tokens_out), SUM(cost_usd) FROM ledger GROUP BY tag"
        ).fetchall()