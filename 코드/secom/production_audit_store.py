"""Persistent, append-only SQLite adapter for the SECOM audit ledger.

The adapter persists only aggregate audit events accepted by ``audit_ledger``.
SQLite is a local durability prototype; it is not presented as a replacement for
production identity, authorization, off-host backup, or immutable retention.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from audit_ledger import LEDGER_COLUMNS, append_event, empty_ledger, verify_ledger


SCHEMA_VERSION = 1
REQUIRED_TRIGGERS = {"audit_events_no_update", "audit_events_no_delete"}


def _connect(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def initialize_store(database_path: str | Path) -> Path:
    """Create an append-only SQLite audit store and return its resolved path."""
    path = Path(database_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = _connect(path)
    try:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS audit_store_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                sequence INTEGER PRIMARY KEY,
                recorded_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                actor TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE
            );
            CREATE TRIGGER IF NOT EXISTS audit_events_no_update
            BEFORE UPDATE ON audit_events
            BEGIN
                SELECT RAISE(ABORT, 'audit_events is append-only');
            END;
            CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
            BEFORE DELETE ON audit_events
            BEGIN
                SELECT RAISE(ABORT, 'audit_events is append-only');
            END;
            """
        )
        connection.execute(
            "INSERT OR IGNORE INTO audit_store_metadata(key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        connection.commit()
    finally:
        connection.close()
    return path


def _read_with_connection(connection: sqlite3.Connection) -> pd.DataFrame:
    rows = connection.execute(
        """
        SELECT sequence, recorded_at, event_type, actor, batch_id, payload_json,
               previous_hash, event_hash
        FROM audit_events
        ORDER BY sequence
        """
    ).fetchall()
    if not rows:
        return empty_ledger()
    return pd.DataFrame([dict(row) for row in rows], columns=LEDGER_COLUMNS)


def read_store(database_path: str | Path) -> pd.DataFrame:
    path = initialize_store(database_path)
    connection = _connect(path)
    try:
        ledger = _read_with_connection(connection)
    finally:
        connection.close()
    verify_ledger(ledger)
    return ledger


def append_store_event(
    database_path: str | Path,
    *,
    event_type: str,
    payload: Mapping[str, Any],
    actor: str = "",
    batch_id: str = "",
    recorded_at: str | pd.Timestamp | None = None,
) -> dict:
    """Atomically append one validated event and return its stored row."""
    path = initialize_store(database_path)
    connection = _connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        current = _read_with_connection(connection)
        updated = append_event(
            current,
            event_type=event_type,
            payload=payload,
            actor=actor,
            batch_id=batch_id,
            recorded_at=recorded_at,
        )
        row = updated.iloc[-1].to_dict()
        connection.execute(
            """
            INSERT INTO audit_events(
                sequence, recorded_at, event_type, actor, batch_id, payload_json,
                previous_hash, event_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(row[column] for column in LEDGER_COLUMNS),
        )
        connection.commit()
        return row
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def verify_store(database_path: str | Path) -> dict:
    """Verify SQLite integrity, append-only controls, and the full hash chain."""
    path = Path(database_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = _connect(path)
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        metadata = {
            row["key"]: row["value"]
            for row in connection.execute(
                "SELECT key, value FROM audit_store_metadata ORDER BY key"
            ).fetchall()
        }
        triggers = {
            str(row["name"])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall()
        }
        ledger = _read_with_connection(connection)
    finally:
        connection.close()
    if integrity.lower() != "ok":
        raise ValueError(f"SQLite 무결성 검사 실패: {integrity}")
    if metadata.get("schema_version") != str(SCHEMA_VERSION):
        raise ValueError("감사 저장소 스키마 버전이 일치하지 않습니다.")
    ledger_result = verify_ledger(ledger)
    missing_triggers = REQUIRED_TRIGGERS - triggers
    if missing_triggers:
        raise ValueError(f"append-only trigger가 누락되었습니다: {sorted(missing_triggers)}")
    return {
        "valid": True,
        "backend": "SQLite",
        "schema_version": SCHEMA_VERSION,
        "sqlite_integrity_check": integrity,
        "append_only_triggers_present": True,
        **ledger_result,
    }
