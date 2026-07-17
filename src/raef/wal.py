"""General write-ahead logging interface for RAEF.

This module defines a backend-agnostic WAL interface with a SQLite
implementation used by the runtime store.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .runtime_store import SQLiteRuntimeStore
from .utils import parse_datetime, require_dict, require_non_empty_str


@dataclass(slots=True)
class WalEvent:
    """One WAL event record."""

    seq_id: int
    run_id: str
    event_type: str
    entity_type: str
    entity_id: str
    payload: dict[str, Any]
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq_id": self.seq_id,
            "run_id": self.run_id,
            "event_type": self.event_type,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "payload": self.payload,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WalEvent":
        return cls(
            seq_id=int(payload.get("seq_id", 0)),
            run_id=require_non_empty_str(payload.get("run_id"), "run_id"),
            event_type=require_non_empty_str(payload.get("event_type"), "event_type"),
            entity_type=require_non_empty_str(payload.get("entity_type"), "entity_type"),
            entity_id=require_non_empty_str(payload.get("entity_id"), "entity_id"),
            payload=require_dict(payload.get("payload"), "payload"),
            created_at=parse_datetime(payload.get("created_at")),
        )


@dataclass(slots=True)
class WalCheckpoint:
    """One WAL checkpoint record."""

    checkpoint_id: str
    run_id: str
    seq_id: int
    snapshot_type: str
    snapshot_payload: dict[str, Any]
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "seq_id": self.seq_id,
            "snapshot_type": self.snapshot_type,
            "snapshot_payload": self.snapshot_payload,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WalCheckpoint":
        return cls(
            checkpoint_id=require_non_empty_str(payload.get("checkpoint_id"), "checkpoint_id"),
            run_id=require_non_empty_str(payload.get("run_id"), "run_id"),
            seq_id=int(payload.get("seq_id", 0)),
            snapshot_type=require_non_empty_str(payload.get("snapshot_type"), "snapshot_type"),
            snapshot_payload=require_dict(payload.get("snapshot_payload"), "snapshot_payload"),
            created_at=parse_datetime(payload.get("created_at")),
        )


class WriteAheadLog(Protocol):
    """Backend-agnostic WAL interface compatible with future DB backends."""

    def init_schema(self) -> None:
        """Initialize storage schema/shape if missing."""

    def append_event(
        self,
        run_id: str,
        event_type: str,
        entity_type: str,
        entity_id: str,
        payload: dict[str, Any],
    ) -> WalEvent:
        """Append one event and return persisted event with sequence id."""

    def append_events(self, events: list[dict[str, Any]]) -> list[WalEvent]:
        """Append multiple events atomically in order."""

    def read_events(
        self,
        run_id: str,
        after_seq: int | None = None,
        limit: int = 1000,
    ) -> list[WalEvent]:
        """Read run events ordered by seq_id."""

    def latest_seq(self, run_id: str) -> int | None:
        """Return latest sequence id for run, or None when no events exist."""

    def write_checkpoint(
        self,
        run_id: str,
        seq_id: int,
        snapshot_type: str,
        snapshot_payload: dict[str, Any],
    ) -> WalCheckpoint:
        """Persist a checkpoint tied to an existing event sequence id."""

    def read_latest_checkpoint(
        self,
        run_id: str,
        snapshot_type: str | None = None,
    ) -> WalCheckpoint | None:
        """Read latest checkpoint for run (optionally filtered by snapshot type)."""

    def has_event(self, run_id: str, event_type: str, entity_id: str) -> bool:
        """Return True when an event exists for run/type/entity combination."""


class SQLiteWriteAheadLog:
    """SQLite implementation of the WriteAheadLog interface."""

    def __init__(
        self,
        db_path: Path | None = None,
        *,
        store: SQLiteRuntimeStore | None = None,
        busy_timeout_ms: int = 5000,
    ) -> None:
        if store is None:
            if db_path is None:
                repo_root = Path(__file__).resolve().parents[2]
                db_path = repo_root / "data" / "raef_runtime" / "runtime.sqlite"
            store = SQLiteRuntimeStore(db_path, busy_timeout_ms=busy_timeout_ms)
        self.store = store

    def init_schema(self) -> None:
        self.store.init_schema()

    def append_event(
        self,
        run_id: str,
        event_type: str,
        entity_type: str,
        entity_id: str,
        payload: dict[str, Any],
    ) -> WalEvent:
        return self.append_events(
            [
                {
                    "run_id": run_id,
                    "event_type": event_type,
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "payload": payload,
                }
            ]
        )[0]

    def append_events(self, events: list[dict[str, Any]]) -> list[WalEvent]:
        return self.store.append_wal_events(events)

    def read_events(
        self,
        run_id: str,
        after_seq: int | None = None,
        limit: int = 1000,
    ) -> list[WalEvent]:
        if after_seq is not None and (not isinstance(after_seq, int) or after_seq < 0):
            raise ValueError("after_seq must be a non-negative integer when provided")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        return self.store.read_wal_events(run_id, after_seq=after_seq, limit=limit)

    def latest_seq(self, run_id: str) -> int | None:
        return self.store.latest_wal_seq(run_id)

    def write_checkpoint(
        self,
        run_id: str,
        seq_id: int,
        snapshot_type: str,
        snapshot_payload: dict[str, Any],
    ) -> WalCheckpoint:
        if not isinstance(seq_id, int) or seq_id <= 0:
            raise ValueError("seq_id must be a positive integer")
        return self.store.write_wal_checkpoint(run_id, seq_id, snapshot_type, snapshot_payload)

    def read_latest_checkpoint(
        self,
        run_id: str,
        snapshot_type: str | None = None,
    ) -> WalCheckpoint | None:
        return self.store.read_latest_wal_checkpoint(run_id, snapshot_type)

    def has_event(self, run_id: str, event_type: str, entity_id: str) -> bool:
        return self.store.has_wal_event(run_id, event_type, entity_id)
