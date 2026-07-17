"""Shared SQLite runtime store for RAEF.

This module is the single canonical persistence layer for:
- planner projections,
- agent/context projections,
- external/tool result cache,
- WAL events and checkpoints.

The JSON mirrors used elsewhere are soft observability writes only.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator
from uuid import uuid4

from .utils import json_sha256, json_size_bytes, parse_datetime, require_dict, require_non_empty_str, utc_now


class SQLiteRuntimeStore:
    """Canonical SQLite-backed runtime store shared by all services."""

    def __init__(
        self,
        db_path: Path,
        *,
        busy_timeout_ms: int = 5000,
        payload_inline_limit_bytes: int = 16_384,
    ) -> None:
        self.db_path = db_path
        self.busy_timeout_ms = busy_timeout_ms
        self.payload_inline_limit_bytes = payload_inline_limit_bytes
        self._schema_lock = threading.Lock()
        self._local = threading.local()
        self.init_schema()

    def init_schema(self) -> None:
        with self._schema_lock:
            #  (run_id, plan_item_id) as primary key
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS planner_states (
                        run_id TEXT PRIMARY KEY,
                        version INTEGER NOT NULL,
                        plan_source_text TEXT NOT NULL,
                        plan_schema_json TEXT NOT NULL,
                        active_item_id TEXT,
                        cursor_index INTEGER,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS planner_items (
                        run_id TEXT NOT NULL,
                        plan_item_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT NOT NULL,
                        status TEXT NOT NULL,
                        sequence_index INTEGER NOT NULL,
                        depends_on_json TEXT NOT NULL,
                        acceptance_criteria_json TEXT NOT NULL,
                        llm_output TEXT,
                        llm_output_history_json TEXT NOT NULL,
                        tool_call_refs_json TEXT NOT NULL,
                        notes_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (run_id, plan_item_id),
                        FOREIGN KEY (run_id) REFERENCES planner_states(run_id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_planner_items_run_sequence
                    ON planner_items(run_id, sequence_index);

                    CREATE TABLE IF NOT EXISTS agent_contexts (
                        run_id TEXT PRIMARY KEY,
                        turn_index INTEGER NOT NULL,
                        step_index INTEGER NOT NULL,
                        planner_version INTEGER NOT NULL,
                        memory_json TEXT NOT NULL,
                        pending_execution_id TEXT,
                        pending_execution_ids_json TEXT NOT NULL,
                        last_checkpoint_seq INTEGER,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS agent_messages (
                        run_id TEXT NOT NULL,
                        message_index INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        name TEXT,
                        tool_call_id TEXT,
                        metadata_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (run_id, message_index),
                        FOREIGN KEY (run_id) REFERENCES agent_contexts(run_id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_agent_messages_run_index
                    ON agent_messages(run_id, message_index);

                    CREATE TABLE IF NOT EXISTS payload_artifacts (
                        payload_id TEXT PRIMARY KEY,
                        payload_json TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        sha256 TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS external_results (
                        execution_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        plan_item_id TEXT,
                        tool_name TEXT NOT NULL,
                        operation_type TEXT NOT NULL,
                        request_payload_json TEXT,
                        response_payload_json TEXT,
                        request_payload_ref_id TEXT,
                        response_payload_ref_id TEXT,
                        result_status TEXT NOT NULL,
                        execution_status TEXT NOT NULL,
                        error_message TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (run_id) REFERENCES agent_contexts(run_id) ON DELETE CASCADE,
                        FOREIGN KEY (request_payload_ref_id) REFERENCES payload_artifacts(payload_id),
                        FOREIGN KEY (response_payload_ref_id) REFERENCES payload_artifacts(payload_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_external_results_run_created
                    ON external_results(run_id, created_at, execution_id);

                    CREATE TABLE IF NOT EXISTS wal_events (
                        seq_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_wal_events_run_seq
                    ON wal_events(run_id, seq_id);

                    CREATE INDEX IF NOT EXISTS idx_wal_events_lookup
                    ON wal_events(run_id, event_type, entity_id);

                    CREATE TABLE IF NOT EXISTS wal_checkpoints (
                        checkpoint_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        seq_id INTEGER NOT NULL,
                        snapshot_type TEXT NOT NULL,
                        snapshot_payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_wal_checkpoints_run_type
                    ON wal_checkpoints(run_id, snapshot_type, seq_id);

                    CREATE TABLE IF NOT EXISTS evaluation_spans (
                        span_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        step_index INTEGER,
                        plan_item_id TEXT,
                        attempt_no INTEGER NOT NULL,
                        phase TEXT NOT NULL,
                        parent_span_id TEXT,
                        status TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        finished_at TEXT,
                        duration_ms REAL,
                        metadata_json TEXT NOT NULL,
                        error_message TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_evaluation_spans_run_step_attempt
                    ON evaluation_spans(run_id, step_index, plan_item_id, attempt_no, started_at);

                    CREATE INDEX IF NOT EXISTS idx_evaluation_spans_parent
                    ON evaluation_spans(parent_span_id, started_at);
                    """
                )

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            self._local.depth += 1
            try:
                yield conn
            finally:
                self._local.depth -= 1
            return

        conn = self._connect()
        self._local.conn = conn
        self._local.depth = 1
        try:
            conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            self._local.depth = 0
            self._local.conn = None
            conn.close()

    def load_planner_state(self, run_id: str):
        from .planner_state import PlannerItem, PlannerItemStatus, PlannerState

        run_id = require_non_empty_str(run_id, "run_id")
        conn, owned = self._borrow_connection()
        try:
            row = conn.execute(
                """
                SELECT run_id, version, plan_source_text, plan_schema_json, active_item_id, cursor_index, updated_at
                FROM planner_states WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            item_rows = conn.execute(
                """
                SELECT plan_item_id, title, description, status, sequence_index, depends_on_json,
                       acceptance_criteria_json, llm_output, llm_output_history_json,
                       tool_call_refs_json, notes_json, updated_at
                FROM planner_items
                WHERE run_id = ?
                ORDER BY sequence_index ASC, plan_item_id ASC
                """,
                (run_id,),
            ).fetchall()
        finally:
            if owned:
                conn.close()

        items = [
            PlannerItem(
                plan_item_id=str(item_row[0]),
                title=str(item_row[1]),
                description=str(item_row[2]),
                status=PlannerItemStatus(str(item_row[3])),
                sequence_index=int(item_row[4]),
                depends_on=_json_loads(str(item_row[5]), default=[]),
                acceptance_criteria=_json_loads(str(item_row[6]), default=[]),
                llm_output=item_row[7],
                llm_output_history=_json_loads(str(item_row[8]), default=[]),
                tool_call_refs=_json_loads(str(item_row[9]), default=[]),
                notes=_json_loads(str(item_row[10]), default=[]),
                updated_at=parse_datetime(item_row[11]),
            )
            for item_row in item_rows
        ]
        return PlannerState(
            run_id=str(row[0]),
            version=int(row[1]),
            plan_source_text=str(row[2]),
            items=items,
            plan_schema=_json_loads(str(row[3]), default={}),
            active_item_id=row[4],
            cursor_index=row[5],
            updated_at=parse_datetime(row[6]),
        )

    def save_planner_state(self, state) -> None:
        payload = state.to_dict()
        with self.transaction(immediate=True) as conn:
            conn.execute(
                """
                INSERT INTO planner_states(run_id, version, plan_source_text, plan_schema_json, active_item_id, cursor_index, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    version = excluded.version,
                    plan_source_text = excluded.plan_source_text,
                    plan_schema_json = excluded.plan_schema_json,
                    active_item_id = excluded.active_item_id,
                    cursor_index = excluded.cursor_index,
                    updated_at = excluded.updated_at
                """,
                (
                    payload["run_id"],
                    payload["version"],
                    payload["plan_source_text"],
                    _json_dumps(payload["plan_schema"]),
                    payload["active_item_id"],
                    payload["cursor_index"],
                    payload["updated_at"],
                ),
            )
            conn.execute("DELETE FROM planner_items WHERE run_id = ?", (payload["run_id"],))
            for item in payload["items"]:
                conn.execute(
                    """
                    INSERT INTO planner_items(
                        run_id, plan_item_id, title, description, status, sequence_index,
                        depends_on_json, acceptance_criteria_json, llm_output,
                        llm_output_history_json, tool_call_refs_json, notes_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        payload["run_id"],
                        item["plan_item_id"],
                        item["title"],
                        item["description"],
                        item["status"],
                        item["sequence_index"],
                        _json_dumps(item["depends_on"]),
                        _json_dumps(item["acceptance_criteria"]),
                        item["llm_output"],
                        _json_dumps(item["llm_output_history"]),
                        _json_dumps(item["tool_call_refs"]),
                        _json_dumps(item["notes"]),
                        item["updated_at"],
                    ),
                )

    def load_context_snapshot(self, run_id: str):
        from .agent_state import AgentContextSnapshot, AgentMessage

        run_id = require_non_empty_str(run_id, "run_id")
        conn, owned = self._borrow_connection()
        try:
            row = conn.execute(
                """
                SELECT run_id, turn_index, step_index, planner_version, memory_json,
                       pending_execution_id, pending_execution_ids_json, last_checkpoint_seq, updated_at
                FROM agent_contexts WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                return None
            message_rows = conn.execute(
                """
                SELECT role, content, name, tool_call_id, metadata_json, created_at
                FROM agent_messages
                WHERE run_id = ?
                ORDER BY message_index ASC
                """,
                (run_id,),
            ).fetchall()
        finally:
            if owned:
                conn.close()

        messages = [
            AgentMessage(
                role=str(message_row[0]),
                content=str(message_row[1]),
                name=message_row[2],
                tool_call_id=message_row[3],
                metadata=_json_loads(str(message_row[4]), default={}),
                created_at=parse_datetime(message_row[5]),
            )
            for message_row in message_rows
        ]
        return AgentContextSnapshot(
            run_id=str(row[0]),
            turn_index=int(row[1]),
            step_index=int(row[2]),
            planner_version=int(row[3]),
            messages=messages,
            memory=_json_loads(str(row[4]), default={}),
            pending_execution_id=row[5],
            pending_execution_ids=_json_loads(str(row[6]), default=[]),
            last_checkpoint_seq=row[7],
            updated_at=parse_datetime(row[8]),
        )

    def save_context_snapshot(self, snapshot) -> None:
        payload = snapshot.to_dict()
        with self.transaction(immediate=True) as conn:
            conn.execute(
                """
                INSERT INTO agent_contexts(
                    run_id, turn_index, step_index, planner_version, memory_json,
                    pending_execution_id, pending_execution_ids_json, last_checkpoint_seq, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    turn_index = excluded.turn_index,
                    step_index = excluded.step_index,
                    planner_version = excluded.planner_version,
                    memory_json = excluded.memory_json,
                    pending_execution_id = excluded.pending_execution_id,
                    pending_execution_ids_json = excluded.pending_execution_ids_json,
                    last_checkpoint_seq = excluded.last_checkpoint_seq,
                    updated_at = excluded.updated_at
                """,
                (
                    payload["run_id"],
                    payload["turn_index"],
                    payload["step_index"],
                    payload["planner_version"],
                    _json_dumps(payload["memory"]),
                    payload["pending_execution_id"],
                    _json_dumps(payload["pending_execution_ids"]),
                    payload["last_checkpoint_seq"],
                    payload["updated_at"],
                ),
            )
            # delete on save
            conn.execute("DELETE FROM agent_messages WHERE run_id = ?", (payload["run_id"],))
            for index, message in enumerate(payload["messages"]):
                conn.execute(
                    """
                    INSERT INTO agent_messages(
                        run_id, message_index, role, content, name, tool_call_id, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        payload["run_id"],
                        index,
                        message["role"],
                        message["content"],
                        message["name"],
                        message["tool_call_id"],
                        _json_dumps(message["metadata"]),
                        message["created_at"],
                    ),
                )

    def upsert_external_result(self, result):
        from .agent_state import ExternalResultRecord

        with self.transaction(immediate=True) as conn:
            row = conn.execute(
                "SELECT created_at FROM external_results WHERE execution_id = ?",
                (result.execution_id,),
            ).fetchone()
            created_at = parse_datetime(row[0]) if row is not None else utc_now()
            request_json, request_ref = self._store_payload(
                conn,
                execution_id=result.execution_id,
                kind="request",
                payload=result.request_payload,
            )
            response_json, response_ref = self._store_payload(
                conn,
                execution_id=result.execution_id,
                kind="response",
                payload=result.response_payload,
            )
            conn.execute(
                """
                INSERT INTO external_results(
                    execution_id, run_id, plan_item_id, tool_name, operation_type,
                    request_payload_json, response_payload_json, request_payload_ref_id, response_payload_ref_id,
                    result_status, execution_status, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(execution_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    plan_item_id = excluded.plan_item_id,
                    tool_name = excluded.tool_name,
                    operation_type = excluded.operation_type,
                    request_payload_json = excluded.request_payload_json,
                    response_payload_json = excluded.response_payload_json,
                    request_payload_ref_id = excluded.request_payload_ref_id,
                    response_payload_ref_id = excluded.response_payload_ref_id,
                    result_status = excluded.result_status,
                    execution_status = excluded.execution_status,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """,
                (
                    result.execution_id,
                    result.run_id,
                    result.plan_item_id,
                    result.tool_name,
                    result.operation_type,
                    request_json,
                    response_json,
                    request_ref["payload_id"] if request_ref is not None else None,
                    response_ref["payload_id"] if response_ref is not None else None,
                    result.result_status.value,
                    result.execution_status.value,
                    result.error_message,
                    created_at.isoformat(),
                    result.updated_at.isoformat(),
                ),
            )

        return ExternalResultRecord(
            execution_id=result.execution_id,
            run_id=result.run_id,
            plan_item_id=result.plan_item_id,
            tool_name=result.tool_name,
            operation_type=result.operation_type,
            request_payload=result.request_payload,
            response_payload=result.response_payload,
            request_payload_ref=request_ref,
            response_payload_ref=response_ref,
            result_status=result.result_status,
            execution_status=result.execution_status,
            error_message=result.error_message,
            created_at=created_at,
            updated_at=result.updated_at,
        )

    def get_external_result(self, execution_id: str):
        from .agent_state import ExternalResultRecord, ExternalResultStatus
        from .models import normalize_execution_status

        execution_id = require_non_empty_str(execution_id, "execution_id")
        conn, owned = self._borrow_connection()
        try:
            row = conn.execute(
                """
                SELECT execution_id, run_id, plan_item_id, tool_name, operation_type,
                       request_payload_json, response_payload_json, request_payload_ref_id, response_payload_ref_id,
                       result_status, execution_status, error_message, created_at, updated_at
                FROM external_results
                WHERE execution_id = ?
                """,
                (execution_id,),
            ).fetchone()
            if row is None:
                return None
            request_payload, request_ref = self._load_payload(row[5], row[7], conn)
            response_payload, response_ref = self._load_payload(row[6], row[8], conn)
        finally:
            if owned:
                conn.close()

        return ExternalResultRecord(
            execution_id=str(row[0]),
            run_id=str(row[1]),
            plan_item_id=row[2],
            tool_name=str(row[3]),
            operation_type=str(row[4]),
            request_payload=request_payload,
            response_payload=response_payload,
            request_payload_ref=request_ref,
            response_payload_ref=response_ref,
            result_status=ExternalResultStatus(str(row[9])),
            execution_status=normalize_execution_status(str(row[10])),
            error_message=row[11],
            created_at=parse_datetime(row[12]),
            updated_at=parse_datetime(row[13]),
        )

    def list_external_results(self, run_id: str):
        run_id = require_non_empty_str(run_id, "run_id")
        conn, owned = self._borrow_connection()
        try:
            execution_ids = [
                str(row[0])
                for row in conn.execute(
                    "SELECT execution_id FROM external_results WHERE run_id = ? ORDER BY created_at ASC, execution_id ASC",
                    (run_id,),
                ).fetchall()
            ]
        finally:
            if owned:
                conn.close()
        return [record for execution_id in execution_ids if (record := self.get_external_result(execution_id)) is not None]

    def append_wal_events(self, events: list[dict[str, Any]]):
        from .wal import WalEvent

        if not isinstance(events, list) or not events:
            raise ValueError("events must be a non-empty list")

        created: list[WalEvent] = []
        with self.transaction(immediate=True) as conn:
            for raw in events:
                if not isinstance(raw, dict):
                    raise ValueError("each event input must be an object")
                now = utc_now()
                cur = conn.execute(
                    """
                    INSERT INTO wal_events(run_id, event_type, entity_type, entity_id, payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        require_non_empty_str(raw.get("run_id"), "run_id"),
                        require_non_empty_str(raw.get("event_type"), "event_type"),
                        require_non_empty_str(raw.get("entity_type"), "entity_type"),
                        require_non_empty_str(raw.get("entity_id"), "entity_id"),
                        _json_dumps(require_dict(raw.get("payload"), "payload")),
                        now.isoformat(),
                    ),
                )
                if cur.lastrowid is None:
                    raise RuntimeError("wal_events insert did not produce a sequence id")
                created.append(
                    WalEvent(
                        seq_id=int(cur.lastrowid),
                        run_id=str(raw["run_id"]),
                        event_type=str(raw["event_type"]),
                        entity_type=str(raw["entity_type"]),
                        entity_id=str(raw["entity_id"]),
                        payload=dict(raw["payload"]),
                        created_at=now,
                    )
                )
        return created

    def read_wal_events(self, run_id: str, *, after_seq: int | None = None, limit: int = 1000):
        from .wal import WalEvent

        run_id = require_non_empty_str(run_id, "run_id")
        conn, owned = self._borrow_connection()
        try:
            sql = (
                "SELECT seq_id, run_id, event_type, entity_type, entity_id, payload_json, created_at "
                "FROM wal_events WHERE run_id = ?"
            )
            params: list[Any] = [run_id]
            if after_seq is not None:
                sql += " AND seq_id > ?"
                params.append(after_seq)
            sql += " ORDER BY seq_id ASC LIMIT ?"
            params.append(limit)
            rows = conn.execute(sql, tuple(params)).fetchall()
        finally:
            if owned:
                conn.close()

        return [
            WalEvent(
                seq_id=int(row[0]),
                run_id=str(row[1]),
                event_type=str(row[2]),
                entity_type=str(row[3]),
                entity_id=str(row[4]),
                payload=_json_loads(str(row[5]), default={}),
                created_at=parse_datetime(row[6]),
            )
            for row in rows
        ]

    def list_run_ids(self) -> list[str]:
        """Return all run ids known to the store, newest WAL activity first."""

        conn, owned = self._borrow_connection()
        try:
            rows = conn.execute(
                """
                SELECT run_id, MAX(seq_id) AS latest_seq
                FROM wal_events
                GROUP BY run_id
                ORDER BY latest_seq DESC
                """
            ).fetchall()
        finally:
            if owned:
                conn.close()
        return [str(row[0]) for row in rows]

    def latest_wal_seq(self, run_id: str) -> int | None:
        run_id = require_non_empty_str(run_id, "run_id")
        conn, owned = self._borrow_connection()
        try:
            row = conn.execute(
                "SELECT MAX(seq_id) FROM wal_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        finally:
            if owned:
                conn.close()
        if row is None or row[0] is None:
            return None
        return int(row[0])

    def has_wal_event(self, run_id: str, event_type: str, entity_id: str) -> bool:
        run_id = require_non_empty_str(run_id, "run_id")
        event_type = require_non_empty_str(event_type, "event_type")
        entity_id = require_non_empty_str(entity_id, "entity_id")
        conn, owned = self._borrow_connection()
        try:
            row = conn.execute(
                "SELECT 1 FROM wal_events WHERE run_id = ? AND event_type = ? AND entity_id = ? LIMIT 1",
                (run_id, event_type, entity_id),
            ).fetchone()
        finally:
            if owned:
                conn.close()
        return row is not None

    def write_wal_checkpoint(self, run_id: str, seq_id: int, snapshot_type: str, snapshot_payload: dict[str, Any]):
        from .wal import WalCheckpoint

        run_id = require_non_empty_str(run_id, "run_id")
        snapshot_type = require_non_empty_str(snapshot_type, "snapshot_type")
        snapshot_payload = require_dict(snapshot_payload, "snapshot_payload")
        checkpoint_id = str(uuid4())
        created_at = utc_now()

        with self.transaction(immediate=True) as conn:
            exists = conn.execute(
                "SELECT 1 FROM wal_events WHERE run_id = ? AND seq_id = ?",
                (run_id, seq_id),
            ).fetchone()
            if exists is None:
                raise ValueError("checkpoint seq_id must reference an existing event in the same run")
            conn.execute(
                """
                INSERT INTO wal_checkpoints(checkpoint_id, run_id, seq_id, snapshot_type, snapshot_payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    run_id,
                    seq_id,
                    snapshot_type,
                    _json_dumps(snapshot_payload),
                    created_at.isoformat(),
                ),
            )

        return WalCheckpoint(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            seq_id=seq_id,
            snapshot_type=snapshot_type,
            snapshot_payload=snapshot_payload,
            created_at=created_at,
        )

    def read_latest_wal_checkpoint(self, run_id: str, snapshot_type: str | None = None):
        from .wal import WalCheckpoint

        run_id = require_non_empty_str(run_id, "run_id")
        conn, owned = self._borrow_connection()
        try:
            sql = (
                "SELECT checkpoint_id, run_id, seq_id, snapshot_type, snapshot_payload_json, created_at "
                "FROM wal_checkpoints WHERE run_id = ?"
            )
            params: list[Any] = [run_id]
            if snapshot_type is not None:
                sql += " AND snapshot_type = ?"
                params.append(require_non_empty_str(snapshot_type, "snapshot_type"))
            sql += " ORDER BY seq_id DESC, created_at DESC LIMIT 1"
            row = conn.execute(sql, tuple(params)).fetchone()
        finally:
            if owned:
                conn.close()

        if row is None:
            return None
        return WalCheckpoint(
            checkpoint_id=str(row[0]),
            run_id=str(row[1]),
            seq_id=int(row[2]),
            snapshot_type=str(row[3]),
            snapshot_payload=_json_loads(str(row[4]), default={}),
            created_at=parse_datetime(row[5]),
        )

    def next_evaluation_attempt_no(
        self,
        run_id: str,
        *,
        step_index: int,
        plan_item_id: str | None,
    ) -> int:
        run_id = require_non_empty_str(run_id, "run_id")
        if not isinstance(step_index, int) or step_index < 0:
            raise ValueError("step_index must be a non-negative integer")

        conn, owned = self._borrow_connection()
        try:
            if plan_item_id is None:
                row = conn.execute(
                    """
                    SELECT MAX(attempt_no)
                    FROM evaluation_spans
                    WHERE run_id = ? AND step_index = ? AND plan_item_id IS NULL AND phase = 'step_total'
                    """,
                    (run_id, step_index),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT MAX(attempt_no)
                    FROM evaluation_spans
                    WHERE run_id = ? AND step_index = ? AND plan_item_id = ? AND phase = 'step_total'
                    """,
                    (run_id, step_index, plan_item_id),
                ).fetchone()
        finally:
            if owned:
                conn.close()

        current = 0 if row is None or row[0] is None else int(row[0])
        return current + 1

    def start_evaluation_span(
        self,
        *,
        run_id: str,
        step_index: int | None,
        plan_item_id: str | None,
        attempt_no: int,
        phase: str,
        parent_span_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run_id = require_non_empty_str(run_id, "run_id")
        phase = require_non_empty_str(phase, "phase")
        if step_index is not None and (not isinstance(step_index, int) or step_index < 0):
            raise ValueError("step_index must be a non-negative integer when provided")
        if not isinstance(attempt_no, int) or attempt_no <= 0:
            raise ValueError("attempt_no must be a positive integer")
        metadata = require_dict(metadata or {}, "metadata")

        span_id = f"eval_{uuid4().hex}"
        now = utc_now()
        payload = {
            "span_id": span_id,
            "run_id": run_id,
            "step_index": step_index,
            "plan_item_id": plan_item_id,
            "attempt_no": attempt_no,
            "phase": phase,
            "parent_span_id": parent_span_id,
            "status": "running",
            "started_at": now.isoformat(),
            "finished_at": None,
            "duration_ms": None,
            "metadata": metadata,
            "error_message": None,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        with self.transaction(immediate=True) as conn:
            conn.execute(
                """
                INSERT INTO evaluation_spans(
                    span_id, run_id, step_index, plan_item_id, attempt_no, phase, parent_span_id,
                    status, started_at, finished_at, duration_ms, metadata_json, error_message,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["span_id"],
                    payload["run_id"],
                    payload["step_index"],
                    payload["plan_item_id"],
                    payload["attempt_no"],
                    payload["phase"],
                    payload["parent_span_id"],
                    payload["status"],
                    payload["started_at"],
                    payload["finished_at"],
                    payload["duration_ms"],
                    _json_dumps(payload["metadata"]),
                    payload["error_message"],
                    payload["created_at"],
                    payload["updated_at"],
                ),
            )
        return payload

    def finish_evaluation_span(
        self,
        span_id: str,
        *,
        status: str,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        span_id = require_non_empty_str(span_id, "span_id")
        status = require_non_empty_str(status, "status")
        if status not in {"succeeded", "failed", "interrupted"}:
            raise ValueError("status must be succeeded, failed, or interrupted")
        metadata_update = require_dict(metadata or {}, "metadata")

        with self.transaction(immediate=True) as conn:
            row = conn.execute(
                """
                SELECT span_id, run_id, step_index, plan_item_id, attempt_no, phase, parent_span_id,
                       status, started_at, finished_at, duration_ms, metadata_json, error_message,
                       created_at, updated_at
                FROM evaluation_spans
                WHERE span_id = ?
                """,
                (span_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"evaluation span not found: {span_id}")

            existing = _evaluation_span_from_row(row)
            if existing["finished_at"] is not None:
                return existing

            finished_at = utc_now()
            started_at = parse_datetime(existing["started_at"])
            duration_ms = max(0.0, (finished_at - started_at).total_seconds() * 1000.0)
            merged_metadata = {
                **existing["metadata"],
                **metadata_update,
            }
            conn.execute(
                """
                UPDATE evaluation_spans
                SET status = ?, finished_at = ?, duration_ms = ?, metadata_json = ?,
                    error_message = ?, updated_at = ?
                WHERE span_id = ?
                """,
                (
                    status,
                    finished_at.isoformat(),
                    duration_ms,
                    _json_dumps(merged_metadata),
                    error_message,
                    finished_at.isoformat(),
                    span_id,
                ),
            )

        return {
            **existing,
            "status": status,
            "finished_at": finished_at.isoformat(),
            "duration_ms": duration_ms,
            "metadata": merged_metadata,
            "error_message": error_message,
            "updated_at": finished_at.isoformat(),
        }

    def list_evaluation_spans(self, run_id: str) -> list[dict[str, Any]]:
        run_id = require_non_empty_str(run_id, "run_id")
        conn, owned = self._borrow_connection()
        try:
            rows = conn.execute(
                """
                SELECT span_id, run_id, step_index, plan_item_id, attempt_no, phase, parent_span_id,
                       status, started_at, finished_at, duration_ms, metadata_json, error_message,
                       created_at, updated_at
                FROM evaluation_spans
                WHERE run_id = ?
                ORDER BY started_at ASC, span_id ASC
                """,
                (run_id,),
            ).fetchall()
        finally:
            if owned:
                conn.close()
        return [_evaluation_span_from_row(row) for row in rows]

    def flush(self) -> None:
        """SQLite is the canonical durable store, so there is nothing to flush here."""

    # if payload too large, store seperately
    def _store_payload(
        self,
        conn: sqlite3.Connection,
        *,
        execution_id: str,
        kind: str,
        payload: dict[str, Any] | None,
    ) -> tuple[str | None, dict[str, Any] | None]:
        if payload is None:
            return None, None
        size_bytes = json_size_bytes(payload)
        if size_bytes <= self.payload_inline_limit_bytes:
            return _json_dumps(payload), None

        sha256 = json_sha256(payload)
        payload_id = f"{execution_id}:{kind}:{sha256[:16]}"
        conn.execute(
            """
            INSERT INTO payload_artifacts(payload_id, payload_json, size_bytes, sha256, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(payload_id) DO NOTHING
            """,
            (payload_id, _json_dumps(payload), size_bytes, sha256, utc_now().isoformat()),
        )
        return None, {
            "storage": "sqlite",
            "payload_id": payload_id,
            "size_bytes": size_bytes,
            "sha256": sha256,
        }

    def _load_payload(
        self,
        inline_payload_json: str | None,
        ref_id: str | None,
        conn: sqlite3.Connection,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if inline_payload_json is not None:
            return _json_loads(str(inline_payload_json), default={}), None
        if ref_id is None:
            return None, None
        row = conn.execute(
            "SELECT payload_json, size_bytes, sha256 FROM payload_artifacts WHERE payload_id = ?",
            (ref_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"payload artifact not found: {ref_id}")
        return _json_loads(str(row[0]), default={}), {
            "storage": "sqlite",
            "payload_id": ref_id,
            "size_bytes": int(row[1]),
            "sha256": str(row[2]),
        }

    def _borrow_connection(self) -> tuple[sqlite3.Connection, bool]:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn, False
        return self._connect(), True

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.busy_timeout_ms / 1000,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        return conn


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _json_loads(raw: str, *, default: Any) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _evaluation_span_from_row(row: Any) -> dict[str, Any]:
    return {
        "span_id": str(row[0]),
        "run_id": str(row[1]),
        "step_index": row[2],
        "plan_item_id": row[3],
        "attempt_no": int(row[4]),
        "phase": str(row[5]),
        "parent_span_id": row[6],
        "status": str(row[7]),
        "started_at": str(row[8]),
        "finished_at": row[9],
        "duration_ms": None if row[10] is None else float(row[10]),
        "metadata": _json_loads(str(row[11]), default={}),
        "error_message": row[12],
        "created_at": str(row[13]),
        "updated_at": str(row[14]),
    }
