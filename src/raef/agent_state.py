"""Agent context and external result management for RAEF.

This module provides a local, durable state layer for:
- conversation/message history used in later model inference,
- runtime context pointers (turn index, step index, pending execution),
- cached tool-call results for recovery after crash/failure.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Protocol

from .models import ExecutionStatus, normalize_execution_status
from .runtime_store import SQLiteRuntimeStore
from .utils import (
    optional_str,
    parse_datetime,
    require_dict_or_default,
    require_non_empty_str,
    utc_now,
)


@dataclass(slots=True)
class AgentMessage:
    """One message in the persisted agent interaction history."""

    role: str  #  system / user / assistant / tool
    content: str  # actual text
    name: str | None = None  # tool name / function name
    tool_call_id: str | None = None  # reference to execution_id
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentMessage":
        role = require_non_empty_str(payload, "role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("role must be one of system,user,assistant,tool")

        return cls(
            role=role,
            content=require_non_empty_str(payload, "content"),
            name=optional_str(payload.get("name")),
            tool_call_id=optional_str(payload.get("tool_call_id")),
            metadata=require_dict_or_default(payload.get("metadata")),
            created_at=parse_datetime(payload.get("created_at")),
        )


"""
Example for turn index:

messages:
0: system
1: user
2: assistant
3: tool
4: assistant

turn_index = 5

"""

"""
TODO : Enforce AgentContext stay consistent with PlannerState

"""


# snapshot of the agent
@dataclass(slots=True)
class AgentContextSnapshot:
    """Persisted runtime context for one run."""

    run_id: str  # identifier for one agent execution session
    turn_index: int = 0  # number of messages processed so far
    step_index: int = 0  # synchronized cache of current step in PlannerState
    planner_version: int = 0  # synchronized cache of PlannerState version
    messages: list[AgentMessage] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)  # any k-v store that would form the agent memory
    pending_execution_id: str | None = None
    pending_execution_ids: list[str] = field(default_factory=list)  # supports multiple in-flight tool calls
    last_checkpoint_seq: int | None = None  # pointer to WAL position
    updated_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "turn_index": self.turn_index,
            "step_index": self.step_index,
            "planner_version": self.planner_version,
            "messages": [m.to_dict() for m in self.messages],
            "memory": self.memory,
            "pending_execution_id": self.pending_execution_id,
            "pending_execution_ids": list(self.pending_execution_ids),
            "last_checkpoint_seq": self.last_checkpoint_seq,
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentContextSnapshot":
        raw_messages = payload.get("messages", [])
        if not isinstance(raw_messages, list):
            raise ValueError("messages must be a list")

        raw_pending = payload.get("pending_execution_ids")
        if raw_pending is None:
            pending_ids = []
        elif isinstance(raw_pending, list):
            pending_ids = _normalize_pending_ids(raw_pending)
        else:
            raise ValueError("pending_execution_ids must be a list when provided")

        pending_execution_id = optional_str(payload.get("pending_execution_id"))
        if pending_execution_id is not None and pending_execution_id not in pending_ids:
            pending_ids.append(pending_execution_id)
        if pending_ids:
            pending_execution_id = pending_ids[-1]

        turn_index = payload.get("turn_index", 0)
        if not isinstance(turn_index, int) or turn_index < 0:
            raise ValueError("turn_index must be a non-negative integer")

        step_index = payload.get("step_index", 0)
        if not isinstance(step_index, int) or step_index < 0:
            raise ValueError("step_index must be a non-negative integer")

        planner_version = payload.get("planner_version", 0)
        if not isinstance(planner_version, int) or planner_version < 0:
            raise ValueError("planner_version must be a non-negative integer")

        last_checkpoint_seq = payload.get("last_checkpoint_seq")
        if last_checkpoint_seq is not None and (
            not isinstance(last_checkpoint_seq, int) or last_checkpoint_seq < 0
        ):
            raise ValueError("last_checkpoint_seq must be a non-negative integer when provided")

        return cls(
            run_id=require_non_empty_str(payload, "run_id"),
            turn_index=turn_index,
            step_index=step_index,
            planner_version=planner_version,
            messages=[AgentMessage.from_dict(msg) for msg in raw_messages],
            memory=require_dict_or_default(payload.get("memory")),
            pending_execution_id=pending_execution_id,
            pending_execution_ids=pending_ids,
            last_checkpoint_seq=last_checkpoint_seq,
            updated_at=parse_datetime(payload.get("updated_at")),
        )


class ExternalResultStatus(str, Enum):
    """Normalized status for cached external/tool outcome."""

    OK = "ok"  # Tool succeeded : safe to reuse result
    ERROR = "error"  # Tool failed definitively : may retry or surface error
    TIMEOUT = "timeout"  # uncertain completion : danger zone
    UNKNOWN = "unknown"  # not yet classified : needs verification


"""
TODO:
1. similar to planner_state, always r/w to durable, need change to in memory
2. what if the input/return files/size too large?

"""


@dataclass(slots=True)
class ExternalResultRecord:
    """Cached record of one tool/external call result."""

    execution_id: str  # unique ID per tool call
    run_id: str
    plan_item_id: str | None  # which plan step triggered it (PlannerItem)
    tool_name: str
    operation_type: str
    request_payload: dict[str, Any] | None  # exact input arguments, possibly stored via sqlite payload artifacts
    response_payload: dict[str, Any] | None
    request_payload_ref: dict[str, Any] | None = None
    response_payload_ref: dict[str, Any] | None = None
    result_status: ExternalResultStatus = ExternalResultStatus.UNKNOWN
    execution_status: ExecutionStatus = ExecutionStatus.INTENT_LOGGED
    error_message: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "run_id": self.run_id,
            "plan_item_id": self.plan_item_id,
            "tool_name": self.tool_name,
            "operation_type": self.operation_type,
            "request_payload": self.request_payload,
            "response_payload": self.response_payload,
            "request_payload_ref": self.request_payload_ref,
            "response_payload_ref": self.response_payload_ref,
            "result_status": self.result_status.value,
            "execution_status": self.execution_status.value,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExternalResultRecord":
        request_payload_raw = payload.get("request_payload")
        if request_payload_raw is not None and not isinstance(request_payload_raw, dict):
            raise ValueError("request_payload must be an object or null")

        response_payload_raw = payload.get("response_payload")
        if response_payload_raw is not None and not isinstance(response_payload_raw, dict):
            raise ValueError("response_payload must be an object or null")

        request_payload_ref = _optional_ref_dict(payload.get("request_payload_ref"), "request_payload_ref")
        response_payload_ref = _optional_ref_dict(payload.get("response_payload_ref"), "response_payload_ref")
        if request_payload_raw is None and request_payload_ref is None:
            raise ValueError("request_payload must be present inline or via request_payload_ref")

        return cls(
            execution_id=require_non_empty_str(payload, "execution_id"),
            run_id=require_non_empty_str(payload, "run_id"),
            plan_item_id=optional_str(payload.get("plan_item_id")),
            tool_name=require_non_empty_str(payload, "tool_name"),
            operation_type=require_non_empty_str(payload, "operation_type"),
            request_payload=request_payload_raw,
            response_payload=response_payload_raw,
            request_payload_ref=request_payload_ref,
            response_payload_ref=response_payload_ref,
            result_status=ExternalResultStatus(str(payload.get("result_status", "unknown"))),
            execution_status=normalize_execution_status(str(payload.get("execution_status", "intent_logged"))),
            error_message=optional_str(payload.get("error_message")),
            created_at=parse_datetime(payload.get("created_at")),
            updated_at=parse_datetime(payload.get("updated_at")),
        )


class AgentStateRepository(Protocol):
    """Persistence contract for agent context and external results."""

    def load_context(self, run_id: str) -> AgentContextSnapshot | None:
        """Load context snapshot for run_id."""

    def save_context(self, snapshot: AgentContextSnapshot) -> AgentContextSnapshot:
        """Persist context snapshot for run_id."""

    def upsert_external_result(self, result: ExternalResultRecord) -> ExternalResultRecord:
        """Insert or update one external result record."""

    def get_external_result(self, execution_id: str) -> ExternalResultRecord | None:
        """Lookup external result by execution id across cached runs."""

    def list_external_results(self, run_id: str) -> list[ExternalResultRecord]:
        """Return external results for one run, sorted by creation time."""

    def flush(self) -> None:
        """Persist any dirty in-memory state."""


class SQLiteAgentStateRepository:
    """SQLite-backed repository for local development and crash recovery."""

    def __init__(
        self,
        store: SQLiteRuntimeStore | None = None,
        db_path: Path | None = None,
        *,
        busy_timeout_ms: int = 5000,
        payload_inline_limit_bytes: int = 16_384,
    ) -> None:
        if store is None:
            if db_path is None:
                repo_root = Path(__file__).resolve().parents[2]
                db_path = repo_root / "data" / "raef_runtime" / "runtime.sqlite"
            store = SQLiteRuntimeStore(
                db_path,
                busy_timeout_ms=busy_timeout_ms,
                payload_inline_limit_bytes=payload_inline_limit_bytes,
            )
        self.store = store

    def load_context(self, run_id: str) -> AgentContextSnapshot | None:
        snapshot = self.store.load_context_snapshot(run_id)
        if snapshot is None:
            return None
        return AgentContextSnapshot.from_dict(snapshot.to_dict())

    def save_context(self, snapshot: AgentContextSnapshot) -> AgentContextSnapshot:
        snapshot.updated_at = utc_now()
        self.store.save_context_snapshot(snapshot)
        return AgentContextSnapshot.from_dict(snapshot.to_dict())

    def upsert_external_result(self, result: ExternalResultRecord) -> ExternalResultRecord:
        result.updated_at = utc_now()
        stored = self.store.upsert_external_result(result)
        return ExternalResultRecord.from_dict(stored.to_dict())

    def get_external_result(self, execution_id: str) -> ExternalResultRecord | None:
        record = self.store.get_external_result(execution_id)
        if record is None:
            return None
        return ExternalResultRecord.from_dict(record.to_dict())

    def list_external_results(self, run_id: str) -> list[ExternalResultRecord]:
        return [
            ExternalResultRecord.from_dict(record.to_dict())
            for record in self.store.list_external_results(run_id)
        ]

    def flush(self) -> None:
        self.store.flush()


class AgentContextService:
    """High-level API for context history and external result cache management."""

    def __init__(self, repository: AgentStateRepository | None = None) -> None:
        self.repository = repository or SQLiteAgentStateRepository()

    def init_context(
        self,
        run_id: str,
        seed_messages: list[dict[str, Any]] | None = None,
        initial_memory: dict[str, Any] | None = None,
        force_reset: bool = False,
    ) -> AgentContextSnapshot:
        existing = self.repository.load_context(run_id)
        if existing is not None and not force_reset:
            return existing

        messages = [self._message_from_dict(msg) for msg in (seed_messages or [])]
        snapshot = AgentContextSnapshot(
            run_id=run_id,
            turn_index=len(messages),
            messages=messages,
            memory=dict(initial_memory or {}),
        )
        return self.repository.save_context(snapshot)

    def load_context(self, run_id: str) -> AgentContextSnapshot | None:
        return self.repository.load_context(run_id)

    def append_message(
        self,
        run_id: str,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
        *,
        name: str | None = None,
        tool_call_id: str | None = None,
    ) -> AgentContextSnapshot:
        snapshot = self._load_required_context(run_id)
        message = AgentMessage(
            role=role,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            metadata=dict(meta or {}),
        )
        # Reuse strict validation path.
        AgentMessage.from_dict(message.to_dict())

        snapshot.messages.append(message)
        snapshot.turn_index += 1
        snapshot.updated_at = utc_now()
        return self.repository.save_context(snapshot)

    def append_messages(self, run_id: str, messages: Iterable[dict[str, Any]]) -> AgentContextSnapshot:
        snapshot = self._load_required_context(run_id)
        for msg in messages:
            parsed = self._message_from_dict(msg)
            snapshot.messages.append(parsed)
            snapshot.turn_index += 1
        snapshot.updated_at = utc_now()
        return self.repository.save_context(snapshot)

    def sync_planner_state(
        self,
        run_id: str,
        *,
        planner_version: int,
        step_index: int | None = None,
    ) -> AgentContextSnapshot:
        snapshot = self._load_required_context(run_id)
        if not isinstance(planner_version, int) or planner_version < 0:
            raise ValueError("planner_version must be a non-negative integer")
        if step_index is not None and (not isinstance(step_index, int) or step_index < 0):
            raise ValueError("step_index must be a non-negative integer when provided")
        snapshot.planner_version = planner_version
        if step_index is not None:
            snapshot.step_index = step_index
        snapshot.updated_at = utc_now()
        return self.repository.save_context(snapshot)

    def set_step_index(self, run_id: str, step_index: int) -> AgentContextSnapshot:
        snapshot = self._load_required_context(run_id)
        return self.sync_planner_state(
            run_id,
            planner_version=snapshot.planner_version,
            step_index=step_index,
        )

    def set_planner_version(self, run_id: str, planner_version: int) -> AgentContextSnapshot:
        snapshot = self._load_required_context(run_id)
        return self.sync_planner_state(
            run_id,
            planner_version=planner_version,
            step_index=snapshot.step_index,
        )

    def add_pending_execution(self, run_id: str, execution_id: str) -> AgentContextSnapshot:
        snapshot = self._load_required_context(run_id)
        execution_id = require_non_empty_str(execution_id, "execution_id")
        if execution_id not in snapshot.pending_execution_ids:
            snapshot.pending_execution_ids.append(execution_id)
        snapshot.pending_execution_id = snapshot.pending_execution_ids[-1]
        snapshot.updated_at = utc_now()
        return self.repository.save_context(snapshot)

    def clear_pending_execution(self, run_id: str, execution_id: str | None = None) -> AgentContextSnapshot:
        snapshot = self._load_required_context(run_id)
        if execution_id is None:
            snapshot.pending_execution_ids = []
            snapshot.pending_execution_id = None
        else:
            normalized = require_non_empty_str(execution_id, "execution_id")
            snapshot.pending_execution_ids = [
                existing for existing in snapshot.pending_execution_ids if existing != normalized
            ]
            snapshot.pending_execution_id = (
                snapshot.pending_execution_ids[-1] if snapshot.pending_execution_ids else None
            )
        snapshot.updated_at = utc_now()
        return self.repository.save_context(snapshot)

    def set_pending_execution(
        self,
        run_id: str,
        execution_id: str | None,
    ) -> AgentContextSnapshot:
        if execution_id is None:
            return self.clear_pending_execution(run_id)
        return self.add_pending_execution(run_id, execution_id)

    def checkpoint_context(
        self,
        run_id: str,
        *,
        last_checkpoint_seq: int | None,
    ) -> AgentContextSnapshot:
        snapshot = self._load_required_context(run_id)
        if last_checkpoint_seq is not None and (
            not isinstance(last_checkpoint_seq, int) or last_checkpoint_seq < 0
        ):
            raise ValueError("last_checkpoint_seq must be a non-negative integer when provided")
        snapshot.last_checkpoint_seq = last_checkpoint_seq
        snapshot.updated_at = utc_now()
        return self.repository.save_context(snapshot)

    def save_external_result(
        self,
        run_id: str,
        result: ExternalResultRecord | dict[str, Any],
    ) -> ExternalResultRecord:
        self._load_required_context(run_id)
        record = result if isinstance(result, ExternalResultRecord) else ExternalResultRecord.from_dict(result)
        if record.run_id != run_id:
            raise ValueError("external result run_id must match the target context run_id")
        return self.repository.upsert_external_result(record)

    def get_external_result(self, execution_id: str) -> ExternalResultRecord | None:
        return self.repository.get_external_result(execution_id)

    def list_external_results(self, run_id: str) -> list[ExternalResultRecord]:
        self._load_required_context(run_id)
        return self.repository.list_external_results(run_id)

    def get_messages_for_inference(
        self,
        run_id: str,
        *,
        max_messages: int | None = None,
        roles: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        snapshot = self._load_required_context(run_id)
        messages = list(snapshot.messages)
        if roles is not None:
            messages = [message for message in messages if message.role in roles]
        if max_messages is not None:
            if not isinstance(max_messages, int) or max_messages <= 0:
                raise ValueError("max_messages must be a positive integer when provided")
            messages = messages[-max_messages:]
        return [message.to_dict() for message in messages]

    def flush(self) -> None:
        self.repository.flush()

    def _load_required_context(self, run_id: str) -> AgentContextSnapshot:
        snapshot = self.repository.load_context(run_id)
        if snapshot is None:
            raise ValueError(f"agent context not found for run_id={run_id}")
        return snapshot

    @staticmethod
    def _message_from_dict(payload: dict[str, Any]) -> AgentMessage:
        return AgentMessage.from_dict(payload)


def _normalize_pending_ids(raw_pending: list[Any]) -> list[str]:
    pending_ids: list[str] = []
    for raw in raw_pending:
        normalized = optional_str(raw)
        if normalized is None:
            continue
        text = normalized.strip()
        if not text:
            continue
        if text not in pending_ids:
            pending_ids.append(text)
    return pending_ids


def _optional_ref_dict(value: Any, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object when provided")
    payload_id = value.get("payload_id")
    if not isinstance(payload_id, str) or not payload_id.strip():
        raise ValueError(f"{field_name} must include payload_id")
    storage = value.get("storage")
    if storage is not None and not isinstance(storage, str):
        raise ValueError(f"{field_name}.storage must be a string when provided")
    size_bytes = value.get("size_bytes")
    if size_bytes is not None and (not isinstance(size_bytes, int) or size_bytes < 0):
        raise ValueError(f"{field_name}.size_bytes must be a non-negative integer when provided")
    sha256 = value.get("sha256")
    if sha256 is not None and not isinstance(sha256, str):
        raise ValueError(f"{field_name}.sha256 must be a string when provided")
    return {
        "storage": storage or "sqlite",
        "payload_id": payload_id.strip(),
        "size_bytes": size_bytes,
        "sha256": sha256,
    }
