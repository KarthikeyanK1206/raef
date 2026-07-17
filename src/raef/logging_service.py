"""Application-facing logging orchestration service for RAEF.

This module provides a facade that coordinates three lower-level components:
- planner_state: explicit plan list and item lifecycle,
- agent_state: history/context and external result cache,
- wal: write-ahead event/checkpoint log.

Design goal:
- log-first mutation flow for transaction manager and recovery protocol users.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import threading
from typing import Any, Literal

from .agent_state import (
    AgentContextService,
    ExternalResultRecord,
    ExternalResultStatus,
    SQLiteAgentStateRepository,
)
from .models import ExecutionStatus, normalize_execution_status
from .planner_state import PlannerState, PlannerStateService, SQLitePlannerStateRepository
from .logging_observability import SoftJSONObservability
from .runtime_store import SQLiteRuntimeStore
from .utils import utc_now
from .wal import SQLiteWriteAheadLog, WalCheckpoint, WalEvent, WriteAheadLog

logger = logging.getLogger(__name__)


class LoggingEventType(str, Enum):
    """Canonical event types emitted by the logging facade."""

    RUN_STARTED = "RUN_STARTED"
    PLAN_CREATED = "PLAN_CREATED"
    PLAN_ITEM_LLM_OUTPUT_ATTACHED = "PLAN_ITEM_LLM_OUTPUT_ATTACHED"
    PLAN_ITEM_STATUS_UPDATED = "PLAN_ITEM_STATUS_UPDATED"
    PLAN_ACTIVE_ITEM_SET = "PLAN_ACTIVE_ITEM_SET"
    CONTEXT_INITIALIZED = "CONTEXT_INITIALIZED"
    CONTEXT_MESSAGE_APPENDED = "CONTEXT_MESSAGE_APPENDED"
    CONTEXT_PENDING_EXECUTION_SET = "CONTEXT_PENDING_EXECUTION_SET"
    CONTEXT_SYNCHRONIZED = "CONTEXT_SYNCHRONIZED"
    TOOL_INTENT_RECORDED = "TOOL_INTENT_RECORDED"
    TOOL_DISPATCHED = "TOOL_DISPATCHED"
    EXTERNAL_RESULT_RECORDED = "EXTERNAL_RESULT_RECORDED"
    CHECKPOINT_WRITTEN = "CHECKPOINT_WRITTEN"


@dataclass(slots=True)
class LoggingServiceConfig:
    """Configuration for logging facade behavior."""

    checkpoint_every_n_events: int = 0
    enforce_execution_dedup: bool = True
    payload_inline_limit_bytes: int = 16_384
    sqlite_busy_timeout_ms: int = 5000
    soft_write_enabled: bool = True


class LoggingService:
    """Facade that enforces log-first operations over planner/context state."""

    _ALLOWED_EXECUTION_TRANSITIONS: dict[ExecutionStatus, set[ExecutionStatus]] = {
        ExecutionStatus.INTENT_LOGGED: {
            ExecutionStatus.DISPATCHED,
            ExecutionStatus.ACKED,
            ExecutionStatus.VERIFIED_COMMITTED,
            ExecutionStatus.VERIFIED_NOT_FOUND,
            ExecutionStatus.FAILED,
        },
        ExecutionStatus.DISPATCHED: {
            ExecutionStatus.ACKED,
            ExecutionStatus.VERIFIED_COMMITTED,
            ExecutionStatus.VERIFIED_NOT_FOUND,
            ExecutionStatus.FAILED,
        },
        ExecutionStatus.ACKED: {
            ExecutionStatus.VERIFIED_COMMITTED,
        },
        ExecutionStatus.VERIFIED_COMMITTED: set(),
        ExecutionStatus.VERIFIED_NOT_FOUND: {
            ExecutionStatus.ACKED,
            ExecutionStatus.FAILED,
        },
        ExecutionStatus.FAILED: {
            ExecutionStatus.DISPATCHED,
            ExecutionStatus.VERIFIED_COMMITTED,
            ExecutionStatus.VERIFIED_NOT_FOUND,
        },
    }

    def __init__(
        self,
        *,
        store: SQLiteRuntimeStore | None = None,
        planner_service: PlannerStateService | None = None,
        context_service: AgentContextService | None = None,
        wal: WriteAheadLog | None = None,
        observability: Any | None = None,
        config: LoggingServiceConfig | None = None,
        soft_write_root: Path | None = None,
    ) -> None:
        self.config = config or LoggingServiceConfig()
        self.store = store or SQLiteRuntimeStore(
            (Path(__file__).resolve().parents[2] / "data" / "raef_runtime" / "runtime.sqlite"),
            busy_timeout_ms=self.config.sqlite_busy_timeout_ms,
            payload_inline_limit_bytes=self.config.payload_inline_limit_bytes,
        )
        self.planner_service = planner_service or PlannerStateService(
            SQLitePlannerStateRepository(self.store)
        )
        self.context_service = context_service or AgentContextService(
            SQLiteAgentStateRepository(
                self.store,
                payload_inline_limit_bytes=self.config.payload_inline_limit_bytes,
            )
        )
        self.wal = wal or SQLiteWriteAheadLog(store=self.store)
        self.observability = observability
        if self.observability is None and self.config.soft_write_enabled:
            resolved_soft_write_root = soft_write_root or (self.store.db_path.parent / "soft_logs")
            self.observability = SoftJSONObservability(resolved_soft_write_root, reader=self)
        self.wal.init_schema()
        self._run_locks: dict[str, threading.RLock] = {}
        self._run_locks_guard = threading.Lock()

    # Builds a LoggingService rooted at a specific local folder
    @classmethod
    def with_data_root( 
        cls,
        data_root: Path,
        *,
        checkpoint_every_n_events: int = 0,
        wal_backend: Literal["sqlite"] = "sqlite",
        enforce_execution_dedup: bool = True,
        payload_inline_limit_bytes: int = 16_384,
        sqlite_busy_timeout_ms: int = 5000,
        soft_write_enabled: bool = True,
    ) -> "LoggingService":
        """Create a service instance using a custom local data root path."""
        if wal_backend != "sqlite":
            raise ValueError("wal_backend must be 'sqlite'")
        store = SQLiteRuntimeStore(
            data_root / "runtime.sqlite",
            busy_timeout_ms=sqlite_busy_timeout_ms,
            payload_inline_limit_bytes=payload_inline_limit_bytes,
        )
        config = LoggingServiceConfig(
            checkpoint_every_n_events=checkpoint_every_n_events,
            enforce_execution_dedup=enforce_execution_dedup,
            payload_inline_limit_bytes=payload_inline_limit_bytes,
            sqlite_busy_timeout_ms=sqlite_busy_timeout_ms,
            soft_write_enabled=soft_write_enabled,
        )
        return cls(store=store, config=config, soft_write_root=data_root / "soft_logs")

    def start_run(
        self,
        run_id: str,
        initial_messages: list[dict[str, Any]],
        plan_source_text: str,
        plan_items: list[dict[str, Any]],
        *,
        initial_memory: dict[str, Any] | None = None,
        force_reset: bool = False,
        plan_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        events: list[WalEvent] = []
        checkpoint: WalCheckpoint | None = None
        with self._run_write(run_id):
            existing_planner = self.planner_service.load_plan(run_id)
            existing_context = self.context_service.load_context(run_id)
            if existing_planner is not None and existing_context is not None and not force_reset:
                logger.info("resuming existing run: run_id=%s", run_id)
                result = {
                    "planner_state": existing_planner.to_dict(),
                    "context": existing_context.to_dict(),
                }
            else:
                logger.info(
                    "starting run: run_id=%s plan_items=%d force_reset=%s",
                    run_id,
                    len(plan_items),
                    force_reset,
                )
                events.append(
                    self._append_event(
                        run_id=run_id,
                        event_type=LoggingEventType.RUN_STARTED,
                        entity_type="run",
                        entity_id=run_id,
                        payload={
                            "initial_message_count": len(initial_messages),
                            "plan_item_count": len(plan_items),
                        },
                    )
                )
                events.append(
                    self._append_event(
                        run_id=run_id,
                        event_type=LoggingEventType.PLAN_CREATED,
                        entity_type="planner",
                        entity_id=run_id,
                        payload={
                            "plan_source_text": plan_source_text,
                            "plan_items": plan_items,
                            "plan_schema": plan_schema or {},
                        },
                    )
                )
                planner_state = self.planner_service.create_plan(
                    run_id=run_id,
                    plan_source_text=plan_source_text,
                    items=plan_items,
                    plan_schema=plan_schema,
                )
                events.append(
                    self._append_event(
                        run_id=run_id,
                        event_type=LoggingEventType.CONTEXT_INITIALIZED,
                        entity_type="context",
                        entity_id=run_id,
                        payload={
                            "initial_messages": initial_messages,
                            "initial_memory": initial_memory or {},
                        },
                    )
                )
                context_snapshot = self.context_service.init_context(
                    run_id=run_id,
                    seed_messages=initial_messages,
                    initial_memory=initial_memory,
                    force_reset=force_reset,
                )
                events.append(self._sync_context_with_planner_locked(run_id=run_id, planner_state=planner_state))
                checkpoint = self._maybe_auto_checkpoint_locked(run_id, events)
                result = {
                    "planner_state": planner_state.to_dict(),
                    "context": context_snapshot.to_dict(),
                }
        self._emit_after_commit(run_id, events=events, checkpoint=checkpoint)
        return result

    def record_context_message(
        self,
        run_id: str,
        *,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
        name: str | None = None,
        tool_call_id: str | None = None,
    ) -> dict[str, Any]:
        events: list[WalEvent] = []
        checkpoint: WalCheckpoint | None = None
        with self._run_write(run_id):
            events.append(
                self._append_event(
                    run_id=run_id,
                    event_type=LoggingEventType.CONTEXT_MESSAGE_APPENDED,
                    entity_type="context",
                    entity_id=run_id,
                    payload={
                        "role": role,
                        "content": content,
                        "meta": meta or {},
                        "name": name,
                        "tool_call_id": tool_call_id,
                    },
                )
            )
            context = self.context_service.append_message(
                run_id=run_id,
                role=role,
                content=content,
                meta=meta,
                name=name,
                tool_call_id=tool_call_id,
            )
            checkpoint = self._maybe_auto_checkpoint_locked(run_id, events)
            result = context.to_dict()
        self._emit_after_commit(run_id, events=events, checkpoint=checkpoint)
        return result

    def record_llm_turn(
        self,
        run_id: str,
        plan_item_id: str,
        llm_output: str,
        assistant_message: str,
        *,
        assistant_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        events: list[WalEvent] = []
        checkpoint: WalCheckpoint | None = None
        with self._run_write(run_id):
            events.append(
                self._append_event(
                    run_id=run_id,
                    event_type=LoggingEventType.PLAN_ITEM_LLM_OUTPUT_ATTACHED,
                    entity_type="planner_item",
                    entity_id=plan_item_id,
                    payload={"llm_output": llm_output},
                )
            )
            planner_state = self.planner_service.attach_llm_output(
                run_id=run_id,
                plan_item_id=plan_item_id,
                llm_output=llm_output,
            )
            events.append(
                self._append_event(
                    run_id=run_id,
                    event_type=LoggingEventType.CONTEXT_MESSAGE_APPENDED,
                    entity_type="context",
                    entity_id=run_id,
                    payload={
                        "role": "assistant",
                        "content": assistant_message,
                        "meta": assistant_meta or {},
                        "name": None,
                        "tool_call_id": None,
                    },
                )
            )
            context = self.context_service.append_message(
                run_id=run_id,
                role="assistant",
                content=assistant_message,
                meta=assistant_meta,
            )
            events.append(self._sync_context_with_planner_locked(run_id=run_id, planner_state=planner_state))
            checkpoint = self._maybe_auto_checkpoint_locked(run_id, events)
            result = {
                "planner_state": planner_state.to_dict(),
                "context": context.to_dict(),
            }
        self._emit_after_commit(run_id, events=events, checkpoint=checkpoint)
        return result

    def record_tool_intent(
        self,
        run_id: str,
        plan_item_id: str,
        execution_id: str,
        tool_name: str,
        request_payload: dict[str, Any],
        *,
        operation_type: str,
    ) -> dict[str, Any]:
        events: list[WalEvent] = []
        checkpoint: WalCheckpoint | None = None
        with self._run_write(run_id):
            if self.config.enforce_execution_dedup and self._tool_intent_exists_locked(run_id, execution_id):
                planner_state = self.planner_service.load_plan(run_id)
                context = self.context_service.load_context(run_id)
                if planner_state is None or context is None:
                    raise ValueError(f"run projections missing for run_id={run_id}")
                result = {
                    "planner_state": planner_state.to_dict(),
                    "context": context.to_dict(),
                }
            else:
                context_snapshot = self.context_service.load_context(run_id)
                pending_after = _project_pending_ids_with_add(context_snapshot, execution_id)
                events.append(
                    self._append_event(
                        run_id=run_id,
                        event_type=LoggingEventType.TOOL_INTENT_RECORDED,
                        entity_type="tool_intent",
                        entity_id=execution_id,
                        payload={
                            "plan_item_id": plan_item_id,
                            "tool_name": tool_name,
                            "operation_type": operation_type,
                            "request_payload": request_payload,
                            "execution_status": ExecutionStatus.INTENT_LOGGED.value,
                        },
                    )
                )
                planner_state = self.planner_service.attach_tool_call_ref(
                    run_id=run_id,
                    plan_item_id=plan_item_id,
                    execution_id=execution_id,
                )
                events.append(
                    self._append_event(
                        run_id=run_id,
                        event_type=LoggingEventType.CONTEXT_PENDING_EXECUTION_SET,
                        entity_type="context",
                        entity_id=run_id,
                        payload={
                            "pending_execution_id": pending_after[-1],
                            "pending_execution_ids": pending_after,
                        },
                    )
                )
                context = self.context_service.add_pending_execution(
                    run_id=run_id,
                    execution_id=execution_id,
                )
                if self.context_service.get_external_result(execution_id) is None:
                    self.context_service.save_external_result(
                        run_id=run_id,
                        result=ExternalResultRecord(
                            execution_id=execution_id,
                            run_id=run_id,
                            plan_item_id=plan_item_id,
                            tool_name=tool_name,
                            operation_type=operation_type,
                            request_payload=request_payload,
                            response_payload=None,
                            result_status=ExternalResultStatus.UNKNOWN,
                            execution_status=ExecutionStatus.INTENT_LOGGED,
                            error_message=None,
                        ),
                    )
                events.append(self._sync_context_with_planner_locked(run_id=run_id, planner_state=planner_state))
                checkpoint = self._maybe_auto_checkpoint_locked(run_id, events)
                result = {
                    "planner_state": planner_state.to_dict(),
                    "context": context.to_dict(),
                }
        self._emit_after_commit(run_id, events=events, checkpoint=checkpoint)
        return result

    def record_tool_dispatch(self, run_id: str, execution_id: str) -> ExternalResultRecord:
        events: list[WalEvent] = []
        checkpoint: WalCheckpoint | None = None
        with self._run_write(run_id):
            cached = self.context_service.get_external_result(execution_id)
            if cached is None:
                raise ValueError(f"tool intent not found for execution_id={execution_id}")
            if cached.run_id != run_id:
                raise ValueError("execution_id belongs to another run")
            self._assert_execution_transition(cached.execution_status, ExecutionStatus.DISPATCHED)
            events.append(
                self._append_event(
                    run_id=run_id,
                    event_type=LoggingEventType.TOOL_DISPATCHED,
                    entity_type="tool_intent",
                    entity_id=execution_id,
                    payload={
                        "execution_status": ExecutionStatus.DISPATCHED.value,
                        "tool_name": cached.tool_name,
                    },
                )
            )
            record = self.context_service.save_external_result(
                run_id=run_id,
                result=ExternalResultRecord(
                    execution_id=cached.execution_id,
                    run_id=cached.run_id,
                    plan_item_id=cached.plan_item_id,
                    tool_name=cached.tool_name,
                    operation_type=cached.operation_type,
                    request_payload=cached.request_payload,
                    response_payload=cached.response_payload,
                    request_payload_ref=cached.request_payload_ref,
                    response_payload_ref=cached.response_payload_ref,
                    result_status=cached.result_status,
                    execution_status=ExecutionStatus.DISPATCHED,
                    error_message=cached.error_message,
                    created_at=cached.created_at,
                ),
            )
            checkpoint = self._maybe_auto_checkpoint_locked(run_id, events)
        self._emit_after_commit(run_id, events=events, checkpoint=checkpoint)
        return record

    def record_tool_result(
        self,
        run_id: str,
        execution_id: str,
        result_status: str,
        *,
        response_payload: dict[str, Any] | None,
        error_message: str | None = None,
        tool_name: str | None = None,
        operation_type: str | None = None,
        request_payload: dict[str, Any] | None = None,
        plan_item_id: str | None = None,
        execution_status: str | None = None,
        clear_pending: bool = True,
    ) -> ExternalResultRecord:
        events: list[WalEvent] = []
        checkpoint: WalCheckpoint | None = None
        with self._run_write(run_id):
            normalized_status = _parse_result_status(result_status)
            normalized_execution_status = _resolve_execution_status(
                execution_status=execution_status,
                result_status=normalized_status,
            )
            cached = self.context_service.get_external_result(execution_id)
            if cached is not None:
                if cached.run_id != run_id:
                    raise ValueError("execution_id belongs to another run")
                self._assert_execution_transition(cached.execution_status, normalized_execution_status)

            tool_name = tool_name or (cached.tool_name if cached is not None else None)
            operation_type = operation_type or (cached.operation_type if cached is not None else None)
            request_payload = request_payload or (cached.request_payload if cached is not None else None)
            plan_item_id = (
                plan_item_id if plan_item_id is not None else (cached.plan_item_id if cached is not None else None)
            )
            if tool_name is None or operation_type is None or request_payload is None:
                raise ValueError(
                    "tool_name, operation_type, and request_payload are required for first result record"
                )

            events.append(
                self._append_event(
                    run_id=run_id,
                    event_type=LoggingEventType.EXTERNAL_RESULT_RECORDED,
                    entity_type="external_result",
                    entity_id=execution_id,
                    payload={
                        "result_status": normalized_status.value,
                        "execution_status": normalized_execution_status.value,
                        "tool_name": tool_name,
                        "operation_type": operation_type,
                        "plan_item_id": plan_item_id,
                        "has_response": response_payload is not None,
                        "error_message": error_message,
                    },
                )
            )
            record = self.context_service.save_external_result(
                run_id=run_id,
                result=ExternalResultRecord(
                    execution_id=execution_id,
                    run_id=run_id,
                    plan_item_id=plan_item_id,
                    tool_name=tool_name,
                    operation_type=operation_type,
                    request_payload=request_payload,
                    response_payload=response_payload,
                    result_status=normalized_status,
                    execution_status=normalized_execution_status,
                    error_message=error_message,
                    created_at=cached.created_at if cached is not None else utc_now(),
                ),
            )
            should_clear_pending = clear_pending and normalized_execution_status not in {
                ExecutionStatus.INTENT_LOGGED,
                ExecutionStatus.DISPATCHED,
            }
            if should_clear_pending:
                context = self.context_service.load_context(run_id)
                if context is not None and execution_id in context.pending_execution_ids:
                    pending_after = [value for value in context.pending_execution_ids if value != execution_id]
                    events.append(
                        self._append_event(
                            run_id=run_id,
                            event_type=LoggingEventType.CONTEXT_PENDING_EXECUTION_SET,
                            entity_type="context",
                            entity_id=run_id,
                            payload={
                                "pending_execution_id": pending_after[-1] if pending_after else None,
                                "pending_execution_ids": pending_after,
                            },
                        )
                    )
                    self.context_service.clear_pending_execution(run_id=run_id, execution_id=execution_id)
            checkpoint = self._maybe_auto_checkpoint_locked(run_id, events)
        self._emit_after_commit(run_id, events=events, checkpoint=checkpoint)
        return record

    def advance_plan_item(
        self,
        run_id: str,
        plan_item_id: str,
        new_status: str,
        *,
        note: str | None = None,
    ) -> PlannerState:
        events: list[WalEvent] = []
        checkpoint: WalCheckpoint | None = None
        with self._run_write(run_id):
            events.append(
                self._append_event(
                    run_id=run_id,
                    event_type=LoggingEventType.PLAN_ITEM_STATUS_UPDATED,
                    entity_type="planner_item",
                    entity_id=plan_item_id,
                    payload={"new_status": new_status, "note": note},
                )
            )
            planner_state = self.planner_service.update_item_status(
                run_id=run_id,
                plan_item_id=plan_item_id,
                status=new_status,
                note=note,
            )
            events.append(self._sync_context_with_planner_locked(run_id=run_id, planner_state=planner_state))
            checkpoint = self._maybe_auto_checkpoint_locked(run_id, events)
        self._emit_after_commit(run_id, events=events, checkpoint=checkpoint)
        return planner_state

    def set_active_plan_item(self, run_id: str, plan_item_id: str) -> PlannerState:
        events: list[WalEvent] = []
        checkpoint: WalCheckpoint | None = None
        with self._run_write(run_id):
            events.append(
                self._append_event(
                    run_id=run_id,
                    event_type=LoggingEventType.PLAN_ACTIVE_ITEM_SET,
                    entity_type="planner",
                    entity_id=run_id,
                    payload={"plan_item_id": plan_item_id},
                )
            )
            planner_state = self.planner_service.set_active_item(
                run_id=run_id,
                plan_item_id=plan_item_id,
            )
            events.append(self._sync_context_with_planner_locked(run_id=run_id, planner_state=planner_state))
            checkpoint = self._maybe_auto_checkpoint_locked(run_id, events)
        self._emit_after_commit(run_id, events=events, checkpoint=checkpoint)
        return planner_state

    def checkpoint(
        self,
        run_id: str,
        *,
        snapshot_type: str = "agent_context",
        include_external_results: bool = False,
    ) -> WalCheckpoint | None:
        events: list[WalEvent] = []
        with self._run_write(run_id):
            checkpoint = self._checkpoint_locked(
                run_id,
                snapshot_type=snapshot_type,
                include_external_results=include_external_results,
                events=events,
            )
        self._emit_after_commit(run_id, events=events, checkpoint=checkpoint)
        return checkpoint

    def rebuild_state(
        self,
        run_id: str,
        *,
        snapshot_type: str | None = None,
        event_limit: int = 1000,
    ) -> dict[str, Any]:
        checkpoint = self.wal.read_latest_checkpoint(run_id, snapshot_type=snapshot_type)
        after_seq = checkpoint.seq_id if checkpoint is not None else None
        events = self.wal.read_events(run_id=run_id, after_seq=after_seq, limit=event_limit)
        planner_state = self.planner_service.load_plan(run_id)
        context = self.context_service.load_context(run_id)
        external_results = (
            [record.to_dict() for record in self.context_service.list_external_results(run_id)]
            if context is not None
            else []
        )
        return {
            "run_id": run_id,
            "latest_seq": self.wal.latest_seq(run_id),
            "checkpoint": checkpoint.to_dict() if checkpoint is not None else None,
            "events_after_checkpoint": [event.to_dict() for event in events],
            "planner_state": planner_state.to_dict() if planner_state is not None else None,
            "context": context.to_dict() if context is not None else None,
            "external_results": external_results,
        }

    def get_recovery_bundle(self, run_id: str) -> dict[str, Any]:
        """Alias for recovery protocol consumers."""
        return self.rebuild_state(run_id=run_id)

    def read_run_events(
        self,
        run_id: str,
        *,
        after_seq: int | None = None,
        limit: int = 1000,
    ) -> list[WalEvent]:
        """Expose WAL read API for transaction/recovery tooling."""
        return self.wal.read_events(run_id, after_seq=after_seq, limit=limit)

    def latest_seq(self, run_id: str) -> int | None:
        """Expose latest sequence lookup for external orchestrators."""
        return self.wal.latest_seq(run_id)

    def list_external_results(self, run_id: str) -> list[ExternalResultRecord]:
        return self.context_service.list_external_results(run_id)

    def get_external_result(self, execution_id: str) -> ExternalResultRecord | None:
        return self.context_service.get_external_result(execution_id)

    def list_pending_executions(self, run_id: str) -> list[str]:
        context = self.context_service.load_context(run_id)
        if context is None:
            return []
        return list(context.pending_execution_ids)

    def flush(self) -> None:
        self.store.flush()
        if self.observability is not None:
            self.observability.flush()

    def close(self) -> None:
        if self.observability is not None:
            self.observability.close()

    def _append_event(
        self,
        *,
        run_id: str,
        event_type: LoggingEventType,
        entity_type: str,
        entity_id: str,
        payload: dict[str, Any],
    ) -> WalEvent:
        _validate_event_payload(event_type, payload)
        return self.wal.append_event(
            run_id=run_id,
            event_type=event_type.value,
            entity_type=entity_type,
            entity_id=entity_id,
            payload=payload,
        )

    def _sync_context_with_planner_locked(self, *, run_id: str, planner_state: PlannerState) -> WalEvent:
        context = self.context_service.load_context(run_id)
        if context is None:
            raise ValueError(f"agent context not found for run_id={run_id}")
        event = self._append_event(
            run_id=run_id,
            event_type=LoggingEventType.CONTEXT_SYNCHRONIZED,
            entity_type="context",
            entity_id=run_id,
            payload={
                "planner_version": planner_state.version,
                "cursor_index": planner_state.cursor_index,
            },
        )
        self.context_service.sync_planner_state(
            run_id,
            planner_version=planner_state.version,
            step_index=planner_state.cursor_index,
        )
        return event

    def _maybe_auto_checkpoint_locked(self, run_id: str, events: list[WalEvent]) -> WalCheckpoint | None:
        interval = self.config.checkpoint_every_n_events
        if interval <= 0:
            return None
        seq = self.wal.latest_seq(run_id)
        if seq is None or seq == 0:
            return None
        if seq % interval != 0:
            return None
        return self._checkpoint_locked(run_id, events=events)

    def _checkpoint_locked(
        self,
        run_id: str,
        *,
        snapshot_type: str = "agent_context",
        include_external_results: bool = False,
        events: list[WalEvent] | None = None,
    ) -> WalCheckpoint | None:
        latest_seq = self.wal.latest_seq(run_id)
        if latest_seq is None:
            return None
        planner_state = self.planner_service.load_plan(run_id)
        context_snapshot = self.context_service.load_context(run_id)
        external_results = (
            [record.to_dict() for record in self.context_service.list_external_results(run_id)]
            if include_external_results and context_snapshot is not None
            else None
        )
        snapshot_payload = {
            "planner": planner_state.to_dict() if planner_state is not None else None,
            "context": context_snapshot.to_dict() if context_snapshot is not None else None,
            "external_results": external_results,
        }
        checkpoint = self.wal.write_checkpoint(
            run_id=run_id,
            seq_id=latest_seq,
            snapshot_type=snapshot_type,
            snapshot_payload=snapshot_payload,
        )
        logger.debug(
            "checkpoint written: run_id=%s seq_id=%d snapshot_type=%s",
            run_id,
            latest_seq,
            snapshot_type,
        )
        if context_snapshot is not None:
            self.context_service.checkpoint_context(run_id, last_checkpoint_seq=latest_seq)
        checkpoint_event = self._append_event(
            run_id=run_id,
            event_type=LoggingEventType.CHECKPOINT_WRITTEN,
            entity_type="checkpoint",
            entity_id=checkpoint.checkpoint_id,
            payload={
                "checkpoint_id": checkpoint.checkpoint_id,
                "checkpoint_seq": checkpoint.seq_id,
                "snapshot_type": checkpoint.snapshot_type,
            },
        )
        if events is not None:
            events.append(checkpoint_event)
        return checkpoint

    def _tool_intent_exists_locked(self, run_id: str, execution_id: str) -> bool:
        existing_result = self.context_service.get_external_result(execution_id)
        if existing_result is not None:
            if existing_result.run_id != run_id:
                raise ValueError(
                    f"execution_id={execution_id} already belongs to run_id={existing_result.run_id}"
                )
            return True
        return self.wal.has_event(
            run_id=run_id,
            event_type=LoggingEventType.TOOL_INTENT_RECORDED.value,
            entity_id=execution_id,
        )

    def _assert_execution_transition(
        self,
        current: ExecutionStatus,
        nxt: ExecutionStatus,
    ) -> None:
        if current == nxt:
            return
        allowed = self._ALLOWED_EXECUTION_TRANSITIONS[current]
        if nxt not in allowed:
            raise ValueError(f"illegal execution status transition: {current.value} -> {nxt.value}")

    @contextmanager
    def _run_write(self, run_id: str):
        lock = self._get_run_lock(run_id)
        with lock:
            with self.store.transaction(immediate=True):
                yield

    def _get_run_lock(self, run_id: str) -> threading.RLock:
        with self._run_locks_guard:
            lock = self._run_locks.get(run_id)
            if lock is None:
                lock = threading.RLock()
                self._run_locks[run_id] = lock
            return lock

    def _emit_after_commit(
        self,
        run_id: str,
        *,
        events: list[WalEvent],
        checkpoint: WalCheckpoint | None,
    ) -> None:
        del events, checkpoint
        if self.observability is not None:
            self.observability.record_run(run_id)


def _parse_result_status(raw: str) -> ExternalResultStatus:
    normalized = raw.strip().lower()
    return ExternalResultStatus(normalized)


def _resolve_execution_status(
    *,
    execution_status: str | None,
    result_status: ExternalResultStatus,
) -> ExecutionStatus:
    if execution_status is not None:
        return normalize_execution_status(execution_status)
    if result_status == ExternalResultStatus.OK:
        return ExecutionStatus.ACKED
    if result_status == ExternalResultStatus.ERROR:
        return ExecutionStatus.FAILED
    return ExecutionStatus.DISPATCHED


def _project_pending_ids_with_add(context: Any, execution_id: str) -> list[str]:
    if context is None:
        return [execution_id]
    pending = list(getattr(context, "pending_execution_ids", []))
    if execution_id not in pending:
        pending.append(execution_id)
    return pending


def _validate_event_payload(event_type: LoggingEventType, payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("event payload must be an object")

    required_by_type: dict[LoggingEventType, set[str]] = {
        LoggingEventType.RUN_STARTED: {"initial_message_count", "plan_item_count"},
        LoggingEventType.PLAN_CREATED: {"plan_source_text", "plan_items"},
        LoggingEventType.PLAN_ITEM_LLM_OUTPUT_ATTACHED: {"llm_output"},
        LoggingEventType.PLAN_ITEM_STATUS_UPDATED: {"new_status"},
        LoggingEventType.PLAN_ACTIVE_ITEM_SET: {"plan_item_id"},
        LoggingEventType.CONTEXT_INITIALIZED: {"initial_messages", "initial_memory"},
        LoggingEventType.CONTEXT_MESSAGE_APPENDED: {"role", "content", "meta"},
        LoggingEventType.CONTEXT_PENDING_EXECUTION_SET: {"pending_execution_id", "pending_execution_ids"},
        LoggingEventType.CONTEXT_SYNCHRONIZED: {"planner_version", "cursor_index"},
        LoggingEventType.TOOL_INTENT_RECORDED: {
            "plan_item_id",
            "tool_name",
            "operation_type",
            "request_payload",
            "execution_status",
        },
        LoggingEventType.TOOL_DISPATCHED: {"execution_status", "tool_name"},
        LoggingEventType.EXTERNAL_RESULT_RECORDED: {
            "result_status",
            "execution_status",
            "tool_name",
            "operation_type",
            "has_response",
        },
        LoggingEventType.CHECKPOINT_WRITTEN: {
            "checkpoint_id",
            "checkpoint_seq",
            "snapshot_type",
        },
    }
    missing = [k for k in required_by_type[event_type] if k not in payload]
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"invalid {event_type.value} payload, missing keys: {joined}")
