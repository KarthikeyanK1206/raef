"""End-to-end tests for verification-driven recovery of ambiguous writes.

Two directions of ambiguity are covered:
- response lost: the write committed at the target, the caller never saw the
  ack. Recovery must prove the commit and reuse it (no double booking).
- request lost: the write never reached the target. Recovery must prove the
  absence and replay exactly one more time.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from raef.logging_service import LoggingService
from raef.models import ExecutionStatus
from raef.recovery.common import RecoveryAction
from raef.recovery.recovery.handler import RecoveryCoordinator
from raef.tools.mock_target import IdempotencyMode, JsonKVStore, MockTargetService
from raef.txn_manager import (
    AmbiguousToolError,
    TransactionDisposition,
    TransactionManager,
)
from raef.verifier import MockTargetVerifier

BOOKING_PAYLOAD = {
    "action_name": "set_value",
    "payload": {"key": "orders/RA512", "value": {"flight_no": "RA512", "status": "booked"}},
}


class FaultyTargetAdapter:
    """Adapter that injects one ambiguous failure against a real mock target."""

    def __init__(self, target: MockTargetService, *, lose_request: bool) -> None:
        self.target = target
        self.lose_request = lose_request
        self._fault_pending = True

    def invoke(
        self,
        *,
        tool_name: str,
        request_payload: dict[str, Any],
        execution_id: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        del tool_name, timeout_s
        if self.lose_request and self._fault_pending:
            self._fault_pending = False
            raise AmbiguousToolError("request lost before reaching target")
        result = self.target.apply_action(
            action_name=str(request_payload["action_name"]),
            payload=dict(request_payload["payload"]),
            execution_id=execution_id,
        )
        if not self.lose_request and self._fault_pending:
            self._fault_pending = False
            raise AmbiguousToolError("response lost after target committed")
        return result


def _setup(tmp_path, name: str) -> tuple[LoggingService, MockTargetService, TransactionManager]:
    service = LoggingService.with_data_root(tmp_path / name, soft_write_enabled=False)
    service.start_run(
        run_id=f"run-{name}",
        initial_messages=[{"role": "user", "content": "book a ticket"}],
        plan_source_text="1. Book the selected flight",
        plan_items=[{"title": "Book the selected flight"}],
    )
    target = MockTargetService(
        JsonKVStore(tmp_path / name / "target.json"),
        idempotency_mode=IdempotencyMode.IDEMPOTENT,
    )
    return service, target, TransactionManager(service)


def _execute_booking(manager: TransactionManager, run_id: str, adapter: Any):
    return manager.execute_tool(
        run_id=run_id,
        plan_item_id="step_0",
        tool_name="apply-action",
        request_payload=dict(BOOKING_PAYLOAD),
        operation_type="WRITE",
        adapter=adapter,
        idempotency_supported=True,
    )


def test_committed_ambiguous_write_is_verified_and_reused(tmp_path) -> None:
    service, target, manager = _setup(tmp_path, "verify-committed")
    run_id = "run-verify-committed"
    adapter = FaultyTargetAdapter(target, lose_request=False)

    result = _execute_booking(manager, run_id, adapter)
    assert result.disposition == TransactionDisposition.PENDING_RECOVERY

    coordinator = RecoveryCoordinator(
        service,
        default_wait_seconds=0.25,
        verifier=MockTargetVerifier(target),
    )
    now = result.record.updated_at + timedelta(seconds=1)
    decisions = coordinator.recover_run(run_id, now=now)

    assert [d.action for d in decisions] == [RecoveryAction.MARK_COMMITTED]
    record = service.get_external_result(result.execution_id)
    assert record is not None
    assert record.execution_status == ExecutionStatus.VERIFIED_COMMITTED
    assert record.response_payload is not None and record.response_payload.get("verified") is True
    assert service.list_pending_executions(run_id) == []

    # The committed result is reconciled into the durable transcript exactly once.
    context = service.context_service.load_context(run_id)
    assert context is not None
    tool_messages = [m for m in context.messages if m.tool_call_id == result.execution_id]
    assert len(tool_messages) == 1

    # Crucially: the target committed exactly once (no double booking).
    stats = target.store.load()["stats"]
    assert stats["total_commits"] == 1
    assert target.query_state("get_value", {"key": "orders/RA512"})["result"] == {
        "flight_no": "RA512",
        "status": "booked",
    }


def test_lost_ambiguous_write_is_verified_then_replayed(tmp_path) -> None:
    service, target, manager = _setup(tmp_path, "verify-lost")
    run_id = "run-verify-lost"
    adapter = FaultyTargetAdapter(target, lose_request=True)

    result = _execute_booking(manager, run_id, adapter)
    assert result.disposition == TransactionDisposition.PENDING_RECOVERY

    coordinator = RecoveryCoordinator(
        service,
        default_wait_seconds=0.25,
        verifier=MockTargetVerifier(target),
    )
    now = result.record.updated_at + timedelta(seconds=1)
    decisions = coordinator.recover_run(run_id, now=now)

    assert [d.action for d in decisions] == [RecoveryAction.REPLAY]
    record = service.get_external_result(result.execution_id)
    assert record is not None
    assert record.execution_status == ExecutionStatus.VERIFIED_NOT_FOUND
    # Still pending: nothing committed yet, replay has not happened.
    assert service.list_pending_executions(run_id) == [result.execution_id]

    replayed = _execute_booking(manager, run_id, adapter)
    assert replayed.disposition == TransactionDisposition.SUCCEEDED
    assert replayed.execution_status == ExecutionStatus.ACKED
    assert service.list_pending_executions(run_id) == []

    stats = target.store.load()["stats"]
    assert stats["total_commits"] == 1
    assert target.query_state("get_value", {"key": "orders/RA512"})["result"] == {
        "flight_no": "RA512",
        "status": "booked",
    }


def test_recovery_without_verifier_still_hands_off(tmp_path) -> None:
    service, target, manager = _setup(tmp_path, "verify-none")
    run_id = "run-verify-none"
    adapter = FaultyTargetAdapter(target, lose_request=False)

    result = _execute_booking(manager, run_id, adapter)
    assert result.disposition == TransactionDisposition.PENDING_RECOVERY

    handoffs: list[Any] = []
    coordinator = RecoveryCoordinator(
        service,
        default_wait_seconds=0.25,
        on_handoff=handoffs.append,
    )
    now = result.record.updated_at + timedelta(seconds=1)
    decisions = coordinator.recover_run(run_id, now=now)

    assert [d.action for d in decisions] == [RecoveryAction.HANDOFF_RETRY_OR_ABANDON]
    assert len(handoffs) == 1
    record = service.get_external_result(result.execution_id)
    assert record is not None
    assert record.execution_status == ExecutionStatus.DISPATCHED
