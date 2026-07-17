"""Runtime recovery tests for RAEF."""

from __future__ import annotations

from datetime import timedelta
import time

from raef.adapters.langchain_adaptor import ToolPolicy, build_execution_id, wrap_langchain_tool
from raef.logging_service import LoggingService
from raef.models import ExecutionStatus
from raef.recovery.common import RecoveryAction
from raef.recovery.health.file_checker import FileHealthChecker
from raef.recovery.monitor.pulse_monitor import PulseMonitor
from raef.recovery.recovery.check_helper import recover_with_check_api
from raef.recovery.recovery.handler import RecoveryCoordinator


def _start_run(service: LoggingService, run_id: str) -> None:
    service.start_run(
        run_id=run_id,
        initial_messages=[{"role": "user", "content": "start"}],
        plan_source_text="1. do one thing",
        plan_items=[{"title": "do one thing"}],
    )


def test_recovery_instep_waits_with_default_5s_when_timeout_missing(tmp_path) -> None:
    service = LoggingService.with_data_root(tmp_path / "runtime-default-wait", soft_write_enabled=False)
    run_id = "run-recovery-default-wait"
    _start_run(service, run_id)

    service.record_tool_intent(
        run_id=run_id,
        plan_item_id="step_0",
        execution_id="exec-read-1",
        tool_name="get_balance",
        operation_type="READ",
        request_payload={"account_id": "acct_1"},
    )
    dispatched = service.record_tool_dispatch(run_id=run_id, execution_id="exec-read-1")

    coordinator = RecoveryCoordinator(service, default_wait_seconds=5.0)
    now = dispatched.updated_at + timedelta(seconds=3)
    decisions = coordinator.recover_run(run_id, now=now)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.action == RecoveryAction.RESUME_WAITING
    assert decision.wait_timeout_seconds == 5.0
    assert decision.execution_status == ExecutionStatus.DISPATCHED


def test_read_tool_records_dispatch_for_recovery_classification(tmp_path) -> None:
    service = LoggingService.with_data_root(tmp_path / "runtime-read-dispatch", soft_write_enabled=False)
    run_id = "run-read-dispatch"
    _start_run(service, run_id)

    plan_item_id = "step_0"
    tool_name = "query_state"
    arguments = {"query_name": "dump_state"}
    wrapped = wrap_langchain_tool(
        run_id=run_id,
        plan_item_id=plan_item_id,
        tool_name=tool_name,
        tool_fn=lambda args: {"ok": True, "args": args},
        logging_service=service,
        policy=ToolPolicy(is_read_only=True),
    )

    result = wrapped(arguments)
    assert result["ok"] is True

    execution_id = build_execution_id(
        run_id=run_id,
        plan_item_id=plan_item_id,
        tool_name=tool_name,
        arguments=arguments,
    )
    record = service.context_service.get_external_result(execution_id)
    assert record is not None
    assert record.execution_status == ExecutionStatus.ACKED

    events = service.read_run_events(run_id)
    event_types = [(event.event_type, event.entity_id) for event in events]
    assert ("TOOL_DISPATCHED", execution_id) in event_types


def test_recovery_instep_uses_call_timeout_and_handoffs_after_deadline(tmp_path) -> None:
    service = LoggingService.with_data_root(tmp_path / "runtime-timeout", soft_write_enabled=False)
    run_id = "run-recovery-timeout"
    _start_run(service, run_id)

    service.record_tool_intent(
        run_id=run_id,
        plan_item_id="step_0",
        execution_id="exec-write-1",
        tool_name="create_transfer",
        operation_type="WRITE",
        request_payload={"amount": 10, "timeout_seconds": 2},
    )
    dispatched = service.record_tool_dispatch(run_id=run_id, execution_id="exec-write-1")

    coordinator = RecoveryCoordinator(service, default_wait_seconds=5.0)
    now = dispatched.updated_at + timedelta(seconds=3)
    decisions = coordinator.recover_run(run_id, now=now)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.action == RecoveryAction.HANDOFF_RETRY_OR_ABANDON
    assert decision.wait_timeout_seconds == 2.0
    assert decision.execution_status == ExecutionStatus.DISPATCHED


def test_recovery_replays_intent_only_execution(tmp_path) -> None:
    service = LoggingService.with_data_root(tmp_path / "runtime-intent", soft_write_enabled=False)
    run_id = "run-recovery-intent"
    _start_run(service, run_id)

    service.record_tool_intent(
        run_id=run_id,
        plan_item_id="step_0",
        execution_id="exec-intent-only",
        tool_name="query_state",
        operation_type="READ",
        request_payload={"query_name": "dump_state"},
    )

    coordinator = RecoveryCoordinator(service, default_wait_seconds=5.0)
    decisions = coordinator.recover_run(run_id)

    assert len(decisions) == 1
    assert decisions[0].action == RecoveryAction.REPLAY
    assert decisions[0].execution_status == ExecutionStatus.INTENT_LOGGED


def test_recovery_reconciles_missing_tool_message_for_committed_results_idempotently(tmp_path) -> None:
    service = LoggingService.with_data_root(tmp_path / "runtime-conversation", soft_write_enabled=False)
    run_id = "run-recovery-conversation"
    _start_run(service, run_id)

    service.record_tool_intent(
        run_id=run_id,
        plan_item_id="step_0",
        execution_id="exec-acked",
        tool_name="query_state",
        operation_type="READ",
        request_payload={"query_name": "dump_state"},
    )
    service.record_tool_dispatch(run_id=run_id, execution_id="exec-acked")
    service.record_tool_result(
        run_id=run_id,
        execution_id="exec-acked",
        result_status="ok",
        response_payload={"result": {"balance": 100}},
        execution_status=ExecutionStatus.ACKED.value,
    )

    coordinator = RecoveryCoordinator(service, default_wait_seconds=5.0)
    decisions = coordinator.recover_run(run_id, now=None)
    assert len(decisions) == 1
    assert decisions[0].action == RecoveryAction.MARK_COMMITTED

    context = service.context_service.load_context(run_id)
    assert context is not None
    tool_messages = [m for m in context.messages if m.role == "tool" and m.tool_call_id == "exec-acked"]
    assert len(tool_messages) == 1

    decisions_again = coordinator.recover_run(run_id, now=None)
    assert len(decisions_again) == 1
    assert decisions_again[0].action == RecoveryAction.MARK_COMMITTED
    context_again = service.context_service.load_context(run_id)
    assert context_again is not None
    tool_messages_again = [
        m for m in context_again.messages if m.role == "tool" and m.tool_call_id == "exec-acked"
    ]
    assert len(tool_messages_again) == 1


def test_recovery_handoff_callback_is_deduplicated(tmp_path) -> None:
    service = LoggingService.with_data_root(tmp_path / "runtime-handoff", soft_write_enabled=False)
    run_id = "run-recovery-handoff"
    _start_run(service, run_id)

    service.record_tool_intent(
        run_id=run_id,
        plan_item_id="step_0",
        execution_id="exec-uncertain",
        tool_name="create_transfer",
        operation_type="WRITE",
        request_payload={"amount": 10, "timeout_seconds": 1},
    )
    service.record_tool_dispatch(run_id=run_id, execution_id="exec-uncertain")
    uncertain = service.record_tool_result(
        run_id=run_id,
        execution_id="exec-uncertain",
        result_status="timeout",
        response_payload=None,
        error_message="network timeout",
        execution_status=ExecutionStatus.DISPATCHED.value,
        clear_pending=True,
    )

    handoffs = []
    coordinator = RecoveryCoordinator(
        service,
        default_wait_seconds=5.0,
        on_handoff=lambda decision: handoffs.append(decision.execution_id),
    )
    now = uncertain.updated_at + timedelta(seconds=2)
    coordinator.recover_run(run_id, now=now)
    coordinator.recover_run(run_id, now=now)

    assert handoffs == ["exec-uncertain"]


def test_pulse_monitor_respects_cooldown_and_avoids_repeated_recovery_calls() -> None:
    class FakeChecker:
        def is_alive(self) -> bool:
            return False

    class FakeHandler:
        def __init__(self) -> None:
            self.calls = 0

        def handle(self) -> None:
            self.calls += 1

    clock = {"value": 0.0}

    def time_fn() -> float:
        return clock["value"]

    def sleep_fn(seconds: float) -> None:
        clock["value"] += seconds

    handler = FakeHandler()
    monitor = PulseMonitor(
        FakeChecker(),
        handler,
        poll_interval_seconds=0.1,
        recovery_cooldown_seconds=10.0,
        time_fn=time_fn,
        sleep_fn=sleep_fn,
    )
    monitor.run(max_loops=5)

    assert handler.calls == 1


def test_file_health_checker_probe_mode() -> None:
    checker = FileHealthChecker(probe_fn=lambda: True)
    assert checker.is_alive() is True

    checker_fail = FileHealthChecker(probe_fn=lambda: False)
    assert checker_fail.is_alive() is False


def test_file_health_checker_heartbeat_file_mode(tmp_path) -> None:
    heartbeat = tmp_path / "runner.heartbeat"
    heartbeat.write_text("ok", encoding="utf-8")

    checker = FileHealthChecker(
        heartbeat_file=heartbeat,
        stale_after_seconds=2.0,
        clock_fn=lambda: time.time(),
    )
    assert checker.is_alive() is True

    stale_time = heartbeat.stat().st_mtime + 10
    stale_checker = FileHealthChecker(
        heartbeat_file=heartbeat,
        stale_after_seconds=2.0,
        clock_fn=lambda: stale_time,
    )
    assert stale_checker.is_alive() is False


def test_recover_with_check_api_reuses_found_result() -> None:
    request = {"execution_id": "exec-check-hit-1", "payload": {"k": "v"}}
    captured_payload = {}

    def call_check_api(payload: dict[str, object]) -> dict[str, object]:
        captured_payload.update(payload)
        return {
            "found": True,
            "record": {
                "receipt": {"execution_id": "exec-check-hit-1", "status": "committed"},
            },
        }

    result = recover_with_check_api(
        request=request,
        check_request_payload={"scope": "write_status"},
        call_check_api=call_check_api,
    )

    assert captured_payload["execution_id"] == "exec-check-hit-1"
    assert captured_payload["scope"] == "write_status"
    assert result.should_retry is False
    assert result.found_previous_result is True
    assert result.recovered_response_payload == {"execution_id": "exec-check-hit-1", "status": "committed"}


def test_recover_with_check_api_returns_retry_when_not_found() -> None:
    result = recover_with_check_api(
        request={"execution_id": "exec-check-miss-1"},
        check_request_payload={"query": "status"},
        call_check_api=lambda _payload: {"found": False, "record": None},
    )

    assert result.should_retry is True
    assert result.found_previous_result is False
    assert result.recovered_response_payload is None


def test_recover_with_check_api_returns_retry_when_check_call_fails() -> None:
    result = recover_with_check_api(
        request={"execution_id": "exec-check-error-1"},
        check_request_payload={"query": "status"},
        call_check_api=lambda _payload: (_ for _ in ()).throw(RuntimeError("temporary network error")),
    )

    assert result.should_retry is True
    assert result.found_previous_result is False
    assert "check api call failed" in result.reason
