"""Transaction manager tests: log-before-send, replay reuse, and ambiguity."""

from __future__ import annotations

from typing import Any

import pytest

from raef.logging_service import LoggingService
from raef.models import ExecutionStatus
from raef.txn_manager import (
    AmbiguousToolError,
    TransactionDisposition,
    TransactionManager,
    build_execution_id,
)


def _service(tmp_path, name: str) -> LoggingService:
    return LoggingService.with_data_root(tmp_path / name, soft_write_enabled=False)


def _start_run(service: LoggingService, run_id: str) -> None:
    service.start_run(
        run_id=run_id,
        initial_messages=[{"role": "user", "content": "start"}],
        plan_source_text="1. do one thing",
        plan_items=[{"title": "do one thing"}],
    )


class CountingTool:
    """Tool double that counts invocations and can fail on demand."""

    def __init__(self, *, raises: Exception | None = None, raise_once: bool = False) -> None:
        self.calls = 0
        self.raises = raises
        self.raise_once = raise_once

    def __call__(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        if self.raises is not None:
            exc = self.raises
            if self.raise_once:
                self.raises = None
            raise exc
        return {"ok": True, "echo": arguments}


def test_execution_id_is_deterministic_and_argument_sensitive() -> None:
    base = dict(run_id="r", plan_item_id="s0", tool_name="t")
    first = build_execution_id(**base, arguments={"a": 1, "b": 2})
    second = build_execution_id(**base, arguments={"b": 2, "a": 1})
    different = build_execution_id(**base, arguments={"a": 1, "b": 3})
    keyed = build_execution_id(**base, arguments={"a": 1, "b": 2}, invocation_key="retry-1")

    assert first == second  # canonical JSON makes key order irrelevant
    assert first != different
    assert first != keyed


def test_successful_write_is_acked_and_replay_is_reused(tmp_path) -> None:
    service = _service(tmp_path, "txn-success")
    run_id = "run-txn-success"
    _start_run(service, run_id)
    manager = TransactionManager(service)
    tool = CountingTool()

    first = manager.execute_callable(
        run_id=run_id,
        plan_item_id="step_0",
        tool_name="book",
        request_payload={"flight": "RA512"},
        operation_type="WRITE",
        tool_fn=tool,
    )
    assert first.disposition == TransactionDisposition.SUCCEEDED
    assert first.execution_status == ExecutionStatus.ACKED
    assert tool.calls == 1
    assert service.list_pending_executions(run_id) == []

    # A second manager instance simulates a restarted process replaying the
    # same logical call: the durable result is reused, the tool is not rerun.
    replay = TransactionManager(service).execute_callable(
        run_id=run_id,
        plan_item_id="step_0",
        tool_name="book",
        request_payload={"flight": "RA512"},
        operation_type="WRITE",
        tool_fn=tool,
    )
    assert replay.disposition == TransactionDisposition.REUSED
    assert replay.execution_id == first.execution_id
    assert replay.response_payload == first.response_payload
    assert tool.calls == 1


def test_failed_tool_records_error_and_stays_failed(tmp_path) -> None:
    service = _service(tmp_path, "txn-failed")
    run_id = "run-txn-failed"
    _start_run(service, run_id)
    manager = TransactionManager(service)
    tool = CountingTool(raises=ValueError("boom"))

    result = manager.execute_callable(
        run_id=run_id,
        plan_item_id="step_0",
        tool_name="book",
        request_payload={"flight": "RA512"},
        operation_type="WRITE",
        tool_fn=tool,
    )
    assert result.disposition == TransactionDisposition.FAILED
    assert result.execution_status == ExecutionStatus.FAILED
    assert result.record.error_message == "boom"

    replay = manager.execute_callable(
        run_id=run_id,
        plan_item_id="step_0",
        tool_name="book",
        request_payload={"flight": "RA512"},
        operation_type="WRITE",
        tool_fn=tool,
    )
    assert replay.disposition == TransactionDisposition.FAILED
    assert tool.calls == 1  # failed executions are not silently retried


def test_ambiguous_write_is_marked_pending_recovery(tmp_path) -> None:
    service = _service(tmp_path, "txn-ambiguous")
    run_id = "run-txn-ambiguous"
    _start_run(service, run_id)
    manager = TransactionManager(service)
    tool = CountingTool(raises=AmbiguousToolError("response lost"))

    result = manager.execute_callable(
        run_id=run_id,
        plan_item_id="step_0",
        tool_name="book",
        request_payload={"flight": "RA512"},
        operation_type="WRITE",
        tool_fn=tool,
    )
    assert result.disposition == TransactionDisposition.PENDING_RECOVERY
    assert result.execution_status == ExecutionStatus.DISPATCHED
    assert result.record.result_status.value == "timeout"
    # The execution stays pending: it is not safe to drop or retry blindly.
    assert service.list_pending_executions(run_id) == [result.execution_id]

    replay = manager.execute_callable(
        run_id=run_id,
        plan_item_id="step_0",
        tool_name="book",
        request_payload={"flight": "RA512"},
        operation_type="WRITE",
        tool_fn=tool,
    )
    assert replay.disposition == TransactionDisposition.PENDING_RECOVERY
    assert tool.calls == 1  # never blindly re-executed


def test_ambiguous_read_fails_instead_of_pending(tmp_path) -> None:
    service = _service(tmp_path, "txn-ambiguous-read")
    run_id = "run-txn-ambiguous-read"
    _start_run(service, run_id)
    manager = TransactionManager(service)
    tool = CountingTool(raises=AmbiguousToolError("read timeout"))

    result = manager.execute_callable(
        run_id=run_id,
        plan_item_id="step_0",
        tool_name="lookup",
        request_payload={"key": "k"},
        operation_type="READ",
        tool_fn=tool,
    )
    assert result.disposition == TransactionDisposition.FAILED
    assert result.execution_status == ExecutionStatus.FAILED


def test_verified_not_found_execution_is_replayed(tmp_path) -> None:
    service = _service(tmp_path, "txn-replay")
    run_id = "run-txn-replay"
    _start_run(service, run_id)
    manager = TransactionManager(service)
    tool = CountingTool(raises=AmbiguousToolError("request lost"), raise_once=True)

    pending = manager.execute_callable(
        run_id=run_id,
        plan_item_id="step_0",
        tool_name="book",
        request_payload={"flight": "RA512"},
        operation_type="WRITE",
        tool_fn=tool,
    )
    assert pending.disposition == TransactionDisposition.PENDING_RECOVERY

    # Recovery (e.g. via a verifier) proves the write never landed.
    service.record_tool_result(
        run_id=run_id,
        execution_id=pending.execution_id,
        result_status="unknown",
        response_payload=None,
        execution_status=ExecutionStatus.VERIFIED_NOT_FOUND.value,
        clear_pending=False,
    )

    replay = manager.execute_callable(
        run_id=run_id,
        plan_item_id="step_0",
        tool_name="book",
        request_payload={"flight": "RA512"},
        operation_type="WRITE",
        tool_fn=tool,
    )
    assert replay.disposition == TransactionDisposition.SUCCEEDED
    assert replay.execution_status == ExecutionStatus.ACKED
    assert tool.calls == 2  # one lost attempt, one verified-safe replay
    assert service.list_pending_executions(run_id) == []


def test_non_dict_tool_output_fails_transaction(tmp_path) -> None:
    service = _service(tmp_path, "txn-bad-output")
    run_id = "run-txn-bad-output"
    _start_run(service, run_id)
    manager = TransactionManager(service)

    result = manager.execute_callable(
        run_id=run_id,
        plan_item_id="step_0",
        tool_name="bad",
        request_payload={"k": "v"},
        operation_type="WRITE",
        tool_fn=lambda args: "not a dict",  # type: ignore[arg-type, return-value]
    )
    assert result.disposition == TransactionDisposition.FAILED
    assert isinstance(result.exception, ValueError)


def test_execution_id_cannot_be_shared_across_runs(tmp_path) -> None:
    service = _service(tmp_path, "txn-cross-run")
    _start_run(service, "run-a")
    _start_run(service, "run-b")

    service.record_tool_intent(
        run_id="run-a",
        plan_item_id="step_0",
        execution_id="exec-shared",
        tool_name="book",
        operation_type="WRITE",
        request_payload={"k": "v"},
    )
    with pytest.raises(ValueError):
        service.record_tool_intent(
            run_id="run-b",
            plan_item_id="step_0",
            execution_id="exec-shared",
            tool_name="book",
            operation_type="WRITE",
            request_payload={"k": "v"},
        )
