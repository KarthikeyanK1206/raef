"""Comprehensive tests for planner/context/WAL logging components."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import threading

import pytest

from raef.agent_state import (
    AgentContextService,
    ExternalResultRecord,
    ExternalResultStatus,
    SQLiteAgentStateRepository,
)
from raef.logging_service import LoggingService
from raef.models import ExecutionStatus
from raef.planner_state import (
    PlannerItemStatus,
    PlannerStateService,
    SQLitePlannerStateRepository,
)
from raef.wal import SQLiteWriteAheadLog


def _fetch_one(db_path: Path, sql: str, params: tuple[object, ...] = ()):
    with sqlite3.connect(db_path) as conn:
        return conn.execute(sql, params).fetchone()


def test_planner_state_create_update_and_sqlite_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime.sqlite"
    service = PlannerStateService(SQLitePlannerStateRepository(db_path=db_path))

    state = service.create_plan(
        run_id="run-plan-1",
        plan_source_text="1. inspect\n2. call tool\n3. finalize",
        items=[
            {"title": "inspect environment"},
            {"title": "call tool", "depends_on": ["step_0"]},
            {"title": "finalize", "depends_on": ["step_1"]},
        ],
    )

    assert state.version == 1
    assert state.active_item_id == "step_0"
    assert state.cursor_index == 0

    state = service.attach_llm_output("run-plan-1", "step_0", "Need to inspect first")
    state = service.attach_tool_call_ref("run-plan-1", "step_0", "exec-100")
    state = service.update_item_status("run-plan-1", "step_0", "in_progress")
    state = service.update_item_status("run-plan-1", "step_0", "done", note="completed")

    assert state.version == 5
    row = _fetch_one(db_path, "SELECT version, active_item_id, cursor_index FROM planner_states WHERE run_id = ?", ("run-plan-1",))
    assert row == (5, "step_1", 1)
    item_row = _fetch_one(
        db_path,
        "SELECT status, llm_output FROM planner_items WHERE run_id = ? AND plan_item_id = ?",
        ("run-plan-1", "step_0"),
    )
    assert item_row == (PlannerItemStatus.DONE.value, "Need to inspect first")


def test_planner_state_illegal_transition_and_from_text(tmp_path: Path) -> None:
    service = PlannerStateService(SQLitePlannerStateRepository(db_path=tmp_path / "runtime.sqlite"))

    state = service.create_plan_from_text(
        run_id="run-plan-text",
        plan_source_text="1. first\n2. second\n- third",
    )
    assert len(state.items) == 3
    assert state.items[0].title == "first"

    service.update_item_status("run-plan-text", "step_0", "done")
    with pytest.raises(ValueError):
        service.update_item_status("run-plan-text", "step_0", "in_progress")


def test_agent_context_history_and_inference_views_use_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime.sqlite"
    service = AgentContextService(SQLiteAgentStateRepository(db_path=db_path))

    snapshot = service.init_context(
        run_id="run-ctx-1",
        seed_messages=[{"role": "user", "content": "book flight"}],
        initial_memory={"topic": "travel"},
    )
    assert snapshot.turn_index == 1
    service.append_message("run-ctx-1", role="assistant", content="Checking options")
    service.append_messages(
        "run-ctx-1",
        [
            {"role": "tool", "content": "result: 3 flights", "name": "query-state"},
            {"role": "assistant", "content": "Found 3 flights"},
        ],
    )
    service.set_step_index("run-ctx-1", 2)
    service.set_planner_version("run-ctx-1", 7)
    service.set_pending_execution("run-ctx-1", "exec-ctx-1")
    service.checkpoint_context("run-ctx-1", last_checkpoint_seq=10)

    loaded = service.load_context("run-ctx-1")
    assert loaded is not None
    assert loaded.turn_index == 4
    assert loaded.pending_execution_id == "exec-ctx-1"

    row = _fetch_one(
        db_path,
        "SELECT turn_index, step_index, planner_version, pending_execution_id, last_checkpoint_seq FROM agent_contexts WHERE run_id = ?",
        ("run-ctx-1",),
    )
    assert row == (4, 2, 7, "exec-ctx-1", 10)
    count_row = _fetch_one(db_path, "SELECT COUNT(*) FROM agent_messages WHERE run_id = ?", ("run-ctx-1",))
    assert count_row == (4,)

    recent = service.get_messages_for_inference("run-ctx-1", max_messages=2)
    assert len(recent) == 2


def test_external_results_use_sqlite_payload_artifacts_for_large_payloads(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime.sqlite"
    service = AgentContextService(
        SQLiteAgentStateRepository(db_path=db_path, payload_inline_limit_bytes=64)
    )
    service.init_context(run_id="run-result-1")

    large_payload = {"blob": "x" * 512}
    stored = service.save_external_result(
        "run-result-1",
        ExternalResultRecord(
            execution_id="exec-1",
            run_id="run-result-1",
            plan_item_id="step_1",
            tool_name="query-state",
            operation_type="READ",
            request_payload=large_payload,
            response_payload=large_payload,
            result_status=ExternalResultStatus.OK,
        ),
    )
    assert stored.request_payload_ref is not None
    assert stored.response_payload_ref is not None

    row = _fetch_one(
        db_path,
        "SELECT request_payload_json, response_payload_json, request_payload_ref_id, response_payload_ref_id FROM external_results WHERE execution_id = ?",
        ("exec-1",),
    )
    assert row is not None
    assert row[0] is None and row[1] is None
    assert row[2] is not None and row[3] is not None
    artifact_count = _fetch_one(db_path, "SELECT COUNT(*) FROM payload_artifacts")
    assert artifact_count == (2,)

    loaded = service.get_external_result("exec-1")
    assert loaded is not None
    assert loaded.request_payload == large_payload
    assert loaded.response_payload == large_payload


def test_wal_append_read_latest_and_checkpoint_are_in_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime.sqlite"
    wal = SQLiteWriteAheadLog(db_path)
    wal.init_schema()

    ev1 = wal.append_event(
        run_id="run-wal-1",
        event_type="PLAN_CREATED",
        entity_type="planner",
        entity_id="run-wal-1",
        payload={"items": 3},
    )
    assert ev1.seq_id == 1

    evs = wal.append_events(
        [
            {
                "run_id": "run-wal-1",
                "event_type": "CONTEXT_MESSAGE_APPENDED",
                "entity_type": "context",
                "entity_id": "run-wal-1",
                "payload": {"role": "user"},
            },
            {
                "run_id": "run-wal-1",
                "event_type": "EXTERNAL_RESULT_RECORDED",
                "entity_type": "external_result",
                "entity_id": "exec-1",
                "payload": {"status": "ok"},
            },
        ]
    )
    assert [ev.seq_id for ev in evs] == [2, 3]
    assert [ev.seq_id for ev in wal.read_events("run-wal-1")] == [1, 2, 3]
    assert wal.latest_seq("run-wal-1") == 3

    cp = wal.write_checkpoint(
        run_id="run-wal-1",
        seq_id=3,
        snapshot_type="agent_context",
        snapshot_payload={"turn_index": 2},
    )
    assert cp.seq_id == 3
    row = _fetch_one(db_path, "SELECT COUNT(*) FROM wal_events WHERE run_id = ?", ("run-wal-1",))
    assert row == (3,)
    cp_row = _fetch_one(db_path, "SELECT COUNT(*) FROM wal_checkpoints WHERE run_id = ?", ("run-wal-1",))
    assert cp_row == (1,)


def test_logging_service_is_transactional_and_writes_sqlite_rows(tmp_path: Path) -> None:
    data_root = tmp_path / "runtime"
    service = LoggingService.with_data_root(data_root, soft_write_enabled=False)
    service.start_run(
        run_id="run-log-1",
        initial_messages=[{"role": "user", "content": "set value"}],
        plan_source_text="1. write",
        plan_items=[{"title": "write"}],
    )
    service.record_tool_intent(
        run_id="run-log-1",
        plan_item_id="step_0",
        execution_id="exec-log-1",
        tool_name="apply-action",
        operation_type="WRITE",
        request_payload={"action": "set"},
    )
    service.record_tool_dispatch("run-log-1", "exec-log-1")
    record = service.record_tool_result(
        run_id="run-log-1",
        execution_id="exec-log-1",
        result_status="ok",
        response_payload={"ok": True},
        execution_status=ExecutionStatus.ACKED.value,
    )
    assert record.execution_status == ExecutionStatus.ACKED

    db_path = data_root / "runtime.sqlite"
    ext_row = _fetch_one(
        db_path,
        "SELECT result_status, execution_status FROM external_results WHERE execution_id = ?",
        ("exec-log-1",),
    )
    assert ext_row == ("ok", "acked")
    wal_count = _fetch_one(db_path, "SELECT COUNT(*) FROM wal_events WHERE run_id = ?", ("run-log-1",))
    assert wal_count is not None and wal_count[0] >= 5


def test_logging_service_serializes_multi_threaded_context_updates(tmp_path: Path) -> None:
    service = LoggingService.with_data_root(tmp_path / "concurrency", soft_write_enabled=False)
    run_id = "run-concurrency-1"
    service.start_run(
        run_id=run_id,
        initial_messages=[{"role": "user", "content": "hello"}],
        plan_source_text="1. talk",
        plan_items=[{"title": "talk"}],
    )

    def worker(prefix: str) -> None:
        for index in range(10):
            service.record_context_message(run_id=run_id, role="assistant", content=f"{prefix}-{index}")

    threads = [threading.Thread(target=worker, args=("A",)), threading.Thread(target=worker, args=("B",))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    context = service.context_service.load_context(run_id)
    assert context is not None
    assert context.turn_index == 21
    db_path = tmp_path / "concurrency" / "runtime.sqlite"
    message_count = _fetch_one(db_path, "SELECT COUNT(*) FROM agent_messages WHERE run_id = ?", (run_id,))
    assert message_count == (21,)


def test_soft_json_mirror_flushes_async_observations(tmp_path: Path) -> None:
    data_root = tmp_path / "soft-logs"
    service = LoggingService.with_data_root(data_root, soft_write_enabled=True)
    service.start_run(
        run_id="run-soft-1",
        initial_messages=[{"role": "user", "content": "hello"}],
        plan_source_text="1. one",
        plan_items=[{"title": "one"}],
    )
    service.record_context_message(run_id="run-soft-1", role="assistant", content="working")
    service.flush()

    run_file = data_root / "soft_logs" / "runs" / "run-soft-1.json"
    assert run_file.exists()
    with sqlite3.connect(data_root / "runtime.sqlite") as conn:
        row = conn.execute("SELECT COUNT(*) FROM wal_events WHERE run_id = ?", ("run-soft-1",)).fetchone()
    assert row is not None and row[0] >= 3
    service.close()
