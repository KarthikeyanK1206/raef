"""Tests for the raef operator CLI."""

from __future__ import annotations

import json

from raef.cli import main
from raef.logging_service import LoggingService
from raef.models import ExecutionStatus
from raef.txn_manager import TransactionManager


def _seed_run(tmp_path, *, run_id: str = "run-cli") -> str:
    service = LoggingService.with_data_root(tmp_path, soft_write_enabled=False)
    service.start_run(
        run_id=run_id,
        initial_messages=[{"role": "user", "content": "start"}],
        plan_source_text="1. do one thing",
        plan_items=[{"title": "do one thing"}],
    )
    TransactionManager(service).execute_callable(
        run_id=run_id,
        plan_item_id="step_0",
        tool_name="book",
        request_payload={"flight": "RA512"},
        operation_type="WRITE",
        tool_fn=lambda args: {"ok": True},
    )
    service.advance_plan_item(run_id=run_id, plan_item_id="step_0", new_status="done")
    return run_id


def test_missing_store_exits_with_error(tmp_path, capsys) -> None:
    exit_code = main(["--data-root", str(tmp_path / "nowhere"), "runs"])
    assert exit_code == 2
    assert "no runtime store" in capsys.readouterr().err


def test_runs_lists_seeded_run(tmp_path, capsys) -> None:
    run_id = _seed_run(tmp_path)
    exit_code = main(["--data-root", str(tmp_path), "runs"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert run_id in output
    assert "pending=0" in output


def test_show_prints_plan_and_pending(tmp_path, capsys) -> None:
    run_id = _seed_run(tmp_path)
    exit_code = main(["--data-root", str(tmp_path), "show", run_id])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "do one thing" in output
    assert "pending executions: none" in output


def test_executions_json_reports_acked_write(tmp_path, capsys) -> None:
    run_id = _seed_run(tmp_path)
    exit_code = main(["--data-root", str(tmp_path), "--json", "executions", run_id])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert len(payload) == 1
    assert payload[0]["tool_name"] == "book"
    assert payload[0]["execution_status"] == ExecutionStatus.ACKED.value


def test_events_prints_wal_stream(tmp_path, capsys) -> None:
    run_id = _seed_run(tmp_path)
    exit_code = main(["--data-root", str(tmp_path), "events", run_id])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "RUN_STARTED" in output
    assert "TOOL_INTENT_RECORDED" in output
    assert "EXTERNAL_RESULT_RECORDED" in output


def test_report_prints_step_attempts(tmp_path, capsys) -> None:
    run_id = _seed_run(tmp_path)
    exit_code = main(["--data-root", str(tmp_path), "report", run_id])
    assert exit_code == 0
    assert "spans:" in capsys.readouterr().out


def test_recover_dry_run_reports_committed_execution(tmp_path, capsys) -> None:
    run_id = _seed_run(tmp_path)
    exit_code = main(["--data-root", str(tmp_path), "--json", "recover", run_id, "--dry-run"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert len(payload) == 1
    assert payload[0]["action"] == "mark_committed"


def test_audit_passes_on_clean_run(tmp_path, capsys) -> None:
    run_id = _seed_run(tmp_path)
    exit_code = main(["--data-root", str(tmp_path), "audit", run_id])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "audit ok" in output


def test_audit_flags_dangling_pending_execution(tmp_path, capsys) -> None:
    run_id = _seed_run(tmp_path)
    service = LoggingService.with_data_root(tmp_path, soft_write_enabled=False)
    # Corrupt the invariant deliberately: pending id without a durable record.
    service.context_service.add_pending_execution(run_id, "exec-ghost")

    exit_code = main(["--data-root", str(tmp_path), "audit", run_id])
    output = capsys.readouterr().out
    assert exit_code == 1
    assert "VIOLATION" in output
    assert "exec-ghost" in output
