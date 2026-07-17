"""Tests for durable evaluation timing spans."""

from __future__ import annotations

from pathlib import Path

from raef.evaluation import EvaluationRecorder, build_evaluation_report
from raef.logging_service import LoggingService


def _start_run(service: LoggingService, run_id: str) -> None:
    service.start_run(
        run_id=run_id,
        initial_messages=[{"role": "user", "content": "start"}],
        plan_source_text="1. do one thing",
        plan_items=[{"title": "do one thing"}],
    )


def test_evaluation_recorder_records_step_and_phase_durations(tmp_path: Path) -> None:
    service = LoggingService.with_data_root(tmp_path / "runtime-eval", soft_write_enabled=False)
    run_id = "run-eval-success"
    _start_run(service, run_id)

    evaluator = EvaluationRecorder(service)
    with evaluator.time_step(run_id=run_id, step_index=0, plan_item_id="step_0") as step:
        with evaluator.time_phase(phase="llm_generate", parent_step=step):
            pass
        with evaluator.time_phase(phase="tool_transaction", parent_step=step, metadata={"tool_name": "noop"}):
            pass

    report = evaluator.build_report(run_id)
    assert report["span_count"] == 3
    assert len(report["steps"]) == 1

    attempt = report["steps"][0]["attempts"][0]
    assert attempt["status"] == "succeeded"
    assert attempt["duration_ms"] is not None
    assert [phase["phase"] for phase in attempt["phases"]] == [
        "llm_generate",
        "tool_transaction",
    ]
    assert all(phase["status"] == "succeeded" for phase in attempt["phases"])


def test_evaluation_report_marks_open_attempt_interrupted_after_restart(tmp_path: Path) -> None:
    data_root = tmp_path / "runtime-eval-restart"
    service = LoggingService.with_data_root(data_root, soft_write_enabled=False)
    run_id = "run-eval-interrupted"
    _start_run(service, run_id)

    evaluator = EvaluationRecorder(service)
    interrupted = evaluator.start_step(run_id=run_id, step_index=0, plan_item_id="step_0")
    assert interrupted.attempt_no == 1

    restarted_service = LoggingService.with_data_root(data_root, soft_write_enabled=False)
    restarted_evaluator = EvaluationRecorder(restarted_service)
    recovered = restarted_evaluator.start_step(run_id=run_id, step_index=0, plan_item_id="step_0")
    assert recovered.attempt_no == 2
    restarted_evaluator.finish_span(recovered)

    report = build_evaluation_report(restarted_service, run_id)
    attempts = report["steps"][0]["attempts"]
    assert [attempt["attempt_no"] for attempt in attempts] == [1, 2]
    assert [attempt["status"] for attempt in attempts] == ["interrupted", "succeeded"]
    assert attempts[0]["finished_at"] is None
    assert attempts[1]["duration_ms"] is not None
