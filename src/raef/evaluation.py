"""Durable performance evaluation helpers for RAEF runs.

The evaluation recorder stores timing spans in SQLite before work begins.
If the process stops mid-step, the unfinished span remains durable and the
report can classify that attempt as interrupted.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime_store import SQLiteRuntimeStore
from .utils import durable_write_json


@dataclass(frozen=True, slots=True)
class EvaluationSpanHandle:
    """Handle returned when a durable timing span is started."""

    span_id: str
    run_id: str
    step_index: int | None
    plan_item_id: str | None
    attempt_no: int
    phase: str
    parent_span_id: str | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "EvaluationSpanHandle":
        return cls(
            span_id=str(record["span_id"]),
            run_id=str(record["run_id"]),
            step_index=record.get("step_index"),
            plan_item_id=record.get("plan_item_id"),
            attempt_no=int(record["attempt_no"]),
            phase=str(record["phase"]),
            parent_span_id=record.get("parent_span_id"),
        )


class EvaluationRecorder:
    """Records durable step and phase timing spans for one RAEF runtime."""

    def __init__(self, runtime: Any) -> None:
        self.store = _resolve_store(runtime)

    def start_step(
        self,
        *,
        run_id: str,
        step_index: int,
        plan_item_id: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationSpanHandle:
        """Start a new durable attempt for one agent step."""

        attempt_no = self.store.next_evaluation_attempt_no(
            run_id,
            step_index=step_index,
            plan_item_id=plan_item_id,
        )
        record = self.store.start_evaluation_span(
            run_id=run_id,
            step_index=step_index,
            plan_item_id=plan_item_id,
            attempt_no=attempt_no,
            phase="step_total",
            metadata=metadata,
        )
        return EvaluationSpanHandle.from_record(record)

    def start_phase(
        self,
        *,
        phase: str,
        parent_step: EvaluationSpanHandle,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationSpanHandle:
        """Start a child phase span under a step attempt."""

        record = self.store.start_evaluation_span(
            run_id=parent_step.run_id,
            step_index=parent_step.step_index,
            plan_item_id=parent_step.plan_item_id,
            attempt_no=parent_step.attempt_no,
            phase=phase,
            parent_span_id=parent_step.span_id,
            metadata=metadata,
        )
        return EvaluationSpanHandle.from_record(record)

    def finish_span(
        self,
        span: EvaluationSpanHandle | str,
        *,
        status: str = "succeeded",
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Finish a durable span. Re-finishing an already closed span is a no-op."""

        span_id = span.span_id if isinstance(span, EvaluationSpanHandle) else span
        return self.store.finish_evaluation_span(
            span_id,
            status=status,
            error_message=error_message,
            metadata=metadata,
        )

    @contextmanager
    def time_phase(
        self,
        *,
        phase: str,
        parent_step: EvaluationSpanHandle,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[EvaluationSpanHandle]:
        """Context manager for timing a child phase under a step attempt."""

        span = self.start_phase(phase=phase, parent_step=parent_step, metadata=metadata)
        try:
            yield span
        except BaseException as exc:
            self.finish_span(
                span,
                status=_status_for_exception(exc),
                error_message=str(exc),
                metadata={"exception_type": type(exc).__name__},
            )
            raise
        else:
            self.finish_span(span, status="succeeded")

    @contextmanager
    def time_step(
        self,
        *,
        run_id: str,
        step_index: int,
        plan_item_id: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[EvaluationSpanHandle]:
        """Context manager for timing a complete agent step attempt."""

        span = self.start_step(
            run_id=run_id,
            step_index=step_index,
            plan_item_id=plan_item_id,
            metadata=metadata,
        )
        try:
            yield span
        except BaseException as exc:
            self.finish_span(
                span,
                status=_status_for_exception(exc),
                error_message=str(exc),
                metadata={"exception_type": type(exc).__name__},
            )
            raise
        else:
            self.finish_span(span, status="succeeded")

    def build_report(
        self,
        run_id: str,
        *,
        mark_open_as_interrupted: bool = True,
    ) -> dict[str, Any]:
        return build_evaluation_report(
            self.store,
            run_id,
            mark_open_as_interrupted=mark_open_as_interrupted,
        )

    def write_report_json(
        self,
        run_id: str,
        output_path: str | Path,
        *,
        mark_open_as_interrupted: bool = True,
    ) -> dict[str, Any]:
        report = self.build_report(
            run_id,
            mark_open_as_interrupted=mark_open_as_interrupted,
        )
        durable_write_json(Path(output_path), report)
        return report


def build_evaluation_report(
    runtime: Any,
    run_id: str,
    *,
    mark_open_as_interrupted: bool = True,
) -> dict[str, Any]:
    """Build a step/attempt timing report from durable evaluation spans."""

    store = _resolve_store(runtime)
    spans = store.list_evaluation_spans(run_id)
    step_spans = [span for span in spans if span["phase"] == "step_total"]
    child_spans: dict[str, list[dict[str, Any]]] = defaultdict(list)
    orphan_phases: list[dict[str, Any]] = []

    for span in spans:
        if span["phase"] == "step_total":
            continue
        parent_span_id = span.get("parent_span_id")
        if isinstance(parent_span_id, str) and parent_span_id:
            child_spans[parent_span_id].append(span)
        else:
            orphan_phases.append(span)

    steps_by_key: dict[tuple[int | None, str | None], dict[str, Any]] = {}
    for step_span in sorted(step_spans, key=_span_sort_key):
        key = (step_span.get("step_index"), step_span.get("plan_item_id"))
        step = steps_by_key.setdefault(
            key,
            {
                "step_index": step_span.get("step_index"),
                "plan_item_id": step_span.get("plan_item_id"),
                "attempts": [],
            },
        )
        phase_records = [
            _span_report_record(phase, mark_open_as_interrupted=mark_open_as_interrupted)
            for phase in sorted(child_spans.get(step_span["span_id"], []), key=_span_sort_key)
        ]
        attempt = _span_report_record(
            step_span,
            mark_open_as_interrupted=mark_open_as_interrupted,
        )
        attempt["phases"] = phase_records
        step["attempts"].append(attempt)

    completed_step_duration_ms = sum(
        span["duration_ms"] or 0.0
        for span in step_spans
        if span.get("finished_at") is not None
    )
    completed_phase_duration_ms: dict[str, float] = defaultdict(float)
    for span in spans:
        if span["phase"] == "step_total" or span.get("duration_ms") is None:
            continue
        completed_phase_duration_ms[span["phase"]] += float(span["duration_ms"])

    return {
        "run_id": run_id,
        "span_count": len(spans),
        "completed_step_duration_ms": completed_step_duration_ms,
        "completed_phase_duration_ms": dict(sorted(completed_phase_duration_ms.items())),
        "steps": sorted(
            steps_by_key.values(),
            key=lambda item: (
                item["step_index"] if item["step_index"] is not None else -1,
                item["plan_item_id"] or "",
            ),
        ),
        "orphan_phases": [
            _span_report_record(phase, mark_open_as_interrupted=mark_open_as_interrupted)
            for phase in sorted(orphan_phases, key=_span_sort_key)
        ],
    }


def _resolve_store(runtime: Any) -> SQLiteRuntimeStore:
    store = getattr(runtime, "store", runtime)
    if not isinstance(store, SQLiteRuntimeStore):
        raise TypeError("runtime must be a LoggingService or SQLiteRuntimeStore")
    return store


def _span_report_record(
    span: dict[str, Any],
    *,
    mark_open_as_interrupted: bool,
) -> dict[str, Any]:
    status = str(span["status"])
    if span.get("finished_at") is None and status == "running" and mark_open_as_interrupted:
        status = "interrupted"
    return {
        "span_id": span["span_id"],
        "attempt_no": span["attempt_no"],
        "phase": span["phase"],
        "status": status,
        "recorded_status": span["status"],
        "started_at": span["started_at"],
        "finished_at": span["finished_at"],
        "duration_ms": span["duration_ms"],
        "metadata": span["metadata"],
        "error_message": span["error_message"],
    }


def _span_sort_key(span: dict[str, Any]) -> tuple[int, int, str, str]:
    return (
        int(span["step_index"]) if span.get("step_index") is not None else -1,
        int(span["attempt_no"]),
        str(span["started_at"]),
        str(span["span_id"]),
    )


def _status_for_exception(exc: BaseException) -> str:
    if isinstance(exc, (KeyboardInterrupt, SystemExit)) or type(exc).__name__ == "SimulatedCrash":
        return "interrupted"
    return "failed"
