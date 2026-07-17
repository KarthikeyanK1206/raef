"""Operator CLI for inspecting and recovering RAEF runtime stores.

Every subcommand reads the SQLite runtime store under ``--data-root`` (the
same directory passed to ``LoggingService.with_data_root``). Inspection
commands never mutate state; ``recover`` applies recovery decisions unless
``--dry-run`` is given.

Examples:
    raef --data-root ./data/openrouter_mock_crash_e2e runs
    raef --data-root ./data/openrouter_mock_crash_e2e show <run_id>
    raef --data-root ./data/openrouter_mock_crash_e2e executions <run_id>
    raef --data-root ./data/openrouter_mock_crash_e2e recover <run_id> --dry-run
    raef --data-root ./data/openrouter_mock_crash_e2e audit <run_id>
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any

from .evaluation import build_evaluation_report
from .logging_service import LoggingService
from .models import ExecutionStatus
from .recovery.recovery.handler import RecoveryCoordinator
from .recovery.recovery.strategy import RuntimeRecoveryStrategy

_TERMINAL_EXECUTION_STATUSES = {
    ExecutionStatus.ACKED,
    ExecutionStatus.VERIFIED_COMMITTED,
    ExecutionStatus.FAILED,
}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    data_root = Path(args.data_root)
    db_path = data_root / "runtime.sqlite"
    if not db_path.exists():
        print(f"error: no runtime store at {db_path}", file=sys.stderr)
        return 2

    service = LoggingService.with_data_root(data_root, soft_write_enabled=False)
    try:
        handler = _COMMANDS[args.command]
        return handler(service, args)
    finally:
        service.close()


def _cmd_runs(service: LoggingService, args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    for run_id in service.store.list_run_ids():
        planner = service.planner_service.load_plan(run_id)
        statuses: dict[str, int] = {}
        if planner is not None:
            for item in planner.items:
                statuses[item.status.value] = statuses.get(item.status.value, 0) + 1
        rows.append(
            {
                "run_id": run_id,
                "latest_seq": service.latest_seq(run_id),
                "pending_executions": len(service.list_pending_executions(run_id)),
                "plan_item_statuses": statuses,
            }
        )
    if args.json:
        _print_json(rows)
        return 0
    if not rows:
        print("no runs found")
        return 0
    for row in rows:
        status_text = ", ".join(f"{k}={v}" for k, v in sorted(row["plan_item_statuses"].items())) or "-"
        print(
            f"{row['run_id']}  seq={row['latest_seq']}  "
            f"pending={row['pending_executions']}  plan[{status_text}]"
        )
    return 0


def _cmd_show(service: LoggingService, args: argparse.Namespace) -> int:
    bundle = service.get_recovery_bundle(args.run_id)
    if bundle["planner_state"] is None and bundle["context"] is None:
        print(f"error: run not found: {args.run_id}", file=sys.stderr)
        return 2
    if args.json:
        _print_json(bundle)
        return 0

    print(f"run: {args.run_id}")
    print(f"latest_seq: {bundle['latest_seq']}")
    checkpoint = bundle["checkpoint"]
    if checkpoint is not None:
        print(f"checkpoint: seq={checkpoint['seq_id']} type={checkpoint['snapshot_type']}")
    else:
        print("checkpoint: none")

    planner = bundle["planner_state"] or {}
    print(f"plan (version {planner.get('version')}):")
    for item in planner.get("items", []):
        print(f"  [{item['status']:<12}] {item['plan_item_id']:<10} {item['title']}")

    context = bundle["context"] or {}
    print(f"messages: {len(context.get('messages', []))}")
    pending = context.get("pending_execution_ids", [])
    print(f"pending executions: {pending if pending else 'none'}")
    print(f"events after checkpoint: {len(bundle['events_after_checkpoint'])}")
    return 0


def _cmd_events(service: LoggingService, args: argparse.Namespace) -> int:
    events = service.read_run_events(args.run_id, after_seq=args.after_seq, limit=args.limit)
    if args.json:
        _print_json([event.to_dict() for event in events])
        return 0
    if not events:
        print("no events found")
        return 0
    for event in events:
        print(f"{event.seq_id:>6}  {event.created_at.isoformat()}  {event.event_type:<32} {event.entity_id}")
    return 0


def _cmd_executions(service: LoggingService, args: argparse.Namespace) -> int:
    records = service.list_external_results(args.run_id)
    if args.json:
        _print_json([record.to_dict() for record in records])
        return 0
    if not records:
        print("no executions found")
        return 0
    for record in records:
        error_suffix = f"  error={record.error_message}" if record.error_message else ""
        print(
            f"{record.execution_id}  {record.tool_name:<14} {record.operation_type:<6} "
            f"{record.result_status.value:<8} {record.execution_status.value:<20}{error_suffix}"
        )
    return 0


def _cmd_pending(service: LoggingService, args: argparse.Namespace) -> int:
    pending = service.list_pending_executions(args.run_id)
    if args.json:
        _print_json(pending)
        return 0
    if not pending:
        print("no pending executions")
        return 0
    for execution_id in pending:
        record = service.get_external_result(execution_id)
        status = record.execution_status.value if record is not None else "missing_record"
        print(f"{execution_id}  {status}")
    return 0


def _cmd_report(service: LoggingService, args: argparse.Namespace) -> int:
    report = build_evaluation_report(service, args.run_id)
    if args.json:
        _print_json(report)
        return 0
    print(f"run: {args.run_id}")
    print(f"spans: {report['span_count']}")
    print(f"completed step duration: {report['completed_step_duration_ms']:.3f} ms")
    for phase, duration in report["completed_phase_duration_ms"].items():
        print(f"  {phase:<24} {duration:>12.3f} ms")
    for step in report["steps"]:
        for attempt in step["attempts"]:
            duration = attempt["duration_ms"]
            duration_text = f"{duration:.3f}" if isinstance(duration, (int, float)) else "n/a"
            print(
                f"step {step['step_index']} attempt {attempt['attempt_no']}: "
                f"{attempt['status']:<12} {duration_text:>12} ms"
            )
    return 0


def _cmd_recover(service: LoggingService, args: argparse.Namespace) -> int:
    if args.dry_run:
        strategy = RuntimeRecoveryStrategy(default_wait_seconds=args.wait_seconds)
        now = datetime.now(UTC)
        decisions = [
            strategy.decide(
                run_id=args.run_id,
                execution_id=record.execution_id,
                record=record,
                now=now,
            )
            for record in service.list_external_results(args.run_id)
        ]
    else:
        coordinator = RecoveryCoordinator(service, default_wait_seconds=args.wait_seconds)
        decisions = coordinator.recover_run(args.run_id)

    payload = [
        {
            "execution_id": decision.execution_id,
            "action": decision.action.value,
            "reason": decision.reason,
            "execution_status": decision.execution_status.value if decision.execution_status else None,
        }
        for decision in decisions
    ]
    if args.json:
        _print_json(payload)
        return 0
    if not payload:
        print("no executions to recover")
        return 0
    for entry in payload:
        print(f"{entry['execution_id']}  {entry['action']:<26} {entry['reason']}")
    return 0


def _cmd_audit(service: LoggingService, args: argparse.Namespace) -> int:
    run_id = args.run_id
    violations: list[str] = []
    notes: list[str] = []

    records = {record.execution_id: record for record in service.list_external_results(run_id)}
    pending = service.list_pending_executions(run_id)

    for execution_id in pending:
        record = records.get(execution_id)
        if record is None:
            violations.append(f"pending execution has no durable record: {execution_id}")
        elif record.execution_status in _TERMINAL_EXECUTION_STATUSES:
            violations.append(
                f"pending execution already terminal ({record.execution_status.value}): {execution_id}"
            )

    intent_events = [
        event
        for event in service.read_run_events(run_id, limit=100_000)
        if event.event_type == "TOOL_INTENT_RECORDED"
    ]
    for event in intent_events:
        if event.entity_id not in records:
            violations.append(f"tool intent without external result record: {event.entity_id}")

    checkpoint = service.wal.read_latest_checkpoint(run_id)
    latest_seq = service.latest_seq(run_id)
    if checkpoint is not None and latest_seq is not None and checkpoint.seq_id > latest_seq:
        violations.append(
            f"checkpoint seq {checkpoint.seq_id} is ahead of latest event seq {latest_seq}"
        )

    open_spans = [
        span
        for span in service.store.list_evaluation_spans(run_id)
        if span["finished_at"] is None
    ]
    if open_spans:
        notes.append(f"{len(open_spans)} open evaluation span(s) (interrupted attempts)")

    non_terminal = [
        record.execution_id
        for record in records.values()
        if record.execution_status not in _TERMINAL_EXECUTION_STATUSES
    ]
    if non_terminal:
        notes.append(f"{len(non_terminal)} non-terminal execution(s): {', '.join(non_terminal)}")

    result = {
        "run_id": run_id,
        "ok": not violations,
        "executions": len(records),
        "intent_events": len(intent_events),
        "violations": violations,
        "notes": notes,
    }
    if args.json:
        _print_json(result)
    else:
        print(f"run: {run_id}")
        print(f"executions: {result['executions']}  intent events: {result['intent_events']}")
        for note in notes:
            print(f"note: {note}")
        if violations:
            for violation in violations:
                print(f"VIOLATION: {violation}")
        else:
            print("audit ok: log-first invariants hold")
    return 0 if not violations else 1


_COMMANDS = {
    "runs": _cmd_runs,
    "show": _cmd_show,
    "events": _cmd_events,
    "executions": _cmd_executions,
    "pending": _cmd_pending,
    "report": _cmd_report,
    "recover": _cmd_recover,
    "audit": _cmd_audit,
}


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="raef",
        description="Inspect and recover RAEF durable runtime stores.",
    )
    parser.add_argument(
        "--data-root",
        default="./data/raef_runtime",
        help="Directory containing runtime.sqlite (default: ./data/raef_runtime)",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("runs", help="List runs in the store")

    show = subparsers.add_parser("show", help="Show plan, checkpoint, and pending state for one run")
    show.add_argument("run_id")

    events = subparsers.add_parser("events", help="Print the WAL event stream for one run")
    events.add_argument("run_id")
    events.add_argument("--after-seq", type=int, default=None)
    events.add_argument("--limit", type=int, default=1000)

    executions = subparsers.add_parser("executions", help="List tool executions and their statuses")
    executions.add_argument("run_id")

    pending = subparsers.add_parser("pending", help="List pending (in-flight) executions")
    pending.add_argument("run_id")

    report = subparsers.add_parser("report", help="Print the durable evaluation timing report")
    report.add_argument("run_id")

    recover = subparsers.add_parser("recover", help="Run recovery decisions for one run")
    recover.add_argument("run_id")
    recover.add_argument("--dry-run", action="store_true", help="Print decisions without applying them")
    recover.add_argument("--wait-seconds", type=float, default=5.0)

    audit = subparsers.add_parser("audit", help="Check log-first invariants for one run")
    audit.add_argument("run_id")

    return parser


if __name__ == "__main__":
    raise SystemExit(main())
