"""End-to-end crash/restart recovery experiment for the mock RAEF agent.

Run from the repository root:

PYTHONPATH=src .venv/bin/python example/crash_recovery_e2e.py --clean

Sweep every crash-injection phase and verify side-effect cardinality from the
mock target's own commit counter (exit non-zero on any duplicate/missing
commit):

PYTHONPATH=src .venv/bin/python example/crash_recovery_e2e.py --sweep-crash-phases
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from raef.evaluation import build_evaluation_report
from raef.logging_service import LoggingService
from raef.tools.crash_simulator import CrashSimulator, SimulatedCrash

from sample_agent_openrouter_mock import run_sample_agent_openrouter_mock


DEFAULT_RUNTIME_DIR = Path("./data/openrouter_mock_crash_e2e")
DEFAULT_SWEEP_ROOT = Path("./data/crash_matrix")
CRASH_EXIT_CODE = 42

# The demo books exactly one flight, so every completed scenario must leave
# exactly one commit in the target store, regardless of where the crash lands.
EXPECTED_COMMITS_PER_RUN = 1

# (phase, crash_step): every injection point the mock runner supports. The
# final-message phase only fires on the FINAL step of the script (step 1);
# all other phases fire on the tool-call step (step 0).
SWEEP_PHASES: list[tuple[str, int]] = [
    ("after_step_started", 0),
    ("before_agent_turn", 0),
    ("after_agent_turn", 0),
    ("after_record_llm_turn", 0),
    ("before_tool_transaction", 0),
    ("after_tool_transaction", 0),
    ("after_recovery", 0),
    ("after_advance_plan_item", 0),
    ("after_record_final_message", 1),
]


def _demo(message: str) -> None:
    print(message, flush=True)


def run_parent(args: argparse.Namespace) -> int:
    runtime_dir = Path(args.runtime_dir)
    target_store_path = runtime_dir / "openrouter_mock_target.json"
    soft_log_path = Path(args.soft_log_path) if args.soft_log_path else runtime_dir / "synthetic_soft_log.json"
    run_id = args.run_id or f"sample-openrouter-mock-crash-e2e-{int(time.time())}"

    _demo("Starting crash/recovery end-to-end demo")
    _demo(f"Run id: {run_id}")
    _demo(f"Runtime directory: {runtime_dir}")

    if args.clean and runtime_dir.exists():
        _demo(f"Cleaning previous runtime directory: {runtime_dir}")
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    _demo("Runtime directory is ready")

    if not soft_log_path.exists():
        _demo(f"Writing synthetic soft log: {soft_log_path}")
        _write_synthetic_soft_log(soft_log_path)
    else:
        _demo(f"Using existing soft log: {soft_log_path}")

    _demo(f"Target store path: {target_store_path}")
    _demo(
        "Crash will be injected at "
        f"step {args.crash_step}, phase {args.crash_phase}"
    )

    crash_cmd = _child_cmd(
        args,
        run_id=run_id,
        soft_log_path=soft_log_path,
        runtime_dir=runtime_dir,
        target_store_path=target_store_path,
        crash=True,
    )
    _demo("Launching first agent process; its output will stream below")
    crash_run = subprocess.run(crash_cmd, cwd=REPO_ROOT, text=True)
    _demo(f"First process exited with code {crash_run.returncode}")
    if crash_run.returncode != CRASH_EXIT_CODE:
        _demo(
            "Expected a simulated crash exit code "
            f"{CRASH_EXIT_CODE}, got {crash_run.returncode}"
        )
        return 1
    _demo("Simulated crash confirmed; now restarting the agent with the same run id")

    resume_cmd = _child_cmd(
        args,
        run_id=run_id,
        soft_log_path=soft_log_path,
        runtime_dir=runtime_dir,
        target_store_path=target_store_path,
        crash=False,
    )
    _demo("Launching resumed agent process; recovery output will stream below")
    resume_run = subprocess.run(resume_cmd, cwd=REPO_ROOT, text=True)
    _demo(f"Resumed process exited with code {resume_run.returncode}")
    if resume_run.returncode != 0:
        _demo("Resume failed; see streamed child output above")
        return resume_run.returncode or 1

    _demo("Building recovery evaluation report")
    service = LoggingService.with_data_root(runtime_dir, checkpoint_every_n_events=4)
    report = build_evaluation_report(service, run_id)
    bundle = service.get_recovery_bundle(run_id)
    summary = _summarize_report(report)
    result = {
        "ok": True,
        "run_id": run_id,
        "runtime_dir": str(runtime_dir),
        "soft_log_path": str(soft_log_path),
        "target_store_path": str(target_store_path),
        "inference_log_path": str(runtime_dir / "openrouter_mock_inference.jsonl"),
        "crash": {
            "step": args.crash_step,
            "phase": args.crash_phase,
            "exit_code": crash_run.returncode,
        },
        "resume_exit_code": resume_run.returncode,
        "evaluation_summary": summary,
        "latest_seq": bundle.get("latest_seq"),
        "planner_statuses": [
            {
                "plan_item_id": item.get("plan_item_id"),
                "status": item.get("status"),
            }
            for item in (bundle.get("planner_state") or {}).get("items", [])
        ],
        "report": report,
    }
    output_path = Path(args.output_path) if args.output_path else runtime_dir / "evaluation_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["output_path"] = str(output_path)
    _demo(f"Wrote evaluation summary JSON: {output_path}")

    timing_output_path = (
        Path(args.timing_output_path)
        if args.timing_output_path
        else runtime_dir / "evaluation_timing.txt"
    )
    timing_output_path.parent.mkdir(parents=True, exist_ok=True)
    timing_output_path.write_text(_format_timing_report(summary), encoding="utf-8")
    result["timing_output_path"] = str(timing_output_path)
    _demo(f"Wrote timing report: {timing_output_path}")
    _print_demo_summary(result)
    return 0


def run_child(args: argparse.Namespace) -> int:
    runtime_dir = Path(args.runtime_dir)
    target_store_path = Path(args.target_store_path)
    mode = "crash run" if args.child_crash else "resume run"
    _demo(f"Child process started ({mode})")
    _demo(f"Child run id: {args.run_id}")
    _demo(f"Child runtime directory: {runtime_dir}")

    logging_service = LoggingService.with_data_root(runtime_dir, checkpoint_every_n_events=4)
    _demo("Logging service initialized")

    simulator = None
    if args.child_crash:
        _demo(f"Crash simulator armed for step {args.crash_step}")
        simulator = CrashSimulator(crash_steps={args.crash_step}, crash_once=True)
    else:
        _demo("Crash simulator disabled for resumed run")

    try:
        _demo("Invoking sample OpenRouter mock agent")
        result = run_sample_agent_openrouter_mock(
            run_id=args.run_id,
            logging_service=logging_service,
            soft_log_path=Path(args.soft_log_path),
            force_reset=args.child_crash,
            reset_target=args.child_crash,
            target_store_path=target_store_path,
            inference_log_path=runtime_dir / "openrouter_mock_inference.jsonl",
            simulate_ambiguous_write=args.simulate_ambiguous_write,
            simulate_lost_write=args.simulate_lost_write,
            crash_simulator=simulator,
            crash_phase=args.crash_phase,
            on_llm_response=_print_mock_llm_output,
        )
    except SimulatedCrash as exc:
        _demo(f"Simulated crash raised: {exc}")
        _demo("Building partial evaluation report after crash")
        report = build_evaluation_report(logging_service, args.run_id)
        _print_child_demo_summary(
            crashed=True,
            run_id=args.run_id,
            summary=_summarize_report(report),
            crash_message=str(exc),
        )
        return CRASH_EXIT_CODE

    _demo("Agent completed without crashing")
    _print_child_result(result)
    return 0


def _child_cmd(
    args: argparse.Namespace,
    *,
    run_id: str,
    soft_log_path: Path,
    runtime_dir: Path,
    target_store_path: Path,
    crash: bool,
    crash_step: int | None = None,
    crash_phase: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--run-id",
        run_id,
        "--soft-log-path",
        str(soft_log_path),
        "--runtime-dir",
        str(runtime_dir),
        "--target-store-path",
        str(target_store_path),
        "--crash-step",
        str(crash_step if crash_step is not None else args.crash_step),
        "--crash-phase",
        crash_phase if crash_phase is not None else args.crash_phase,
    ]
    if crash:
        cmd.append("--child-crash")
        # Fault injection only applies to the first process; the resumed
        # process runs against a healthy adapter, as a real restart would.
        if args.simulate_ambiguous_write:
            cmd.append("--simulate-ambiguous-write")
        if args.simulate_lost_write:
            cmd.append("--simulate-lost-write")
    return cmd


def run_sweep(args: argparse.Namespace) -> int:
    """Run the crash scenario once per injection phase and verify cardinality.

    The pass/fail signal is read from the mock target store's own
    ``stats.total_commits`` counter, never from framework logs: exactly one
    commit per business action, zero duplicates, order actually booked.
    """

    phases = _resolve_sweep_phases(args.sweep_phases)
    expected_commits = args.sweep_expected_commits
    sweep_root = Path(args.sweep_root)
    if args.clean and sweep_root.exists():
        _demo(f"Cleaning previous sweep root: {sweep_root}")
        shutil.rmtree(sweep_root)
    sweep_root.mkdir(parents=True, exist_ok=True)

    _demo(f"Crash-matrix sweep: {len(phases)} phase(s), sweep root {sweep_root}")
    rows: list[dict[str, Any]] = []
    for index, (phase, crash_step) in enumerate(phases):
        runtime_dir = sweep_root / f"{index:02d}_{phase}"
        row = _run_sweep_scenario(
            args,
            runtime_dir=runtime_dir,
            crash_phase=phase,
            crash_step=crash_step,
            expected_commits=expected_commits,
        )
        rows.append(row)
        verdict = "ok" if row["ok"] else "VIOLATION"
        _demo(
            f"[{index + 1}/{len(phases)}] {phase:<28} commits={row['target_commits']} "
            f"dup={row['duplicate_commits']} {row['classification']:<26} {verdict}"
        )

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "expected_commits_per_run": expected_commits,
        "phase_count": len(rows),
        "ok": all(row["ok"] for row in rows),
        "phases": rows,
    }
    summary_path = sweep_root / "crash_matrix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    table = _format_sweep_table(rows)
    table_path = sweep_root / "crash_matrix_table.txt"
    table_path.write_text(table, encoding="utf-8")

    _demo("")
    print(table, end="", flush=True)
    _demo("")
    _demo(f"Wrote machine-readable summary: {summary_path}")
    _demo(f"Wrote table: {table_path}")
    if summary["ok"]:
        _demo(
            f"Sweep passed: exactly {expected_commits} target commit(s) in "
            f"all {len(rows)} phase(s), zero duplicates"
        )
        return 0
    failed = [row["crash_phase"] for row in rows if not row["ok"]]
    _demo(f"Sweep FAILED for phase(s): {', '.join(failed)}")
    return 1


def _resolve_sweep_phases(raw: str | None) -> list[tuple[str, int]]:
    if raw is None or not raw.strip():
        return list(SWEEP_PHASES)
    known = dict(SWEEP_PHASES)
    selected: list[tuple[str, int]] = []
    for name in raw.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in known:
            raise SystemExit(
                f"unknown sweep phase: {name} (known: {', '.join(sorted(known))})"
            )
        selected.append((name, known[name]))
    if not selected:
        raise SystemExit("--sweep-phases selected no phases")
    return selected


def _run_sweep_scenario(
    args: argparse.Namespace,
    *,
    runtime_dir: Path,
    crash_phase: str,
    crash_step: int,
    expected_commits: int = EXPECTED_COMMITS_PER_RUN,
) -> dict[str, Any]:
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    soft_log_path = runtime_dir / "synthetic_soft_log.json"
    _write_synthetic_soft_log(soft_log_path)
    target_store_path = runtime_dir / "openrouter_mock_target.json"
    run_id = f"crash-matrix-{crash_phase}"

    crash_cmd = _child_cmd(
        args,
        run_id=run_id,
        soft_log_path=soft_log_path,
        runtime_dir=runtime_dir,
        target_store_path=target_store_path,
        crash=True,
        crash_step=crash_step,
        crash_phase=crash_phase,
    )
    crash_run = subprocess.run(crash_cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    (runtime_dir / "crash_child_output.txt").write_text(
        crash_run.stdout + crash_run.stderr, encoding="utf-8"
    )
    commits_before_restart = _read_target_commits(target_store_path)

    resume_cmd = _child_cmd(
        args,
        run_id=run_id,
        soft_log_path=soft_log_path,
        runtime_dir=runtime_dir,
        target_store_path=target_store_path,
        crash=False,
        crash_step=crash_step,
        crash_phase=crash_phase,
    )
    resume_started = time.monotonic()
    resume_run = subprocess.run(resume_cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    resume_wall_ms = (time.monotonic() - resume_started) * 1000.0
    (runtime_dir / "resume_child_output.txt").write_text(
        resume_run.stdout + resume_run.stderr, encoding="utf-8"
    )

    target_commits = _read_target_commits(target_store_path)
    order_state = _read_order_state(target_store_path)
    order_booked = order_state == {"flight_no": "RA512", "status": "booked"}
    duplicate_commits = max(0, (target_commits or 0) - expected_commits)

    attempts, interrupted_attempts, recovery_phase_ms, execution_statuses = _collect_run_evidence(
        runtime_dir, run_id
    )

    ok = (
        crash_run.returncode == CRASH_EXIT_CODE
        and resume_run.returncode == 0
        and target_commits == expected_commits
        and order_booked
    )
    if not ok:
        classification = "violation"
    elif commits_before_restart == expected_commits:
        classification = "reused_committed_result"
    else:
        classification = "committed_after_restart"

    return {
        "crash_phase": crash_phase,
        "crash_step": crash_step,
        "runtime_dir": str(runtime_dir),
        "crash_exit_code": crash_run.returncode,
        "resume_exit_code": resume_run.returncode,
        "commits_before_restart": commits_before_restart,
        "target_commits": target_commits,
        "expected_commits": expected_commits,
        "duplicate_commits": duplicate_commits,
        "order_booked": order_booked,
        "attempts": attempts,
        "interrupted_attempts": interrupted_attempts,
        "execution_statuses": execution_statuses,
        "recovery_phase_ms": recovery_phase_ms,
        "resume_wall_ms": round(resume_wall_ms, 3),
        "classification": classification,
        "ok": ok,
    }


def _read_target_commits(target_store_path: Path) -> int | None:
    if not target_store_path.exists():
        return None
    payload = json.loads(target_store_path.read_text(encoding="utf-8"))
    return int(payload.get("stats", {}).get("total_commits", 0))


def _read_order_state(target_store_path: Path) -> Any:
    if not target_store_path.exists():
        return None
    payload = json.loads(target_store_path.read_text(encoding="utf-8"))
    return payload.get("domain_state", {}).get("orders/RA512")


def _collect_run_evidence(
    runtime_dir: Path,
    run_id: str,
) -> tuple[list[dict[str, Any]], int, float | None, list[str]]:
    """Read attempt/timing/execution evidence back from the durable store."""

    service = LoggingService.with_data_root(runtime_dir, soft_write_enabled=False)
    try:
        report = build_evaluation_report(service, run_id)
        attempts = [
            {
                "step_index": step.get("step_index"),
                "attempt_no": attempt.get("attempt_no"),
                "status": attempt.get("status"),
            }
            for step in report.get("steps", [])
            for attempt in step.get("attempts", [])
        ]
        interrupted = sum(1 for attempt in attempts if attempt["status"] == "interrupted")
        phase_durations = report.get("completed_phase_duration_ms", {})
        recovery_phase_ms = phase_durations.get("recovery")
        execution_statuses = [
            f"{record.tool_name}:{record.execution_status.value}"
            for record in service.list_external_results(run_id)
        ]
    finally:
        service.close()
    return attempts, interrupted, recovery_phase_ms, execution_statuses


def _format_sweep_table(rows: list[dict[str, Any]]) -> str:
    header = (
        f"{'phase':<28} {'step':>4} {'attempts':>8} {'intr':>4} {'commits':>7} "
        f"{'dup':>3} {'classification':<26} {'recovery_ms':>11} {'resume_ms':>9} {'result':<9}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        recovery_ms = row["recovery_phase_ms"]
        recovery_text = f"{recovery_ms:.3f}" if isinstance(recovery_ms, (int, float)) else "n/a"
        lines.append(
            f"{row['crash_phase']:<28} {row['crash_step']:>4} {len(row['attempts']):>8} "
            f"{row['interrupted_attempts']:>4} {row['target_commits']!s:>7} {row['duplicate_commits']:>3} "
            f"{row['classification']:<26} {recovery_text:>11} {row['resume_wall_ms']:>9.1f} "
            f"{'ok' if row['ok'] else 'VIOLATION':<9}"
        )
    return "\n".join(lines) + "\n"


def _write_synthetic_soft_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tool_call = {
        "kind": "tool_call",
        "tool_call": {
            "name": "apply-action",
            "operation_type": "WRITE",
            "arguments": {
                "action_name": "set_value",
                "payload": {
                    "key": "orders/RA512",
                    "value": {
                        "flight_no": "RA512",
                        "status": "booked",
                    },
                },
            },
            "rationale": "Book the selected flight.",
        },
    }
    payload: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": "Please buy me a ticket from LA to NY for 2026-05-14 09:00 AM PT.",
            },
            {
                "role": "assistant",
                "content": "Booked RA512.",
                "metadata": {
                    "stop_reason": "completed",
                    "model_name": "synthetic-crash-e2e",
                },
            },
        ],
        "planner_state": {
            "items": [
                {
                    "plan_item_id": "step_0",
                    "title": "Book the selected flight",
                    "llm_output": json.dumps(tool_call, sort_keys=True),
                }
            ]
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    phase_counts: dict[str, int] = {}
    for step in report.get("steps", []):
        for attempt in step.get("attempts", []):
            attempts.append(
                {
                    "step_index": step.get("step_index"),
                    "plan_item_id": step.get("plan_item_id"),
                    "attempt_no": attempt.get("attempt_no"),
                    "phase": attempt.get("phase"),
                    "status": attempt.get("status"),
                    "recorded_status": attempt.get("recorded_status"),
                    "duration_ms": attempt.get("duration_ms"),
                }
            )
            for phase in attempt.get("phases", []):
                if not isinstance(phase, dict):
                    continue
                phase_name = phase.get("phase")
                if isinstance(phase_name, str):
                    phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1
    return {
        "span_count": report.get("span_count"),
        "completed_step_duration_ms": report.get("completed_step_duration_ms"),
        "completed_phase_duration_ms": report.get("completed_phase_duration_ms"),
        "completed_phase_counts": phase_counts,
        "attempts": attempts,
    }


def _format_timing_report(summary: dict[str, Any]) -> str:
    lines = [
        _format_duration_line("completed_step_duration_ms", summary.get("completed_step_duration_ms")),
    ]
    phase_durations = summary.get("completed_phase_duration_ms")
    if not isinstance(phase_durations, dict):
        phase_durations = {}
    phase_counts = summary.get("completed_phase_counts")
    if not isinstance(phase_counts, dict):
        phase_counts = {}
    preferred_phase_order = [
        "llm_generate",
        "recovery",
        "tool_transaction",
        "record_llm_turn",
        "advance_plan_item",
        "record_final_message",
    ]
    ordered_phases = [
        phase for phase in preferred_phase_order if phase in phase_durations
    ]
    ordered_phases.extend(
        phase for phase in sorted(phase_durations) if phase not in set(ordered_phases)
    )
    for phase in ordered_phases:
        lines.append(_format_duration_line(phase, phase_durations.get(phase), phase_counts.get(phase)))

    lines.append("")
    attempts = summary.get("attempts")
    if isinstance(attempts, list):
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            step_index = attempt.get("step_index")
            attempt_no = attempt.get("attempt_no")
            status = str(attempt.get("status"))
            duration = _format_ms(attempt.get("duration_ms"))
            lines.append(f"step {step_index} attempt {attempt_no}: {status:<12} {duration:>12} ms")
    return "\n".join(lines) + "\n"


def _format_duration_line(label: str, value: Any, count: Any = None) -> str:
    suffix = f"  count={count}" if isinstance(count, int) else ""
    return f"{label:<28}: {_format_ms(value):>12} ms{suffix}"


def _format_ms(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "n/a"


def _print_mock_llm_output(step_index: int, raw_output: str) -> None:
    _demo(f"Mock LLM output for step {step_index}:")
    print(raw_output, flush=True)


def _print_demo_summary(result: dict[str, Any]) -> None:
    summary = result.get("evaluation_summary")
    if not isinstance(summary, dict):
        summary = {}

    _demo("Demo completed successfully")
    _demo(f"Latest recovered event sequence: {result.get('latest_seq')}")
    _demo(f"Inference log: {result.get('inference_log_path')}")
    _demo(f"Evaluation summary: {result.get('output_path')}")
    _demo(f"Timing report: {result.get('timing_output_path')}")

    attempts = summary.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        _demo("No attempts were reported")
        return

    _demo("Recovered attempt timeline:")
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        step_index = attempt.get("step_index")
        attempt_no = attempt.get("attempt_no")
        status = attempt.get("status")
        phase = attempt.get("phase")
        duration = _format_ms(attempt.get("duration_ms"))
        _demo(
            f"  step {step_index}, attempt {attempt_no}: "
            f"status={status}, phase={phase}, duration_ms={duration}"
        )


def _print_child_demo_summary(
    *,
    crashed: bool,
    run_id: str,
    summary: dict[str, Any],
    crash_message: str | None = None,
) -> None:
    state = "crashed" if crashed else "completed"
    _demo(f"Child process {state} for run id: {run_id}")
    if crash_message:
        _demo(f"Crash message: {crash_message}")

    attempts = summary.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        _demo("No attempts available yet")
        return

    _demo("Current attempt timeline:")
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        _demo(
            "  "
            f"step={attempt.get('step_index')} "
            f"attempt={attempt.get('attempt_no')} "
            f"status={attempt.get('status')} "
            f"phase={attempt.get('phase')}"
        )


def _print_child_result(result: Any) -> None:
    if isinstance(result, dict):
        status = result.get("status") or result.get("final_status") or "complete"
        _demo(f"Child result status: {status}")
        final_message = result.get("final_message") or result.get("message")
        if final_message:
            _demo(f"Child final message: {final_message}")
        return
    _demo(f"Child result: {result}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the mock RAEF agent across a crash/restart boundary.")
    parser.add_argument("--run-id")
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME_DIR))
    parser.add_argument("--target-store-path", default="")
    parser.add_argument("--soft-log-path")
    parser.add_argument("--crash-step", type=int, default=0)
    parser.add_argument("--crash-phase", default="after_tool_transaction")
    parser.add_argument("--output-path")
    parser.add_argument("--timing-output-path")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--simulate-ambiguous-write", action="store_true")
    parser.add_argument("--simulate-lost-write", action="store_true")
    parser.add_argument(
        "--sweep-crash-phases",
        action="store_true",
        help="Run the scenario once per crash-injection phase and verify exactly one "
        "target commit per run (exit 1 on any duplicate/missing commit).",
    )
    parser.add_argument(
        "--sweep-phases",
        default=None,
        help="Comma-separated subset of phases for the sweep (default: all phases).",
    )
    parser.add_argument("--sweep-root", default=str(DEFAULT_SWEEP_ROOT))
    # Test seam: lets the suite prove the sweep really fails on a cardinality
    # mismatch without corrupting a real run.
    parser.add_argument(
        "--sweep-expected-commits",
        type=int,
        default=EXPECTED_COMMITS_PER_RUN,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-crash", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.child:
        if not args.run_id:
            raise SystemExit("--run-id is required in child mode")
        if not args.soft_log_path:
            raise SystemExit("--soft-log-path is required in child mode")
        if not args.target_store_path:
            raise SystemExit("--target-store-path is required in child mode")
        return run_child(args)
    if args.sweep_crash_phases:
        return run_sweep(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
