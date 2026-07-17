"""End-to-end crash/restart experiment using the real OpenRouter sample agent.

This runner calls the actual OpenRouter-backed agent, so it requires network
access and OPENROUTER_API_KEY in the environment.

Run from the repository root:

PYTHONPATH=src .venv/bin/python example/openrouter_crash_recovery_e2e.py --clean
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_local_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            if line.startswith("sk-or-") and "OPENROUTER_API_KEY" not in os.environ:
                os.environ["OPENROUTER_API_KEY"] = line
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env_file(Path(os.getenv("OPENROUTER_ENV_FILE", REPO_ROOT / ".env.openrouter")))

from raef.evaluation import build_evaluation_report
from raef.logging_service import LoggingService
from raef.tools.crash_simulator import CrashSimulator, SimulatedCrash

from sample_agent_openrouter import (
    STABLE_USER_PROMPT,
    _configure_logger,
    _openrouter_models_from_env,
    run_sample_agent_openrouter,
)


DEFAULT_RUNTIME_DIR = Path("./data/openrouter_actual_crash_e2e")
CRASH_EXIT_CODE = 42


def run_parent(args: argparse.Namespace) -> int:
    if not os.getenv("OPENROUTER_API_KEY"):
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "OPENROUTER_API_KEY is required for the real OpenRouter e2e test.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    runtime_dir = Path(args.runtime_dir)
    model_names = _resolve_model_names(args)
    target_store_path = runtime_dir / "openrouter_mock_target.json"
    scenario_file = Path(args.scenario_file) if args.scenario_file else runtime_dir / "scenario_ticket_la_to_ny.json"
    run_id = args.run_id or f"actual-openrouter-crash-e2e-{int(time.time())}"

    if args.clean and runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    if not scenario_file.exists():
        _write_default_scenario(scenario_file)

    crash_run = subprocess.run(
        _child_cmd(
            args,
            run_id=run_id,
            runtime_dir=runtime_dir,
            target_store_path=target_store_path,
            scenario_file=scenario_file,
            crash=True,
        ),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )
    if crash_run.returncode != CRASH_EXIT_CODE:
        _print_child_output("crash-child", crash_run)
        return crash_run.returncode or 1

    resume_run = subprocess.run(
        _child_cmd(
            args,
            run_id=run_id,
            runtime_dir=runtime_dir,
            target_store_path=target_store_path,
            scenario_file=scenario_file,
            crash=False,
        ),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )
    if resume_run.returncode != 0:
        _print_child_output("resume-child", resume_run)
        return resume_run.returncode or 1

    service = LoggingService.with_data_root(runtime_dir, checkpoint_every_n_events=4)
    report = build_evaluation_report(service, run_id)
    bundle = service.get_recovery_bundle(run_id)
    summary = _summarize_report(report)
    result = {
        "ok": True,
        "run_id": run_id,
        "runtime_dir": str(runtime_dir),
        "scenario_file": str(scenario_file),
        "target_store_path": str(target_store_path),
        "models": model_names,
        "inference_log_path": str(runtime_dir / "openrouter_inference.jsonl"),
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

    timing_output_path = (
        Path(args.timing_output_path)
        if args.timing_output_path
        else runtime_dir / "evaluation_timing.txt"
    )
    timing_output_path.parent.mkdir(parents=True, exist_ok=True)
    timing_output_path.write_text(_format_timing_report(summary), encoding="utf-8")
    result["timing_output_path"] = str(timing_output_path)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_child(args: argparse.Namespace) -> int:
    runtime_dir = Path(args.runtime_dir)
    _configure_logger(runtime_dir)
    logging_service = LoggingService.with_data_root(Path(args.runtime_dir), checkpoint_every_n_events=4)
    simulator = CrashSimulator(crash_steps={args.crash_step}, crash_once=True) if args.child_crash else None

    try:
        result = run_sample_agent_openrouter(
            run_id=args.run_id,
            user_prompt=STABLE_USER_PROMPT,
            initial_messages=[{"role": "user", "content": STABLE_USER_PROMPT}],
            logging_service=logging_service,
            scenario_file=Path(args.scenario_file),
            force_reset=args.child_crash,
            reset_target=args.child_crash,
            target_store_path=Path(args.target_store_path),
            model_names=_resolve_model_names(args),
            inference_log_path=runtime_dir / "openrouter_inference.jsonl",
            simulate_ambiguous_write=args.simulate_ambiguous_write,
            crash_simulator=simulator,
            crash_phase=args.crash_phase,
        )
    except SimulatedCrash as exc:
        report = build_evaluation_report(logging_service, args.run_id)
        print(
            json.dumps(
                {
                    "crashed": True,
                    "run_id": args.run_id,
                    "crash_message": str(exc),
                    "evaluation_summary": _summarize_report(report),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return CRASH_EXIT_CODE

    print(json.dumps({"crashed": False, "result": result}, indent=2, sort_keys=True))
    return 0


def _child_cmd(
    args: argparse.Namespace,
    *,
    run_id: str,
    runtime_dir: Path,
    target_store_path: Path,
    scenario_file: Path,
    crash: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--run-id",
        run_id,
        "--runtime-dir",
        str(runtime_dir),
        "--target-store-path",
        str(target_store_path),
        "--scenario-file",
        str(scenario_file),
        "--crash-step",
        str(args.crash_step),
        "--crash-phase",
        args.crash_phase,
    ]
    if crash:
        cmd.append("--child-crash")
    if args.simulate_ambiguous_write:
        cmd.append("--simulate-ambiguous-write")
    for model in _resolve_model_names(args):
        cmd.extend(["--model", model])
    return cmd


def _resolve_model_names(args: argparse.Namespace) -> list[str]:
    model_values = getattr(args, "model", None)
    if model_values:
        return [model.strip() for model in model_values if model.strip()]
    return _openrouter_models_from_env()


def _write_default_scenario(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario_id": "ticket-la-to-ny-openrouter-e2e",
        "description": "Book one LAX to JFK flight for the requested date and confirm it was saved.",
        "target_seed_state": {
            "flights/RA501": {
                "available": True,
                "departure_time": "2026-05-14T07:30:00-07:00",
                "flight_no": "RA501",
                "from": "LAX",
                "price_usd": 280,
                "to": "JFK",
            },
            "flights/RA512": {
                "available": True,
                "departure_time": "2026-05-14T09:00:00-07:00",
                "flight_no": "RA512",
                "from": "LAX",
                "price_usd": 325,
                "to": "JFK",
            },
        },
        "steps": [
            {
                "kind": "tool_call",
                "name": "query-state",
                "operation_type": "READ",
                "arguments": {
                    "query_name": "list_keys",
                    "payload": {"prefix": "flights/"},
                },
                "rationale": "Inspect available flights.",
            },
            {
                "kind": "tool_call",
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
                "rationale": "Persist the booking for RA512.",
            },
            {
                "kind": "tool_call",
                "name": "query-state",
                "operation_type": "READ",
                "arguments": {
                    "query_name": "get_value",
                    "payload": {"key": "orders/RA512"},
                },
                "rationale": "Read back the saved booking.",
            },
            {
                "kind": "final",
                "answer": "Booked RA512 from LAX to JFK and confirmed the order.",
            },
        ],
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


def _print_child_output(label: str, completed: subprocess.CompletedProcess[str]) -> None:
    print(f"--- {label} stdout ---")
    print(completed.stdout)
    print(f"--- {label} stderr ---")
    print(completed.stderr)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the real OpenRouter RAEF agent across a crash/restart boundary.")
    parser.add_argument("--run-id")
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME_DIR))
    parser.add_argument("--target-store-path", default="")
    parser.add_argument("--scenario-file")
    parser.add_argument("--crash-step", type=int, default=1)
    parser.add_argument("--crash-phase", default="after_tool_transaction")
    parser.add_argument("--output-path")
    parser.add_argument("--timing-output-path")
    parser.add_argument(
        "--model",
        action="append",
        help="OpenRouter model to use. Repeat for fallback order; defaults to OPENROUTER_MODEL(S) or sample defaults.",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--simulate-ambiguous-write", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-crash", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.child:
        if not args.run_id:
            raise SystemExit("--run-id is required in child mode")
        if not args.scenario_file:
            raise SystemExit("--scenario-file is required in child mode")
        if not args.target_store_path:
            raise SystemExit("--target-store-path is required in child mode")
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
