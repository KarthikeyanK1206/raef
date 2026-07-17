"""Crash injection helpers for deterministic restart and recovery testing.

Use this module to simulate process crashes at specific step indexes while
replaying history files or future runtime execution flows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from . import mock_agent


class SimulatedCrash(RuntimeError):
    """Raised when crash injection triggers at a configured step."""


def crash(*, reason: str = "simulated_crash") -> None:
    """Crash API used by tests/demos to emulate a hard process stop boundary."""

    raise SimulatedCrash(reason)


@dataclass
class CrashSimulator:
    """Inject deterministic crashes at selected step indexes.

    By default each configured step crashes once. Set crash_once=False when a
    repeated crash should happen on every matching attempt.
    """

    crash_steps: set[int]
    crash_once: bool = True
    _triggered_steps: set[int] = field(default_factory=set, init=False, repr=False)

    def maybe_crash(self, step_index: int, *, phase: str = "execution") -> None:
        if step_index not in self.crash_steps:
            return
        if self.crash_once and step_index in self._triggered_steps:
            return
        self._triggered_steps.add(step_index)
        crash(reason=f"Simulated crash at step={step_index} phase={phase}")

    def reset(self) -> None:
        self._triggered_steps.clear()

    def get_state(self) -> dict[str, Any]:
        return {
            "crash_steps": sorted(self.crash_steps),
            "crash_once": self.crash_once,
            "triggered_steps": sorted(self._triggered_steps),
        }


def simulate_history_execution(
    history_file: str | Path,
    *,
    crash_steps: Sequence[int],
    crash_once: bool = True,
) -> dict[str, Any]:
    """Replay one history script and report where a simulated crash occurs."""
    script = mock_agent.load_script_from_history_file(history_file)
    simulator = CrashSimulator(crash_steps=set(crash_steps), crash_once=crash_once)

    trace: list[dict[str, Any]] = []
    crashed = False
    crash_message: str | None = None

    for step_index, step in enumerate(script):
        try:
            simulator.maybe_crash(step_index, phase="before_agent_turn")
        except SimulatedCrash as exc:
            crashed = True
            crash_message = str(exc)
            trace.append(
                {
                    "step_index": step_index,
                    "phase": "before_agent_turn",
                    "event": "crash",
                    "kind": step.kind.value,
                }
            )
            break

        trace.append(
            {
                "step_index": step_index,
                "phase": "after_agent_turn",
                "event": "ok",
                "kind": step.kind.value,
            }
        )

        if step.kind == mock_agent.DecisionKind.FINAL:
            break

    return {
        "history_file": str(history_file),
        "script_length": len(script),
        "crash_steps": sorted(set(crash_steps)),
        "crashed": crashed,
        "crash_message": crash_message,
        "trace": trace,
        "simulator_state": simulator.get_state(),
    }


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crash simulator for RAEF history scenarios")
    parser.add_argument(
        "--history-file",
        required=True,
        help="Path to history/scenario JSON file.",
    )
    parser.add_argument(
        "--crash-step",
        type=int,
        action="append",
        default=[],
        help="Step index that should crash. Repeat option for multiple steps.",
    )
    parser.add_argument(
        "--repeat-crash",
        action="store_true",
        help="Crash every time a crash step is visited, not just first time.",
    )
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    try:
        outcome = simulate_history_execution(
            args.history_file,
            crash_steps=args.crash_step,
            crash_once=not args.repeat_crash,
        )
        print(json.dumps(outcome, separators=(",", ":"), sort_keys=True))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2


def main() -> int:
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
