"""Reduced crash-matrix sweep: the single-commit invariant, enforced in CI.

The full nine-phase sweep runs via ``--sweep-crash-phases`` (see
``docs/artifacts/2026-07-16-crash-matrix/``). This test drives a three-phase
subset through the real subprocess crash/restart flow and asserts side-effect
cardinality from the mock target store's own ``stats.total_commits`` counter,
not from framework logs.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCRIPT = REPO_ROOT / "example" / "crash_recovery_e2e.py"

# One phase from each regime: before the write is dispatched, immediately
# after the write commits, and after the plan item is already marked done.
REDUCED_PHASES = "before_tool_transaction,after_tool_transaction,after_advance_plan_item"


def _run_sweep(sweep_root: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SWEEP_SCRIPT),
            "--sweep-crash-phases",
            "--sweep-phases",
            REDUCED_PHASES,
            "--sweep-root",
            str(sweep_root),
            *extra_args,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=300,
    )


def test_reduced_sweep_yields_exactly_one_commit_per_phase(tmp_path) -> None:
    sweep_root = tmp_path / "sweep"
    result = _run_sweep(sweep_root)
    assert result.returncode == 0, f"sweep failed:\n{result.stdout}\n{result.stderr}"

    summary = json.loads((sweep_root / "crash_matrix_summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] is True
    assert summary["expected_commits_per_run"] == 1
    rows = {row["crash_phase"]: row for row in summary["phases"]}
    assert set(rows) == set(REDUCED_PHASES.split(","))

    for row in rows.values():
        assert row["crash_exit_code"] == 42
        assert row["resume_exit_code"] == 0
        assert row["target_commits"] == 1
        assert row["duplicate_commits"] == 0
        assert row["order_booked"] is True
        assert row["interrupted_attempts"] >= 1  # the crash really landed

    # Crash before the write: the restarted process performs it exactly once.
    assert rows["before_tool_transaction"]["classification"] == "committed_after_restart"
    assert rows["before_tool_transaction"]["commits_before_restart"] == 0
    # Crash after the write: the restarted process reuses, never re-fires.
    assert rows["after_tool_transaction"]["classification"] == "reused_committed_result"
    assert rows["after_tool_transaction"]["commits_before_restart"] == 1
    assert rows["after_advance_plan_item"]["classification"] == "reused_committed_result"

    # The human-readable table is written alongside the JSON.
    assert (sweep_root / "crash_matrix_table.txt").exists()


def test_sweep_exits_nonzero_on_cardinality_mismatch(tmp_path) -> None:
    sweep_root = tmp_path / "sweep-fail"
    result = subprocess.run(
        [
            sys.executable,
            str(SWEEP_SCRIPT),
            "--sweep-crash-phases",
            "--sweep-phases",
            "after_tool_transaction",
            "--sweep-root",
            str(sweep_root),
            "--sweep-expected-commits",
            "2",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=300,
    )
    assert result.returncode == 1
    summary = json.loads((sweep_root / "crash_matrix_summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] is False
    assert summary["phases"][0]["classification"] == "violation"
