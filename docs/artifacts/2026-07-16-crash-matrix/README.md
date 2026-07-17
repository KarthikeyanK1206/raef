# Crash-matrix evidence run

- Command: `.venv/bin/python example/crash_recovery_e2e.py --sweep-crash-phases --clean`
- Date: 2026-07-16
- Machine: Apple M4 (arm64), macOS 26.5.1 (build 25F80)
- Python: 3.13.13 (project venv)

## What this proves

The crash/recovery scenario (book flight RA512 against the mock target) was
run once per crash-injection phase — all nine phases the demo supports, each
in its own clean runtime directory. After recovery in each run, side-effect
cardinality is asserted from the mock target store's **own**
`stats.total_commits` counter, never from framework logs.

Result: **exactly one commit per business action in all nine phases, zero
duplicates, order booked** (`ok: true` in `crash_matrix_summary.json`; the
sweep exits non-zero on any duplicate or missing commit).

The classification column shows the two recovery regimes, read from evidence
(commit counter before vs after restart), not from the phase name:

- Crash **before** the write dispatches (`after_step_started` …
  `before_tool_transaction`): the restarted process performs the write —
  `committed_after_restart`, commits 0 → 1.
- Crash **after** the write commits (`after_tool_transaction` …
  `after_record_final_message`): the restarted process reuses the durable
  result and never re-fires — `reused_committed_result`, commits stay at 1.
  (`after_advance_plan_item` shows only 2 step attempts because the completed
  plan item is skipped entirely on resume.)

## Files

- `crash_matrix_summary.json` — machine-readable: environment, per-phase rows
  (crash point, exit codes, attempts, interrupted attempts, commits before
  restart, final commits, duplicates, classification, recovery-phase ms,
  resume wall ms), overall `ok`.
- `crash_matrix_table.txt` — the same matrix as a human-readable table.

## Notes

- Per-run timing is fsync-dominated (`synchronous=FULL`); cold-cache runs can
  show higher absolute latencies than this (warm) capture. Cardinality results
  are unaffected.
- A reduced three-phase sweep plus a deliberate cardinality-mismatch case run
  in the test suite (`tests/test_crash_matrix.py`), so the invariant is
  enforced by `pytest` and CI, not only by this recorded run.
