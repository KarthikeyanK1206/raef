# Post-improvement verification run

- Date: 2026-07-16
- Machine: Apple M4 (arm64), macOS 26.5.1 (build 25F80)
- Python: 3.13.13 (project venv)

## Commands and captured outputs

| File | Command | Result |
| --- | --- | --- |
| `quality_gates.txt` | `pytest -q` / `ruff check .` / `mypy src` / `py_compile example/*.py` | 45 passed; ruff clean; mypy clean (33 files); examples compile |
| `demo1_crash.txt` | `python example/crash_recovery_e2e.py --clean` | exit 0; attempt 1 interrupted, attempt 2 succeeded via replay reuse |
| `demo2_ambiguous.txt` | `python example/crash_recovery_e2e.py --clean --runtime-dir data/demo_ambiguous --simulate-ambiguous-write` | exit 0; write resolved to `verified_committed` by target verification |
| `demo3_lost.txt` | `python example/crash_recovery_e2e.py --clean --runtime-dir data/demo_lost_write --simulate-lost-write` | exit 0; write resolved via `verified_not_found` then safe replay to `acked` |
| `db_verification.txt` | SQLite + target-store readback for all three scenarios | every scenario ends with exactly 1 target commit, empty pending set, order booked |
| `cli_transcript.txt` | `raef runs / show / executions / events / report / recover --dry-run / audit` | audit reports "log-first invariants hold" |
| `evaluation_timing_demo1.txt` | copied from demo 1 runtime dir | per-phase durable timing report |
| `evaluation_summary_demo2.json` | copied from demo 2 runtime dir | full evaluation/recovery report |

## Headline result

Across a hard process kill, a committed-but-unacknowledged write, and a lost
write, the run always completes with **exactly one commit at the target** —
verified from the target store's own `stats.total_commits` counter, not from
framework logs.
