# RAEF — Reliable Agent Execution Framework

RAEF makes LLM-agent tool execution durable across process crashes, restarts, and ambiguous network failures. It is a log-first runtime: every tool call is journaled to SQLite *before* it is dispatched, given a deterministic execution ID, and driven through an explicit state machine — so a restarted agent replays to exactly where it stopped, reuses results that already committed, and never blindly re-fires a write it cannot prove the outcome of.

The core package has **zero runtime dependencies** (stdlib `sqlite3`, `json`, `hashlib`), runs entirely locally, and ships with a mock target service and crash injector so every failure mode is reproducible on a laptop — no API keys, no network.

## Why this exists

An agent that books a flight, sends an email, or writes to a database has a side effect it cannot take back. If the process dies after the request is sent but before the response is recorded, a naive retry double-books and a naive skip loses the booking. RAEF's answer:

1. **Log before send.** The intent (tool, arguments, deterministic execution ID) is durable before the network call starts.
2. **Deterministic identity.** `execution_id = sha256(run_id | plan_item_id | tool_name | canonical_args)` — a restarted process recomputes the same ID and finds the durable record.
3. **Explicit ambiguity.** A write whose outcome is unknown raises `AmbiguousToolError` and is parked in `pending_recovery`; it is never silently retried.
4. **Verification-driven recovery.** A pluggable verifier probes the target: proven commit → reuse the result; proven absence → replay exactly once; inconclusive → hand off to a human/policy.

## Execution state machine

```
intent_logged ──> dispatched ──> acked ──────────────┐
                     │                               ├──> (terminal, reusable)
                     │  ambiguous (timeout/crash)    │
                     ▼                               │
              [pending_recovery]                     │
                     │ verify against target         │
        ┌────────────┴─────────────┐                 │
        ▼                          ▼                 │
verified_committed        verified_not_found ──replay──> acked
  (reuse result)            (safe to re-send)
```

Illegal transitions are rejected by the runtime (`logging_service._ALLOWED_EXECUTION_TRANSITIONS`), so a bug in orchestration code cannot, for example, move a verified-committed write back to dispatched.

## Architecture

```
                 agent loop (yours)
                        │
        ┌───────────────┼───────────────────┐
        ▼               ▼                   ▼
  EvaluationRecorder  TransactionManager  RecoveryCoordinator
  (durable timing     (log-before-send,   (decide: resume wait /
   spans; crashes      dedup, replay,      replay / mark committed /
   show up as          ambiguity)          handoff; optional verifier)
   "interrupted")       │                   │
        └───────────────┼───────────────────┘
                        ▼
                  LoggingService  ── facade; per-run lock; event validation
                        │
      ┌─────────────────┼──────────────────┐
      ▼                 ▼                  ▼
 PlannerState      AgentContext       WAL events +
 (plan items,      (messages,         checkpoints
  statuses,         memory, pending
  cursor)           executions)
      └─────────────────┼──────────────────┘
                        ▼
              SQLiteRuntimeStore (single file, WAL mode,
              synchronous=FULL, payload artifact offloading)
                        │
                        ▼ (best-effort, non-canonical)
              soft JSON mirrors for humans
```

- `src/raef/logging_service.py` — the facade. Every mutation appends validated WAL events and updates projections inside one SQLite transaction under a per-run lock.
- `src/raef/txn_manager.py` — `TransactionManager.execute_tool(...)`: dedup on deterministic IDs, log-before-send, result caching, ambiguity classification, and safe replay of `verified_not_found` executions.
- `src/raef/recovery/` — `RecoveryCoordinator` scans durable executions after a restart and decides per record; wire a `verifier` to resolve ambiguous writes against the target instead of handing off.
- `src/raef/verifier.py` — commit probe for the mock target (`WriteVerifierProtocol` is the contract for real targets).
- `src/raef/evaluation.py` — durable timing spans; open spans surface as `interrupted` in reports after a crash.
- `src/raef/cli.py` — operator CLI over any runtime store (`raef runs / show / events / executions / pending / report / recover / audit`).
- `src/raef/tools/` — mock agent, mock target (idempotent / RIFL-style / distinguishable APIs), crash injector.
- `src/raef/adapters/` — decorators and a LangChain-style tool wrapper.

## Guarantees (and honest limits)

- Tool intents, results, plan state, messages, and timing spans survive `kill -9` at any point: SQLite WAL journal with `synchronous=FULL`.
- Completed writes are never re-executed on replay: dedup by deterministic execution ID.
- Ambiguous writes are never silently retried: they park in `pending_recovery` until verified or handed off.
- With a verifier, both ambiguity directions converge to **exactly one commit at the target** — this is demonstrated by tests and the demos below. This property leans on the target: replay-after-verified-absence is fully safe when the target dedupes by execution ID (idempotency key). Against a target with no idempotency support there remains a fundamental window (verify says absent, the original late request lands after the replay) that no client alone can close.
- Single-process, single-writer per run by design. Local-first: durability is per-machine; there is no replication.

## Quickstart

Requires Python 3.12+.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

### Run the crash/recovery demos (mock-only, no network)

```bash
# 1. Hard crash: process killed right after a committed write; restart replays
#    the run and reuses the committed result.
python example/crash_recovery_e2e.py --clean

# 2. Ambiguous write, response lost: the write committed but the ack never
#    arrived. Recovery probes the target, proves the commit, reuses it.
python example/crash_recovery_e2e.py --clean --simulate-ambiguous-write

# 3. Ambiguous write, request lost: the write never reached the target.
#    Recovery proves the absence, then replays exactly once.
python example/crash_recovery_e2e.py --clean --simulate-lost-write

# 4. Full crash matrix: run the scenario once per crash-injection phase (all
#    nine), asserting from the target store's own stats.total_commits counter
#    that every phase ends with exactly one commit. Exits non-zero on any
#    duplicate or missing commit; writes crash_matrix_summary.json + a table.
python example/crash_recovery_e2e.py --sweep-crash-phases --clean
```

Each run writes durable artifacts under `data/openrouter_mock_crash_e2e/` (or `--runtime-dir`): `runtime.sqlite` (canonical store), `evaluation_summary.json`, `evaluation_timing.txt`, the mock target state, and an inference trace. Crash point is configurable: `--crash-step N --crash-phase after_tool_transaction` (phases: `after_step_started`, `before_agent_turn`, `after_agent_turn`, `after_record_llm_turn`, `before_tool_transaction`, `after_tool_transaction`, `after_recovery`, `after_advance_plan_item`, `after_record_final_message`). The sweep accepts `--sweep-phases a,b,c` to run a subset and `--sweep-root DIR` for its output; recorded matrix evidence lives in `docs/artifacts/2026-07-16-crash-matrix/`.

### Inspect any run with the CLI

```bash
raef --data-root data/openrouter_mock_crash_e2e runs
raef --data-root data/openrouter_mock_crash_e2e show <run_id>
raef --data-root data/openrouter_mock_crash_e2e executions <run_id>
raef --data-root data/openrouter_mock_crash_e2e events <run_id>
raef --data-root data/openrouter_mock_crash_e2e report <run_id>
raef --data-root data/openrouter_mock_crash_e2e recover <run_id> --dry-run
raef --data-root data/openrouter_mock_crash_e2e audit <run_id>   # invariant checks
```

All commands accept `--json` for machine-readable output. `audit` exits non-zero when log-first invariants are violated (dangling pending executions, intents without result records, checkpoint ahead of the log).

## Using RAEF in your agent

```python
from pathlib import Path

import raef


def my_tool(arguments: dict) -> dict:
    return {"ok": True, "echo": arguments}


run_id = "demo-run-1"
logging_service = raef.LoggingService.with_data_root(
    Path("./data/my_agent_runtime"),
    checkpoint_every_n_events=4,
)
txn_manager = raef.TransactionManager(logging_service)
evaluator = raef.EvaluationRecorder(logging_service)

logging_service.start_run(
    run_id=run_id,
    initial_messages=[{"role": "user", "content": "book a ticket"}],
    plan_source_text="1. Book the selected flight",
    plan_items=[{"title": "Book the selected flight"}],
)

with evaluator.time_step(run_id=run_id, step_index=0, plan_item_id="step_0") as step:
    with evaluator.time_phase(phase="tool_transaction", parent_step=step):
        result = txn_manager.execute_callable(
            run_id=run_id,
            plan_item_id="step_0",
            tool_name="my_tool",
            request_payload={"flight_no": "RA512"},
            operation_type="WRITE",
            tool_fn=my_tool,
            idempotency_supported=True,
        )

    logging_service.record_context_message(
        run_id=run_id,
        role="tool",
        content=str(result.response_payload),
        tool_call_id=result.execution_id,
        name="my_tool",
    )
    logging_service.advance_plan_item(run_id=run_id, plan_item_id="step_0", new_status="done")
```

Re-running this block with the same `run_id` reuses the committed result (`disposition == REUSED`) instead of re-invoking the tool.

For object-style tools implement `ToolAdapterProtocol` (an `invoke(...)` method that receives the execution ID as an idempotency key); for recovery wire a verifier:

```python
coordinator = raef.RecoveryCoordinator(
    logging_service,
    verifier=my_target_verifier,   # implements raef.WriteVerifierProtocol
)
decisions = coordinator.recover_run(run_id)
```

Write tools should pass `idempotency_supported=True` whenever the target accepts the execution ID as an idempotency key; ambiguous outcomes should raise `raef.AmbiguousToolError`.

## Other examples

- `python example/sample_agent_openrouter_mock.py` — replay an OpenRouter soft log through the local mock stack.
- `python example/sample_agent_openrouter.py` — model-driven flow against the mock target (needs an OpenRouter API key).
- `python example/openrouter_crash_recovery_e2e.py` — crash/recovery for the OpenRouter-backed path (needs a key).

## Development

```bash
pytest            # 47 tests: WAL, txn manager, recovery+verification, verifier, evaluation, CLI, crash matrix
ruff check .
mypy src
```

CI (`.github/workflows/ci.yml`) runs lint, types, the test suite, and all three crash/recovery demos on Python 3.12 and 3.13.

Repository layout notes:

- `data/` is regenerable runtime output and is git-ignored; never commit it.
- The SQLite store is canonical; JSON files under `soft_logs/` are best-effort mirrors for humans.
- Optional extras: `pip install -e '.[http]'` for the HTTP health checker, `'.[mcp]'` for the FastMCP mock-target server.
