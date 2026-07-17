"""Mock replay of the latest OpenRouter run using local infra only.

This sample replays the most recent OpenRouter soft-log run through the local
scripted mock agent, transaction manager, logging service, and merged recovery
coordinator. It can optionally simulate one ambiguous post-commit write using
existing AmbiguousToolError-based infrastructure.

Run from repository root:
PYTHONPATH=src python example/sample_agent_openrouter_mock.py
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import raef
from raef.evaluation import EvaluationRecorder
from raef.logging_service import LoggingService
from raef.recovery.common import RecoveryAction
from raef.recovery.recovery.handler import RecoveryCoordinator
from raef.tools import mock_agent
from raef.tools.crash_simulator import CrashSimulator
from raef.tools.mock_target import IdempotencyMode, JsonKVStore, MockTargetService
from raef.txn_manager import AmbiguousToolError, ToolAdapterProtocol, TransactionDisposition, TransactionManager
from raef.verifier import MockTargetVerifier

SOFT_LOG_ROOT = Path("./data/openrouter_ticket_runtime/soft_logs/runs")
RUNTIME_DIR = Path("./data/openrouter_mock_runtime")
TARGET_STORE_PATH = RUNTIME_DIR / "openrouter_mock_target.json"
INFERENCE_LOG_PATH = RUNTIME_DIR / "openrouter_mock_inference.jsonl"
SIMULATE_AMBIGUOUS_WRITE = True


def _seed_mock_target_store(store_path: Path, seed_state: dict[str, Any]) -> None:
    store = JsonKVStore(store_path)
    store.save(
        {
            "meta": {
                "version": 3,
                "created_at": "2026-04-11T00:00:00+00:00",
                "updated_at": "2026-04-11T00:00:00+00:00",
            },
            "stats": {
                "total_commits": 0,
                "idempotent_commits": 0,
                "rifl_commits": 0,
                "distinguishable_commits": 0,
                "counter_commits": 0,
            },
            "action_log": {},
            "domain_state": json.loads(json.dumps(seed_state)),
            "idempotent_requests": {},
            "rifl_requests": {},
            "distinguishable_state": {},
            "distinguishable_history": {},
            "counters": {},
        }
    )


def _append_jsonl(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


class ReplayToolAdapter(ToolAdapterProtocol):
    """Literal adapter that replays tool calls against the local mock target.

    Fault injection models the two directions of write ambiguity:
    - ``simulate_ambiguous_write``: the write commits at the target, then the
      response is lost (raise after apply). Recovery must prove the commit.
    - ``simulate_lost_write``: the request never reaches the target (raise
      before apply). Recovery must prove absence, then replay safely.
    """

    def __init__(
        self,
        target: MockTargetService,
        *,
        simulate_ambiguous_write: bool = False,
        simulate_lost_write: bool = False,
    ) -> None:
        self.target = target
        self.simulate_ambiguous_write = simulate_ambiguous_write
        self.simulate_lost_write = simulate_lost_write
        self._ambiguous_write_emitted = False

    def invoke(
        self,
        *,
        tool_name: str,
        request_payload: dict[str, Any],
        execution_id: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        del timeout_s
        if tool_name == "query-state":
            return self.target.query_state(
                query_name=str(request_payload["query_name"]),
                payload=dict(request_payload.get("payload", {})),
            )
        if tool_name == "apply-action":
            if self.simulate_lost_write and not self._ambiguous_write_emitted:
                self._ambiguous_write_emitted = True
                raise AmbiguousToolError("request lost in flight; target never observed the write")
            result = self.target.apply_action(
                action_name=str(request_payload["action_name"]),
                payload=dict(request_payload.get("payload", {})),
                execution_id=execution_id,
            )
            if self.simulate_ambiguous_write and not self._ambiguous_write_emitted:
                self._ambiguous_write_emitted = True
                raise AmbiguousToolError("write may have committed before the local process observed the response")
            return {
                **result,
                "applied_action": {
                    "action_name": str(request_payload["action_name"]),
                    "payload": dict(request_payload.get("payload", {})),
                    "execution_id": execution_id,
                },
            }
        raise ValueError(f"unsupported tool: {tool_name}")


@raef.with_logging_service(data_root=RUNTIME_DIR, checkpoint_every_n_events=4)
def run_sample_agent_openrouter_mock(
    *,
    run_id: str,
    logging_service: LoggingService,
    soft_log_path: Path | None = None,
    force_reset: bool = True,
    reset_target: bool = True,
    target_store_path: Path | None = None,
    inference_log_path: Path | None = None,
    simulate_ambiguous_write: bool = SIMULATE_AMBIGUOUS_WRITE,
    simulate_lost_write: bool = False,
    crash_simulator: CrashSimulator | None = None,
    crash_phase: str = "before_agent_turn",
    on_llm_response: Any | None = None,
) -> dict[str, Any]:
    artifact = _load_soft_log_payload(soft_log_path)
    script = _build_script_from_soft_log(artifact)
    initial_messages = _extract_initial_messages(artifact)
    plan_items = _build_plan_items_from_soft_log(artifact)
    plan_source_text = _build_plan_source_text(plan_items)
    seed_state = _extract_seed_state()

    logging_service.start_run(
        run_id=run_id,
        initial_messages=initial_messages,
        plan_source_text=plan_source_text,
        plan_items=plan_items,
        force_reset=force_reset,
    )

    resolved_target_store_path = target_store_path or TARGET_STORE_PATH
    resolved_target_store_path.parent.mkdir(parents=True, exist_ok=True)
    if reset_target:
        _seed_mock_target_store(resolved_target_store_path, seed_state)
    resolved_inference_log_path = inference_log_path or (resolved_target_store_path.parent / INFERENCE_LOG_PATH.name)
    target = MockTargetService(JsonKVStore(resolved_target_store_path), idempotency_mode=IdempotencyMode.IDEMPOTENT)
    coordinator = RecoveryCoordinator(
        logging_service,
        default_wait_seconds=0.25,
        verifier=MockTargetVerifier(target),
    )
    txn_manager = TransactionManager(logging_service)
    adapter = ReplayToolAdapter(
        target,
        simulate_ambiguous_write=simulate_ambiguous_write,
        simulate_lost_write=simulate_lost_write,
    )
    agent = mock_agent.ScriptedMockAgent(script=script)
    evaluator = EvaluationRecorder(logging_service)

    execution_summaries: list[dict[str, Any]] = []
    recovery_decisions: list[dict[str, Any]] = []
    trace: list[str] = []

    context = logging_service.context_service.load_context(run_id)
    if context is not None and not force_reset:
        messages = [
            mock_agent.ChatMessage(
                role=message.role,  # type: ignore[arg-type]
                content=message.content,
                name=message.name,
                tool_call_id=message.tool_call_id,
            )
            for message in context.messages
        ]
    else:
        messages = [mock_agent.ChatMessage(**message) for message in initial_messages]

    planner_state = logging_service.planner_service.load_plan(run_id)
    completed_plan_item_ids = {
        item.plan_item_id
        for item in (planner_state.items if planner_state is not None else [])
        if item.status.value == "done"
    }

    for step_index in range(len(script)):
        plan_item_id = f"step_{min(step_index, len(plan_items) - 1)}"
        if not force_reset and plan_item_id in completed_plan_item_ids and step_index < len(plan_items):
            trace.append(f"[step {step_index}] skipped completed plan_item_id={plan_item_id}")
            continue
        with evaluator.time_step(
            run_id=run_id,
            step_index=step_index,
            plan_item_id=plan_item_id,
            metadata={"script_length": len(script)},
        ) as step_span:
            _maybe_crash(crash_simulator, step_index, crash_phase, "after_step_started")
            with evaluator.time_phase(phase="llm_generate", parent_step=step_span):
                _maybe_crash(crash_simulator, step_index, crash_phase, "before_agent_turn")
                turn_request = mock_agent.AgentTurnRequest(
                    messages=messages,
                    available_tools=[],
                    context=mock_agent.AgentContext(run_id=run_id, step_index=step_index),
                )
                _append_jsonl(
                    resolved_inference_log_path,
                    {
                        "event": "request",
                        "model": "mock-openrouter-replay",
                        "run_id": run_id,
                        "step_index": step_index,
                        "payload": {
                            "messages": [asdict(message) for message in messages],
                            "available_tools": [],
                            "context": asdict(turn_request.context),
                        },
                    },
                )
                response = agent.generate_turn(turn_request)
                raw_response = response.raw_output or _response_to_json(response)
                if on_llm_response is not None:
                    on_llm_response(step_index, raw_response)
                _append_jsonl(
                    resolved_inference_log_path,
                    {
                        "event": "response",
                        "model": response.model_name,
                        "run_id": run_id,
                        "step_index": step_index,
                        "raw_text": raw_response,
                    },
                )
                _maybe_crash(crash_simulator, step_index, crash_phase, "after_agent_turn")

            if response.kind == mock_agent.DecisionKind.TOOL_CALL:
                assert response.tool_call is not None
                llm_output = response.raw_output or _response_to_json(response)
                assistant_message = response.tool_call.rationale or f"Calling {response.tool_call.name}"
                with evaluator.time_phase(phase="record_llm_turn", parent_step=step_span):
                    logging_service.record_llm_turn(
                        run_id=run_id,
                        plan_item_id=plan_item_id,
                        llm_output=llm_output,
                        assistant_message=assistant_message,
                        assistant_meta={"model_name": response.model_name, "decision_kind": response.kind.value},
                    )
                    _maybe_crash(crash_simulator, step_index, crash_phase, "after_record_llm_turn")

                with evaluator.time_phase(
                    phase="tool_transaction",
                    parent_step=step_span,
                    metadata={"tool_name": response.tool_call.name},
                ):
                    _maybe_crash(crash_simulator, step_index, crash_phase, "before_tool_transaction")
                    txn_result = txn_manager.execute_tool(
                        run_id=run_id,
                        plan_item_id=plan_item_id,
                        tool_name=response.tool_call.name,
                        request_payload=dict(response.tool_call.arguments),
                        operation_type=response.tool_call.operation_type.value,
                        adapter=adapter,
                        idempotency_supported=response.tool_call.operation_type == mock_agent.OperationType.WRITE,
                    )
                    _maybe_crash(crash_simulator, step_index, crash_phase, "after_tool_transaction")
                execution_summaries.append(
                    {
                        "execution_id": txn_result.execution_id,
                        "tool_name": response.tool_call.name,
                        "disposition": txn_result.disposition.value,
                        "execution_status": txn_result.execution_status.value,
                        "result_status": txn_result.result_status.value,
                    }
                )

                if txn_result.disposition == TransactionDisposition.FAILED:
                    raise txn_result.exception or RuntimeError(
                        f"tool execution failed for execution_id={txn_result.execution_id}"
                    )

                messages.append(mock_agent.ChatMessage(role="assistant", content=llm_output))

                if txn_result.disposition == TransactionDisposition.PENDING_RECOVERY:
                    record = logging_service.get_external_result(txn_result.execution_id)
                    now = record.updated_at + timedelta(seconds=1) if record is not None else None
                    with evaluator.time_phase(phase="recovery", parent_step=step_span):
                        decisions = coordinator.recover_run(run_id, now=now)
                        _maybe_crash(crash_simulator, step_index, crash_phase, "after_recovery")
                    recovery_decisions.extend(_decision_dicts(decisions))
                    trace.append(
                        f"[step {step_index}] {response.tool_call.name} pending_recovery -> "
                        f"execution_id={txn_result.execution_id} recovery={[d.action.value for d in decisions]}"
                    )
                    resolution = _resolve_recovered_execution(
                        decisions=decisions,
                        execution_id=txn_result.execution_id,
                        txn_manager=txn_manager,
                        logging_service=logging_service,
                        adapter=adapter,
                        run_id=run_id,
                        plan_item_id=plan_item_id,
                        tool_call=response.tool_call,
                    )
                    if resolution is not None:
                        outcome, tool_content = resolution
                        messages.append(
                            mock_agent.ChatMessage(
                                role="tool",
                                name=response.tool_call.name,
                                tool_call_id=txn_result.execution_id,
                                content=tool_content,
                            )
                        )
                        trace.append(
                            f"[step {step_index}] {response.tool_call.name} {outcome} -> "
                            f"execution_id={txn_result.execution_id}"
                        )
                else:
                    with evaluator.time_phase(phase="recovery", parent_step=step_span):
                        decisions = coordinator.recover_run(run_id)
                        _maybe_crash(crash_simulator, step_index, crash_phase, "after_recovery")
                    recovery_decisions.extend(_decision_dicts(decisions))
                    tool_content = json.dumps(txn_result.response_payload, sort_keys=True)
                    messages.append(
                        mock_agent.ChatMessage(
                            role="tool",
                            name=response.tool_call.name,
                            tool_call_id=txn_result.execution_id,
                            content=tool_content,
                        )
                    )
                    trace.append(
                        f"[step {step_index}] {response.tool_call.name} -> {json.dumps(txn_result.response_payload, sort_keys=True)}"
                    )

                with evaluator.time_phase(phase="advance_plan_item", parent_step=step_span):
                    logging_service.advance_plan_item(run_id=run_id, plan_item_id=plan_item_id, new_status="done")
                    _maybe_crash(crash_simulator, step_index, crash_phase, "after_advance_plan_item")
                continue

            assert response.final is not None
            with evaluator.time_phase(phase="record_final_message", parent_step=step_span):
                logging_service.record_context_message(
                    run_id=run_id,
                    role="assistant",
                    content=response.final.answer,
                    meta={"stop_reason": response.final.stop_reason, "model_name": response.model_name},
                )
                _maybe_crash(crash_simulator, step_index, crash_phase, "after_record_final_message")
            with evaluator.time_phase(phase="advance_plan_item", parent_step=step_span):
                logging_service.advance_plan_item(run_id=run_id, plan_item_id=plan_item_id, new_status="done")
                _maybe_crash(crash_simulator, step_index, crash_phase, "after_advance_plan_item")
            trace.append(f"[final] {response.final.answer}")

    checkpoint = logging_service.checkpoint(run_id)
    bundle = logging_service.get_recovery_bundle(run_id)
    evaluation_report = evaluator.build_report(run_id)
    return {
        "run_id": run_id,
        "source_soft_log": str(_resolve_soft_log_path(soft_log_path)),
        "checkpoint": checkpoint.to_dict() if checkpoint is not None else None,
        "bundle": bundle,
        "evaluation_report": evaluation_report,
        "executions": execution_summaries,
        "recovery_decisions": recovery_decisions,
        "trace": trace,
        "target_store_path": str(resolved_target_store_path),
        "recovery_summary": _summarize_recovery(bundle=bundle, recovery_decisions=recovery_decisions),
    }


def _resolve_recovered_execution(
    *,
    decisions: list[Any],
    execution_id: str,
    txn_manager: TransactionManager,
    logging_service: LoggingService,
    adapter: ToolAdapterProtocol,
    run_id: str,
    plan_item_id: str,
    tool_call: mock_agent.ToolCall,
) -> tuple[str, str] | None:
    """Act on the recovery decision for one ambiguous execution.

    Returns (outcome label, tool message content) when the execution reached a
    committed result, or None when it stays handed off to a human/policy.
    """

    decision = next((d for d in decisions if d.execution_id == execution_id), None)
    if decision is None:
        return None

    if decision.action == RecoveryAction.MARK_COMMITTED:
        record = logging_service.get_external_result(execution_id)
        payload = record.response_payload if record is not None else None
        return (
            "verified_committed (reused durable result)",
            json.dumps(payload or {"verified": True}, sort_keys=True),
        )

    if decision.action == RecoveryAction.REPLAY:
        replay_result = txn_manager.execute_tool(
            run_id=run_id,
            plan_item_id=plan_item_id,
            tool_name=tool_call.name,
            request_payload=dict(tool_call.arguments),
            operation_type=tool_call.operation_type.value,
            adapter=adapter,
            idempotency_supported=tool_call.operation_type == mock_agent.OperationType.WRITE,
        )
        if replay_result.disposition in {TransactionDisposition.SUCCEEDED, TransactionDisposition.REUSED}:
            return (
                "verified_not_found (replayed safely)",
                json.dumps(replay_result.response_payload, sort_keys=True),
            )
        return None

    return None


def _maybe_crash(
    crash_simulator: CrashSimulator | None,
    step_index: int,
    configured_phase: str,
    current_phase: str,
) -> None:
    if crash_simulator is None or configured_phase != current_phase:
        return
    crash_simulator.maybe_crash(step_index, phase=current_phase)


def _resolve_soft_log_path(soft_log_path: Path | None) -> Path:
    if soft_log_path is not None:
        return soft_log_path
    candidates = sorted(SOFT_LOG_ROOT.glob("sample-openrouter-run-*.json"))
    if not candidates:
        raise FileNotFoundError(f"no OpenRouter soft log runs found under {SOFT_LOG_ROOT}")
    return candidates[-1]


def _load_soft_log_payload(soft_log_path: Path | None) -> dict[str, Any]:
    path = _resolve_soft_log_path(soft_log_path)
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_initial_messages(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    messages = artifact.get("messages")
    if not isinstance(messages, list):
        messages = artifact.get("checkpoint", {}).get("snapshot_payload", {}).get("context", {}).get("messages", [])
    initial_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            break
        initial_messages.append({
            "role": "user",
            "content": str(message.get("content", "")),
        })
    if not initial_messages:
        initial_messages = [{"role": "user", "content": "Please buy me a ticket from LA to NY for 2026-05-14 09:00 AM PT."}]
    return initial_messages


def _build_plan_items_from_soft_log(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    planner_state = artifact.get("planner_state") or artifact.get("checkpoint", {}).get("snapshot_payload", {}).get("planner", {})
    items = planner_state.get("items", []) if isinstance(planner_state, dict) else []
    plan_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        entry: dict[str, Any] = {"title": str(item.get("title", "step"))}
        depends_on = item.get("depends_on")
        if isinstance(depends_on, list) and depends_on:
            entry["depends_on"] = [str(value) for value in depends_on]
        plan_items.append(entry)
    if not plan_items:
        raise ValueError("soft log did not contain planner items")
    return plan_items


def _build_plan_source_text(plan_items: list[dict[str, Any]]) -> str:
    return "\n".join(f"{index}. {item.get('title', 'step')}" for index, item in enumerate(plan_items, start=1))


def _build_script_from_soft_log(artifact: dict[str, Any]) -> list[mock_agent.AgentTurnResponse]:
    planner_state = artifact.get("planner_state") or artifact.get("checkpoint", {}).get("snapshot_payload", {}).get("planner", {})
    items = planner_state.get("items", []) if isinstance(planner_state, dict) else []
    if not isinstance(items, list) or not items:
        raise ValueError("soft log did not contain replayable planner items")

    script: list[mock_agent.AgentTurnResponse] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw = item.get("llm_output")
        if not isinstance(raw, str) or not raw.strip():
            continue
        payload = json.loads(raw)
        script.append(_parse_script_response(payload, raw_output=raw))

    final_answer = _extract_final_answer(artifact)
    if final_answer is None:
        raise ValueError("soft log did not contain a final assistant answer")
    script.append(
        mock_agent.AgentTurnResponse.from_final_answer(
            answer=final_answer["answer"],
            stop_reason=final_answer["stop_reason"],
            model_name=final_answer["model_name"],
            raw_output=json.dumps(
                {
                    "kind": "final",
                    "final": {
                        "answer": final_answer["answer"],
                        "stop_reason": final_answer["stop_reason"],
                    },
                },
                sort_keys=True,
            ),
        )
    )
    return script


def _parse_script_response(payload: dict[str, Any], *, raw_output: str) -> mock_agent.AgentTurnResponse:
    tool_call = payload.get("tool_call")
    if not isinstance(tool_call, dict):
        raise ValueError("expected tool_call payload in soft-log llm_output")
    operation_type = str(tool_call.get("operation_type", "READ")).upper()
    return mock_agent.AgentTurnResponse.from_tool_call(
        name=str(tool_call["name"]),
        arguments=dict(tool_call.get("arguments", {})),
        operation_type=mock_agent.OperationType(operation_type),
        rationale=tool_call.get("rationale"),
        model_name="mock-openrouter-replay",
        raw_output=raw_output,
    )


def _extract_final_answer(artifact: dict[str, Any]) -> dict[str, str] | None:
    messages = artifact.get("messages")
    if not isinstance(messages, list):
        messages = artifact.get("checkpoint", {}).get("snapshot_payload", {}).get("context", {}).get("messages", [])
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue
        metadata = message.get("metadata", {}) if isinstance(message.get("metadata"), dict) else {}
        if metadata.get("decision_kind") == "tool_call":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        return {
            "answer": content,
            "stop_reason": str(metadata.get("stop_reason", "completed")),
            "model_name": str(metadata.get("model_name", "mock-openrouter-replay")),
        }
    return None


def _extract_seed_state() -> dict[str, Any]:
    return {
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
        "flights/RA530": {
            "available": True,
            "departure_time": "2026-05-14T12:00:00-07:00",
            "flight_no": "RA530",
            "from": "LAX",
            "price_usd": 310,
            "to": "JFK",
        },
    }


def _decision_dicts(decisions: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "execution_id": decision.execution_id,
            "action": decision.action.value,
            "reason": decision.reason,
            "execution_status": decision.execution_status.value if decision.execution_status else None,
        }
        for decision in decisions
    ]


def _summarize_recovery(*, bundle: dict[str, Any], recovery_decisions: list[dict[str, Any]]) -> dict[str, Any]:
    context = bundle.get("context", {}) if isinstance(bundle, dict) else {}
    external_results = bundle.get("external_results", []) if isinstance(bundle, dict) else []
    pending_ids = context.get("pending_execution_ids", []) if isinstance(context, dict) else []
    successful_readback = any(
        isinstance(item, dict)
        and item.get("tool_name") == "query-state"
        and isinstance(item.get("response_payload", {}).get("result"), dict)
        and item.get("request_payload", {}).get("payload", {}).get("key") == "orders/RA512"
        for item in external_results
    )
    return {
        "pending_execution_ids": pending_ids,
        "had_handoff": any(item.get("action") == "handoff_retry_or_abandon" for item in recovery_decisions),
        "had_resume_waiting": any(item.get("action") == "resume_waiting" for item in recovery_decisions),
        "successful_order_readback": successful_readback,
    }


def _response_to_json(response: mock_agent.AgentTurnResponse) -> str:
    payload: dict[str, Any] = {
        "kind": response.kind.value,
        "model_name": response.model_name,
    }
    if response.tool_call is not None:
        payload["tool_call"] = {
            "name": response.tool_call.name,
            "arguments": response.tool_call.arguments,
            "operation_type": response.tool_call.operation_type.value,
            "rationale": response.tool_call.rationale,
        }
    if response.final is not None:
        payload["final"] = {
            "answer": response.final.answer,
            "stop_reason": response.final.stop_reason,
        }
    return json.dumps(payload, sort_keys=True)


def main() -> int:
    result = run_sample_agent_openrouter_mock(run_id=f"sample-openrouter-mock-run-{int(time.time())}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
