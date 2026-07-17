"""OpenRouter-backed sample agent wired to RAEF logging.

Run from repository root:
python example/sample_agent_openrouter.py
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import raef
from raef.evaluation import EvaluationRecorder
from raef.logging_service import LoggingService
from raef.recovery.recovery.handler import RecoveryCoordinator
from raef.tools import mock_agent
from raef.tools.crash_simulator import CrashSimulator
from raef.tools.mock_target import IdempotencyMode, JsonKVStore, MockTargetService
from raef.txn_manager import AmbiguousToolError, ToolAdapterProtocol, TransactionDisposition, TransactionManager

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "minimax/minimax-m2.5:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
]
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

SCENARIO_FILE = Path("./data/mock_agent_history_ticket_la_to_ny.json")
RUNTIME_DIR = Path("./data/openrouter_ticket_runtime")
TARGET_STORE_PATH = RUNTIME_DIR / "openrouter_mock_target.json"
DEBUG_LOG_PATH = RUNTIME_DIR / "openrouter_debug.log"
INFERENCE_LOG_PATH = RUNTIME_DIR / "openrouter_inference.jsonl"

STABLE_USER_PROMPT = "Please buy me a ticket from LA to NY for 2026-05-14 09:00 AM PT."
MAX_STEPS = 8
MODEL_RETRY_DELAY_SECONDS = 1.0
LOGGER_NAME = "example.sample_agent_openrouter"
SIMULATE_AMBIGUOUS_WRITE = os.getenv("RAEF_SIMULATE_AMBIGUOUS_WRITE", "").strip() == "1"


def _configure_logger(runtime_dir: Path | None = None) -> logging.Logger:
    resolved_runtime_dir = runtime_dir or RUNTIME_DIR
    resolved_runtime_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(resolved_runtime_dir / DEBUG_LOG_PATH.name, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def _logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def _cli_log(message: str, *, level: int = logging.INFO) -> None:
    _logger().log(level, message)


def _short_json(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True)
    return rendered if len(rendered) <= 260 else rendered[:257] + "..."


def _openrouter_models_from_env() -> list[str]:
    raw = os.getenv("OPENROUTER_MODELS") or os.getenv("OPENROUTER_MODEL")
    if not raw:
        return list(OPENROUTER_MODELS)
    return [model.strip() for model in raw.split(",") if model.strip()]


def _append_jsonl(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


class OpenRouterToolAdapter(ToolAdapterProtocol):
    """Transaction-manager adapter for the OpenRouter sample target."""

    def __init__(
        self,
        target: MockTargetService,
        *,
        scenario: dict[str, Any],
        simulate_ambiguous_write: bool = False,
    ) -> None:
        self.target = target
        self.scenario = scenario
        self.simulate_ambiguous_write = simulate_ambiguous_write
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
            query_name, payload = _normalize_query_state_arguments(request_payload, scenario=self.scenario)
            _cli_log(
                f"[tool] normalized query-state query_name={query_name} payload={_short_json(payload)}",
                level=logging.DEBUG,
            )
            return self.target.query_state(query_name=query_name, payload=payload)
        if tool_name == "apply-action":
            action_name, payload, resolved_execution_id = _normalize_apply_action_arguments(
                request_payload,
                fallback_execution_id=execution_id or "",
                scenario=self.scenario,
            )
            payload = _validate_apply_action_payload(
                action_name=action_name,
                payload=payload,
            )
            _cli_log(
                f"[tool] normalized apply-action action_name={action_name} execution_id={resolved_execution_id} payload={_short_json(payload)}",
                level=logging.DEBUG,
            )
            result = self.target.apply_action(
                action_name=action_name,
                payload=payload,
                execution_id=resolved_execution_id,
            )
            if self.simulate_ambiguous_write and not self._ambiguous_write_emitted:
                self._ambiguous_write_emitted = True
                raise AmbiguousToolError("write may have committed before the local process observed the response")
            return {
                **result,
                "applied_action": {
                    "action_name": action_name,
                    "payload": payload,
                    "execution_id": resolved_execution_id,
                },
            }
        raise ValueError(f"unsupported tool name: {tool_name}")


@raef.with_logging_service(data_root=RUNTIME_DIR, checkpoint_every_n_events=4)
def run_sample_agent_openrouter(
    *,
    run_id: str,
    user_prompt: str,
    logging_service: LoggingService,
    initial_messages: list[dict[str, str]] | None = None,
    scenario_file: Path | None = None,
    force_reset: bool = True,
    reset_target: bool = True,
    target_store_path: Path | None = None,
    model_names: list[str] | None = None,
    inference_log_path: Path | None = None,
    simulate_ambiguous_write: bool = SIMULATE_AMBIGUOUS_WRITE,
    crash_simulator: CrashSimulator | None = None,
    crash_phase: str = "before_agent_turn",
) -> dict[str, object]:
    """Execute a model-driven ticket-booking flow against the local mock target."""
    del user_prompt
    del initial_messages

    scenario = mock_agent.load_history_payload(scenario_file or SCENARIO_FILE)
    _cli_log(f"[run] scenario={scenario.get('scenario_id', 'unknown')} prompt={STABLE_USER_PROMPT}")
    plan_items = _build_plan_items(scenario)
    resolved_target_store_path = target_store_path or TARGET_STORE_PATH
    if reset_target:
        _seed_mock_target_store(resolved_target_store_path, scenario.get("target_seed_state", {}))

    stable_messages = [{"role": "user", "content": STABLE_USER_PROMPT}]
    logging_service.start_run(
        run_id=run_id,
        initial_messages=stable_messages,
        plan_source_text=_build_plan_source_text(scenario),
        plan_items=plan_items,
        force_reset=force_reset,
    )

    target = MockTargetService(
        JsonKVStore(resolved_target_store_path),
        idempotency_mode=IdempotencyMode.IDEMPOTENT,
    )
    coordinator = RecoveryCoordinator(logging_service, default_wait_seconds=0.25)
    txn_manager = TransactionManager(logging_service)
    adapter = OpenRouterToolAdapter(
        target,
        scenario=scenario,
        simulate_ambiguous_write=simulate_ambiguous_write,
    )
    agent = OpenRouterAgent(
        api_key=OPENROUTER_API_KEY,
        model_names=model_names or _openrouter_models_from_env(),
        scenario=scenario,
        inference_log_path=inference_log_path or (resolved_target_store_path.parent / INFERENCE_LOG_PATH.name),
    )
    evaluator = EvaluationRecorder(logging_service)

    execution_summaries: list[dict[str, Any]] = []
    recovery_decisions: list[dict[str, Any]] = []

    context = logging_service.context_service.load_context(run_id)
    planner_state = logging_service.planner_service.load_plan(run_id)
    if context is not None and not force_reset:
        persisted_messages = [
            mock_agent.ChatMessage(
                role=message.role,  # type: ignore[arg-type]
                content=message.content,
                name=message.name,
                tool_call_id=message.tool_call_id,
            )
            for message in context.messages
        ]
        persisted_messages = _restore_raw_llm_outputs(persisted_messages, planner_state)
        messages = [mock_agent.ChatMessage(role="system", content=_build_system_prompt(scenario)), *persisted_messages]
    else:
        messages = [
            mock_agent.ChatMessage(role="system", content=_build_system_prompt(scenario)),
            mock_agent.ChatMessage(role="user", content=STABLE_USER_PROMPT),
        ]
    available_tools = _build_available_tools()
    final_result: dict[str, Any] | None = None
    completed_plan_item_ids = {
        item.plan_item_id
        for item in (planner_state.items if planner_state is not None else [])
        if item.status.value == "done"
    }

    for step_index in range(MAX_STEPS):
        plan_item_id = f"step_{min(step_index, len(plan_items) - 1)}"
        if not force_reset and plan_item_id in completed_plan_item_ids and step_index < len(plan_items):
            _cli_log(f"[step {step_index}] skipping completed plan_item_id={plan_item_id}")
            continue
        with evaluator.time_step(
            run_id=run_id,
            step_index=step_index,
            plan_item_id=plan_item_id,
            metadata={"scenario_id": scenario.get("scenario_id", "unknown")},
        ) as step_span:
            _maybe_crash(crash_simulator, step_index, crash_phase, "after_step_started")
            _cli_log(f"[step {step_index}] requesting next decision")
            with evaluator.time_phase(phase="llm_generate", parent_step=step_span):
                _maybe_crash(crash_simulator, step_index, crash_phase, "before_agent_turn")
                response = agent.generate_turn(
                    mock_agent.AgentTurnRequest(
                        messages=messages,
                        available_tools=available_tools,
                        context=mock_agent.AgentContext(
                            run_id=run_id,
                            step_index=step_index,
                            metadata={"scenario_id": scenario.get("scenario_id", "unknown")},
                        ),
                        max_output_tokens=500,
                        temperature=0.0,
                    )
                )
                _maybe_crash(crash_simulator, step_index, crash_phase, "after_agent_turn")

            if response.kind == mock_agent.DecisionKind.TOOL_CALL:
                assert response.tool_call is not None
                llm_output = response.raw_output or _response_to_json(response)
                assistant_message = response.tool_call.rationale or f"Calling {response.tool_call.name}"
                _cli_log(
                    f"[step {step_index}] model={response.model_name} tool={response.tool_call.name} args={_short_json(response.tool_call.arguments)}"
                )
                with evaluator.time_phase(phase="record_llm_turn", parent_step=step_span):
                    logging_service.record_llm_turn(
                        run_id=run_id,
                        plan_item_id=plan_item_id,
                        llm_output=llm_output,
                        assistant_message=assistant_message,
                        assistant_meta={
                            "model_name": response.model_name,
                            "decision_kind": response.kind.value,
                        },
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
                        request_payload=response.tool_call.arguments,
                        operation_type=response.tool_call.operation_type.value,
                        adapter=adapter,
                        idempotency_supported=response.tool_call.operation_type == mock_agent.OperationType.WRITE,
                    )
                    _maybe_crash(crash_simulator, step_index, crash_phase, "after_tool_transaction")
                execution_summary = {
                    "execution_id": txn_result.execution_id,
                    "tool_name": response.tool_call.name,
                    "disposition": txn_result.disposition.value,
                    "execution_status": txn_result.execution_status.value,
                    "result_status": txn_result.result_status.value,
                }
                execution_summaries.append(execution_summary)

                messages.append(mock_agent.ChatMessage(role="assistant", content=llm_output))

                if txn_result.disposition == TransactionDisposition.FAILED:
                    raise txn_result.exception or RuntimeError(
                        f"tool execution failed for execution_id={txn_result.execution_id}"
                    )

                if txn_result.disposition == TransactionDisposition.PENDING_RECOVERY:
                    record = logging_service.get_external_result(txn_result.execution_id)
                    now = None
                    if record is not None:
                        now = record.updated_at + timedelta(seconds=1)
                    with evaluator.time_phase(phase="recovery", parent_step=step_span):
                        decisions = coordinator.recover_run(run_id, now=now)
                        _maybe_crash(crash_simulator, step_index, crash_phase, "after_recovery")
                    recovery_decisions.extend(
                        {
                            "execution_id": decision.execution_id,
                            "action": decision.action.value,
                            "reason": decision.reason,
                            "execution_status": decision.execution_status.value if decision.execution_status else None,
                        }
                        for decision in decisions
                    )
                    _cli_log(
                        f"[step {step_index}] pending_recovery execution_id={txn_result.execution_id} "
                        f"recovery={[d.action.value for d in decisions]}"
                    )
                else:
                    with evaluator.time_phase(phase="recovery", parent_step=step_span):
                        decisions = coordinator.recover_run(run_id)
                        _maybe_crash(crash_simulator, step_index, crash_phase, "after_recovery")
                    recovery_decisions.extend(
                        {
                            "execution_id": decision.execution_id,
                            "action": decision.action.value,
                            "reason": decision.reason,
                            "execution_status": decision.execution_status.value if decision.execution_status else None,
                        }
                        for decision in decisions
                    )
                    tool_content = json.dumps(txn_result.response_payload, sort_keys=True)
                    messages.append(
                        mock_agent.ChatMessage(
                            role="tool",
                            name=response.tool_call.name,
                            tool_call_id=txn_result.execution_id,
                            content=tool_content,
                        )
                    )
                    _cli_log(f"[step {step_index}] tool_result={_short_json(txn_result.response_payload)}")

                with evaluator.time_phase(phase="advance_plan_item", parent_step=step_span):
                    logging_service.advance_plan_item(
                        run_id=run_id,
                        plan_item_id=plan_item_id,
                        new_status="done",
                    )
                    _maybe_crash(crash_simulator, step_index, crash_phase, "after_advance_plan_item")
                continue

            assert response.final is not None
            _cli_log(f"[step {step_index}] final model={response.model_name} answer={response.final.answer}")
            with evaluator.time_phase(phase="record_final_message", parent_step=step_span):
                logging_service.record_context_message(
                    run_id=run_id,
                    role="assistant",
                    content=response.final.answer,
                    meta={
                        "stop_reason": response.final.stop_reason,
                        "model_name": response.model_name,
                    },
                )
                _maybe_crash(crash_simulator, step_index, crash_phase, "after_record_final_message")
            final_plan_item_id = f"step_{len(plan_items) - 1}"
            with evaluator.time_phase(phase="advance_plan_item", parent_step=step_span):
                logging_service.advance_plan_item(
                    run_id=run_id,
                    plan_item_id=final_plan_item_id,
                    new_status="done",
                )
                _maybe_crash(crash_simulator, step_index, crash_phase, "after_advance_plan_item")
            checkpoint = logging_service.checkpoint(run_id)
            final_result = {
                "run_id": run_id,
                "scenario_id": scenario.get("scenario_id"),
                "model_name": response.model_name,
                "final_answer": response.final.answer,
                "checkpoint": checkpoint.to_dict() if checkpoint is not None else None,
                "bundle": logging_service.get_recovery_bundle(run_id),
                "executions": execution_summaries,
                "recovery_decisions": recovery_decisions,
                "target_store_path": str(resolved_target_store_path),
            }

        if final_result is not None:
            final_result["evaluation_report"] = evaluator.build_report(run_id)
            return final_result

    raise RuntimeError(f"agent did not return a final answer within {MAX_STEPS} steps")


def _maybe_crash(
    crash_simulator: CrashSimulator | None,
    step_index: int,
    configured_phase: str,
    current_phase: str,
) -> None:
    if crash_simulator is None or configured_phase != current_phase:
        return
    crash_simulator.maybe_crash(step_index, phase=current_phase)


class OpenRouterAgent(mock_agent.LocalAgentLLMAPI):
    """Tiny OpenRouter adapter that returns normalized AgentTurnResponse objects."""

    def __init__(
        self,
        *,
        api_key: str,
        model_names: list[str],
        scenario: dict[str, Any],
        inference_log_path: Path | None = None,
    ) -> None:
        if not api_key or api_key == "PASTE_OPENROUTER_API_KEY_HERE":
            raise ValueError("Set OPENROUTER_API_KEY before running this example.")
        cleaned_model_names = [name.strip() for name in model_names if name.strip()]
        if not cleaned_model_names:
            raise ValueError("model_names must include at least one non-empty model")
        self._api_key = api_key
        self._model_names = cleaned_model_names
        self._scenario = scenario
        self._inference_log_path = inference_log_path
        self._calls = 0
        self._last_model_name: str | None = None

    def generate_turn(
        self,
        request: mock_agent.AgentTurnRequest,
    ) -> mock_agent.AgentTurnResponse:
        self._calls += 1
        failures: list[str] = []

        for model_name in self._model_names:
            _cli_log(f"[llm] trying model={model_name}")
            payload = {
                "model": model_name,
                "temperature": request.temperature,
                "response_format": {"type": "json_object"},
                "messages": _build_openrouter_messages(request, self._scenario),
            }
            _append_jsonl(
                self._inference_log_path,
                {
                    "event": "request",
                    "call_index": self._calls,
                    "model": model_name,
                    "run_id": request.context.run_id,
                    "step_index": request.context.step_index,
                    "payload": payload,
                },
            )
            try:
                try:
                    raw_text = self._post_json(payload)
                except RuntimeError as exc:
                    if not _is_json_mode_unsupported(str(exc)):
                        raise
                    fallback_payload = dict(payload)
                    fallback_payload.pop("response_format", None)
                    _cli_log(f"[llm] model={model_name} does not support JSON mode; retrying without response_format")
                    _append_jsonl(
                        self._inference_log_path,
                        {
                            "event": "request",
                            "call_index": self._calls,
                            "model": model_name,
                            "run_id": request.context.run_id,
                            "step_index": request.context.step_index,
                            "payload": fallback_payload,
                            "retry_reason": "json_mode_unsupported",
                        },
                    )
                    raw_text = self._post_json(fallback_payload)
                _append_jsonl(
                    self._inference_log_path,
                    {
                        "event": "response",
                        "call_index": self._calls,
                        "model": model_name,
                        "run_id": request.context.run_id,
                        "step_index": request.context.step_index,
                        "raw_text": raw_text,
                    },
                )
                parsed = json.loads(raw_text)
                self._last_model_name = model_name
                _cli_log(f"[llm] model={model_name} responded")
                _cli_log(f"[llm] raw_response={raw_text}", level=logging.DEBUG)
                return _parse_model_response(parsed, model_name=model_name, raw_output=raw_text)
            except RuntimeError as exc:
                _append_jsonl(
                    self._inference_log_path,
                    {
                        "event": "error",
                        "call_index": self._calls,
                        "model": model_name,
                        "run_id": request.context.run_id,
                        "step_index": request.context.step_index,
                        "error": str(exc),
                    },
                )
                failures.append(f"{model_name}: {exc}")
                _cli_log(f"[llm] model={model_name} failed: {exc}")
                if _should_try_next_model(str(exc)):
                    time.sleep(MODEL_RETRY_DELAY_SECONDS)
                    continue
                raise
            except ValueError as exc:
                _append_jsonl(
                    self._inference_log_path,
                    {
                        "event": "error",
                        "call_index": self._calls,
                        "model": model_name,
                        "run_id": request.context.run_id,
                        "step_index": request.context.step_index,
                        "error": f"invalid JSON contract: {exc}",
                    },
                )
                failures.append(f"{model_name}: invalid JSON contract: {exc}")
                _cli_log(f"[llm] model={model_name} invalid JSON contract: {exc}")
                if _should_try_next_model(str(exc)):
                    continue
                raise

        joined = "\n".join(failures)
        raise RuntimeError(f"OpenRouter failed across free models:\n{joined}")

    def reset(self, run_id: str | None = None) -> None:
        del run_id
        self._calls = 0
        self._last_model_name = None

    def get_state(self) -> dict[str, object]:
        return {
            "adapter": "openrouter-chat-completions",
            "model_names": list(self._model_names),
            "last_model_name": self._last_model_name,
            "calls": self._calls,
        }

    def _post_json(self, payload: dict[str, Any]) -> str:
        body = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            OPENROUTER_ENDPOINT,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/openai/codex",
                "X-Title": "RAEF OpenRouter Sample Agent",
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP error {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc

        try:
            response_payload = json.loads(raw)
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("OpenRouter response was not in the expected chat format") from exc

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            merged = "".join(text_parts).strip()
            if merged:
                return merged
        raise RuntimeError("OpenRouter response content did not contain JSON text")


def _build_available_tools() -> list[mock_agent.ToolDefinition]:
    return [
        mock_agent.ToolDefinition(
            name="query-state",
            description=(
                "Read data from the mock target state store. Use query_name=list_keys with payload prefix to list records, "
                "or query_name=get_value with payload key to fetch a single record."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query_name": {"type": "string"},
                    "payload": {"type": "object"},
                },
                "required": ["query_name", "payload"],
            },
            side_effecting=False,
        ),
        mock_agent.ToolDefinition(
            name="apply-action",
            description=(
                "Write data to the mock target state store. Use action_name=set_value with payload key and value to persist an order."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "action_name": {"type": "string"},
                    "payload": {"type": "object"},
                    "execution_id": {"type": "string"},
                },
                "required": ["action_name", "payload"],
            },
            side_effecting=True,
        ),
    ]


def _build_system_prompt(scenario: dict[str, Any]) -> str:
    return (
        "You are a deterministic agent that must respond with exactly one JSON object.\n"
        "Return either a tool call or a final answer using this schema:\n"
        '{'
        '"kind":"tool_call"|"final",'
        '"tool_call":{"name":str,"arguments":object,"operation_type":"READ"|"WRITE","rationale":str},'
        '"final":{"answer":str,"stop_reason":str}'
        "}\n"
        "Rules:\n"
        "- Do not include markdown fences.\n"
        "- Use only the provided tools.\n"
        "- For tool_call, include name, arguments, operation_type, and rationale.\n"
        "- For final, include answer and stop_reason.\n"
        "- Base decisions on tool results already present in the conversation.\n"
        "- For this ticket-booking scenario, first inspect available flights, then persist one order, then confirm the saved order.\n"
        "- When referring to the booking after persistence, use the canonical order id from tool results instead of inventing an order/... key.\n"
        f"Scenario id: {scenario.get('scenario_id', 'unknown')}.\n"
        f"Scenario description: {scenario.get('description', '')}"
    )


def _build_openrouter_messages(
    request: mock_agent.AgentTurnRequest,
    scenario: dict[str, Any],
) -> list[dict[str, Any]]:
    tools_block = json.dumps([asdict(tool) for tool in request.available_tools], sort_keys=True)
    request_block = json.dumps(
        {
            "context": asdict(request.context),
            "available_tools": [asdict(tool) for tool in request.available_tools],
            "seed_state_preview": scenario.get("target_seed_state", {}),
            "expected_plan_shape": [
                {
                    "kind": step.get("kind"),
                    "name": step.get("name"),
                    "rationale": step.get("rationale"),
                }
                for step in _scenario_steps(scenario)
            ],
        },
        sort_keys=True,
    )

    messages: list[dict[str, Any]] = []
    for message in request.messages:
        content = message.content
        if message.role == "system":
            content = (
                f"{message.content}\n\n"
                f"Current agent request envelope:\n{request_block}\n\n"
                f"Available tools:\n{tools_block}"
            )
        if message.role == "tool":
            tool_name = message.name or "tool"
            messages.append({"role": "user", "content": f"Tool result from {tool_name}: {content}"})
            continue
        messages.append({"role": message.role, "content": content})
    return messages


def _parse_model_response(
    payload: dict[str, Any],
    *,
    model_name: str,
    raw_output: str,
) -> mock_agent.AgentTurnResponse:
    kind = payload.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        if isinstance(payload.get("tool_call"), dict):
            kind = mock_agent.DecisionKind.TOOL_CALL.value
        elif isinstance(payload.get("final"), dict) or isinstance(payload.get("answer"), str):
            kind = mock_agent.DecisionKind.FINAL.value

    if kind == mock_agent.DecisionKind.TOOL_CALL.value:
        tool_call = payload.get("tool_call")
        if not isinstance(tool_call, dict):
            raise ValueError("tool_call response must include an object field named 'tool_call'")
        name = tool_call.get("name")
        arguments = tool_call.get("arguments")
        operation_type = str(tool_call.get("operation_type", "READ")).upper()
        rationale = tool_call.get("rationale")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tool_call.name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise ValueError("tool_call.arguments must be a JSON object")
        if operation_type not in {"READ", "WRITE"}:
            raise ValueError("tool_call.operation_type must be READ or WRITE")
        if rationale is not None and not isinstance(rationale, str):
            raise ValueError("tool_call.rationale must be a string when provided")
        return mock_agent.AgentTurnResponse.from_tool_call(
            name=name,
            arguments=arguments,
            operation_type=mock_agent.OperationType(operation_type),
            rationale=rationale,
            model_name=model_name,
            raw_output=raw_output,
        )

    if kind == mock_agent.DecisionKind.FINAL.value:
        final = payload.get("final")
        if final is None and isinstance(payload.get("answer"), str):
            final = {
                "answer": payload.get("answer"),
                "stop_reason": payload.get("stop_reason", "completed"),
            }
        if not isinstance(final, dict):
            raise ValueError("final response must include an object field named 'final'")
        answer = final.get("answer")
        stop_reason = final.get("stop_reason", "completed")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("final.answer must be a non-empty string")
        if not isinstance(stop_reason, str) or not stop_reason.strip():
            raise ValueError("final.stop_reason must be a non-empty string")
        return mock_agent.AgentTurnResponse.from_final_answer(
            answer=answer,
            stop_reason=stop_reason,
            model_name=model_name,
            raw_output=raw_output,
        )

    raise ValueError("model response must set kind to 'tool_call' or 'final'")


def _should_try_next_model(message: str) -> bool:
    lowered = message.lower()
    return (
        "http error 404" in lowered
        or "http error 408" in lowered
        or "http error 429" in lowered
        or "http error 500" in lowered
        or "http error 502" in lowered
        or "http error 503" in lowered
        or "http error 504" in lowered
        or "no endpoints found" in lowered
        or "temporarily rate-limited" in lowered
    )


def _is_json_mode_unsupported(message: str) -> bool:
    lowered = message.lower()
    return "json mode is not supported" in lowered or "response_format" in lowered and "not supported" in lowered


def _seed_mock_target_store(store_path: Path, seed_state: dict[str, Any]) -> None:
    store = JsonKVStore(store_path)
    data = {
        "meta": {
            "version": 3,
            "created_at": "2026-04-09T00:00:00+00:00",
            "updated_at": "2026-04-09T00:00:00+00:00",
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
    store.save(data)


def _build_plan_source_text(scenario: dict[str, Any]) -> str:
    return "\n".join(
        f"{index}. {_step_title(step)}" for index, step in enumerate(_scenario_steps(scenario), start=1)
    )


def _build_plan_items(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, step in enumerate(_scenario_steps(scenario)):
        item: dict[str, Any] = {"title": _step_title(step)}
        if index > 0:
            item["depends_on"] = [f"step_{index - 1}"]
        items.append(item)
    return items


def _scenario_steps(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    steps = scenario.get("steps", [])
    if not isinstance(steps, list) or not steps:
        raise ValueError("scenario must contain a non-empty 'steps' list")
    normalized = [step for step in steps if isinstance(step, dict)]
    if not normalized:
        raise ValueError("scenario steps must be JSON objects")
    return normalized


def _step_title(step: dict[str, Any]) -> str:
    if step.get("kind") == "tool_call":
        return str(step.get("rationale") or step.get("name") or "tool call")
    return str(step.get("answer") or "final answer")


def _expected_write_step(scenario: dict[str, Any]) -> dict[str, Any] | None:
    for step in _scenario_steps(scenario):
        if step.get("kind") == "tool_call" and str(step.get("operation_type", "")).upper() == "WRITE":
            return step
    return None


def _expected_write_payload(scenario: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    step = _expected_write_step(scenario)
    if step is None:
        return None, None
    arguments = step.get("arguments")
    if not isinstance(arguments, dict):
        return None, None
    payload = arguments.get("payload")
    if not isinstance(payload, dict):
        return None, None
    key = payload.get("key")
    value = payload.get("value")
    return key if isinstance(key, str) else None, value if isinstance(value, dict) else None


def _resolve_execution_id(tool_call: mock_agent.ToolCall, step_index: int, *, run_id: str) -> str:
    raw_execution_id = tool_call.arguments.get("execution_id")
    if isinstance(raw_execution_id, str) and raw_execution_id.strip():
        safe_execution_id = raw_execution_id.strip().replace("/", "-")
        return f"{run_id}-{safe_execution_id}"
    return f"{run_id}-openrouter-exec-{step_index}"


def _normalize_query_state_arguments(
    arguments: dict[str, Any],
    *,
    scenario: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    del scenario
    raw_query_name_text = str(arguments.get("query_name", "")).strip()
    raw_query_name = raw_query_name_text.lower()
    payload = dict(arguments.get("payload", {})) if isinstance(arguments.get("payload"), dict) else {}

    if not payload:
        if isinstance(arguments.get("key"), str):
            payload["key"] = arguments["key"]
        if isinstance(arguments.get("prefix"), str):
            payload["prefix"] = arguments["prefix"]
        if isinstance(arguments.get("order_id"), str):
            payload["key"] = f"orders/{arguments['order_id']}"
        if isinstance(arguments.get("flight_no"), str):
            payload["key"] = f"flights/{arguments['flight_no']}"

    if "/" in raw_query_name_text and "key" not in payload and "prefix" not in payload:
        payload["key"] = raw_query_name_text
        raw_query_name = "get_value"

    if raw_query_name in {"read", "get", "lookup", "fetch"}:
        if "key" in payload:
            raw_query_name = "get_value"
        elif "prefix" in payload:
            raw_query_name = "list_keys"
        else:
            raw_query_name = "dump_state"
    elif raw_query_name in {"list", "list_values", "list_state", "list_flights", "search_flights", "available_flights"}:
        payload.setdefault("prefix", "flights/")
        raw_query_name = "list_keys"
    elif raw_query_name not in {"get_value", "list_keys", "dump_state"}:
        if "key" in payload:
            raw_query_name = "get_value"
        elif "prefix" in payload:
            raw_query_name = "list_keys"
        else:
            raw_query_name = "dump_state"

    return raw_query_name, payload


def _normalize_apply_action_arguments(
    arguments: dict[str, Any],
    *,
    fallback_execution_id: str,
    scenario: dict[str, Any],
) -> tuple[str, dict[str, Any], str]:
    del scenario
    raw_action_name = str(arguments.get("action_name", "")).strip().lower()
    payload = dict(arguments.get("payload", {})) if isinstance(arguments.get("payload"), dict) else {}

    if not payload:
        if isinstance(arguments.get("key"), str):
            payload["key"] = arguments["key"]
        if "value" in arguments:
            payload["value"] = arguments["value"]
        if isinstance(arguments.get("order_id"), str):
            payload["order_id"] = arguments["order_id"]
        if isinstance(arguments.get("flight_no"), str):
            payload["flight_no"] = arguments["flight_no"]

    resolved_execution_id = (
        str(arguments["execution_id"])
        if isinstance(arguments.get("execution_id"), str) and str(arguments.get("execution_id")).strip()
        else fallback_execution_id
    )

    if raw_action_name in {"set_value", "delete_value", "increment_value"}:
        return raw_action_name, payload, resolved_execution_id
    if raw_action_name in {"book_ticket", "purchase_ticket", "create_order", "place_order", "book_flight", "reserve_flight", "ticket_order"}:
        return "set_value", payload, resolved_execution_id
    if raw_action_name in {"write", "set", "update", "store"}:
        return "set_value", payload, resolved_execution_id
    if raw_action_name in {"delete", "remove"}:
        return "delete_value", payload, resolved_execution_id
    if raw_action_name in {"increment", "add"}:
        return "increment_value", payload, resolved_execution_id

    if "value" in payload:
        return "set_value", payload, resolved_execution_id
    if "delta" in payload:
        return "increment_value", payload, resolved_execution_id
    return "set_value", payload, resolved_execution_id


def _validate_apply_action_payload(
    *,
    action_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(payload)
    key = normalized.get("key")
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"apply-action {action_name} requires payload.key to be a non-empty string")

    if action_name == "set_value":
        if "value" not in normalized:
            raise ValueError("apply-action set_value requires payload.value; the adapter no longer invents one")
        return normalized

    if action_name == "increment_value":
        delta = normalized.get("delta", 1)
        if not isinstance(delta, (int, float)):
            raise ValueError("apply-action increment_value requires payload.delta to be numeric when provided")
        return normalized

    if action_name == "delete_value":
        return normalized

    raise ValueError(f"unsupported apply-action name: {action_name}")


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


def _restore_raw_llm_outputs(
    messages: list[mock_agent.ChatMessage],
    planner_state: Any,
) -> list[mock_agent.ChatMessage]:
    """Use persisted planner JSON outputs for prior assistant tool-call turns."""

    if planner_state is None:
        return messages
    raw_outputs = [
        item.llm_output
        for item in getattr(planner_state, "items", [])
        if isinstance(getattr(item, "llm_output", None), str) and getattr(item, "llm_output").strip()
    ]
    if not raw_outputs:
        return messages

    restored: list[mock_agent.ChatMessage] = []
    raw_index = 0
    for message in messages:
        if (
            message.role == "assistant"
            and raw_index < len(raw_outputs)
            and not message.content.lstrip().startswith("{")
        ):
            restored.append(
                mock_agent.ChatMessage(
                    role=message.role,
                    content=raw_outputs[raw_index],
                    name=message.name,
                    tool_call_id=message.tool_call_id,
                )
            )
            raw_index += 1
            continue
        if message.role == "assistant" and message.content.lstrip().startswith("{"):
            raw_index += 1
        restored.append(message)
    return restored


def main() -> int:
    _configure_logger()
    try:
        result = run_sample_agent_openrouter(
            run_id=f"sample-openrouter-run-{int(time.time())}",
            user_prompt=STABLE_USER_PROMPT,
            initial_messages=[{"role": "user", "content": STABLE_USER_PROMPT}],
        )
    except Exception as exc:
        _cli_log(f"[error] run failed: {exc}", level=logging.ERROR)
        _cli_log(f"[error] debug_log_path={DEBUG_LOG_PATH}", level=logging.ERROR)
        _cli_log(f"[error] runtime_dir={RUNTIME_DIR}", level=logging.ERROR)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
