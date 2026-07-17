"""Local agent/LLM mock interface and deterministic scripted implementation.

This module defines a provider-agnostic turn API and a simple mock agent that can
be used by transaction and recovery middleware during local development.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Any, Literal, Protocol, Sequence, cast


class OperationType(str, Enum):
    """Operation type annotation for planned tool calls."""

    READ = "READ"
    WRITE = "WRITE"


class DecisionKind(str, Enum):
    """Agent output category for one generation turn."""

    TOOL_CALL = "tool_call"
    FINAL = "final"


MessageRole = Literal["system", "user", "assistant", "tool"]


class RequestStatus(Enum):
    NOT_SENT = "not_sent"
    IN_STEP = "in_step"
    REPLIED = "replied"


@dataclass(frozen=True)
class RequestRecord:
    request_id: str
    status: RequestStatus
    payload: dict[str, Any]


@dataclass(frozen=True)
class ChatMessage:
    """Conversation or tool transcript message."""

    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolDefinition:
    """Tool metadata presented to the model/agent each turn."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    side_effecting: bool = False


@dataclass(frozen=True)
class AgentContext:
    """Deterministic context carried across turns."""

    run_id: str
    step_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentTurnRequest:
    """Input contract for one agent turn."""

    messages: list[ChatMessage]
    available_tools: list[ToolDefinition]
    context: AgentContext
    max_output_tokens: int = 512
    temperature: float = 0.0


@dataclass(frozen=True)
class ToolCall:
    """Normalized tool call shape independent of LLM provider format."""

    name: str
    arguments: dict[str, Any]
    operation_type: OperationType = OperationType.READ
    rationale: str | None = None


@dataclass(frozen=True)
class FinalAnswer:
    """Final non-tool response from the agent."""

    answer: str
    stop_reason: str = "completed"


@dataclass(frozen=True)
class AgentTurnResponse:
    """Output contract for one agent turn."""

    kind: DecisionKind
    tool_call: ToolCall | None = None
    final: FinalAnswer | None = None
    model_name: str = "mock-scripted-v1"
    raw_output: str | None = None

    def __post_init__(self) -> None:
        is_tool = self.kind == DecisionKind.TOOL_CALL
        if is_tool and self.tool_call is None:
            raise ValueError("tool_call response requires a tool_call payload")
        if is_tool and self.final is not None:
            raise ValueError("tool_call response cannot include final payload")

        is_final = self.kind == DecisionKind.FINAL
        if is_final and self.final is None:
            raise ValueError("final response requires a final payload")
        if is_final and self.tool_call is not None:
            raise ValueError("final response cannot include tool_call payload")

    @classmethod
    def from_tool_call(
        cls,
        *,
        name: str,
        arguments: dict[str, Any],
        operation_type: OperationType = OperationType.READ,
        rationale: str | None = None,
        model_name: str = "mock-scripted-v1",
        raw_output: str | None = None,
    ) -> "AgentTurnResponse":
        _validate_jsonable(arguments, field_name="arguments")
        return cls(
            kind=DecisionKind.TOOL_CALL,
            tool_call=ToolCall(
                name=name,
                arguments=arguments,
                operation_type=operation_type,
                rationale=rationale,
            ),
            model_name=model_name,
            raw_output=raw_output,
        )

    @classmethod
    def from_final_answer(
        cls,
        *,
        answer: str,
        stop_reason: str = "completed",
        model_name: str = "mock-scripted-v1",
        raw_output: str | None = None,
    ) -> "AgentTurnResponse":
        return cls(
            kind=DecisionKind.FINAL,
            final=FinalAnswer(answer=answer, stop_reason=stop_reason),
            model_name=model_name,
            raw_output=raw_output,
        )


class LocalAgentLLMAPI(Protocol):
    """Stable local interface for mock and real LLM adapters."""

    def generate_turn(self, request: AgentTurnRequest) -> AgentTurnResponse:
        """Return one deterministic decision for the provided turn input."""

    def reset(self, run_id: str | None = None) -> None:
        """Reset adapter state for test isolation and new runs."""

    def get_state(self) -> dict[str, object]:
        """Expose adapter state for debugging and tests."""


class ScriptedMockAgent(LocalAgentLLMAPI):
    """Deterministic mock implementation that replays pre-baked turn responses.

    The script index is selected by request.context.step_index. This gives stable
    replay behavior across crash/recovery tests.
    """

    def __init__(
        self,
        script: Sequence[AgentTurnResponse],
        *,
        strict: bool = True,
        default_final_answer: str = "DONE",
    ) -> None:
        self._script = list(script)
        self._strict = strict
        self._default_final_answer = default_final_answer
        self._calls = 0
        self._last_run_id: str | None = None
        self._step_trace: list[int] = []

    def generate_turn(self, request: AgentTurnRequest) -> AgentTurnResponse:
        self._calls += 1
        self._last_run_id = request.context.run_id
        step_index = request.context.step_index
        self._step_trace.append(step_index)

        if step_index < 0:
            raise ValueError("step_index must be non-negative")

        if step_index >= len(self._script):
            if self._strict:
                raise IndexError(
                    f"script has {len(self._script)} steps, cannot serve step_index={step_index}"
                )
            return AgentTurnResponse.from_final_answer(
                answer=self._default_final_answer,
                model_name="mock-scripted-v1",
            )

        return self._script[step_index]

    def reset(self, run_id: str | None = None) -> None:
        if run_id is None or run_id == self._last_run_id:
            self._calls = 0
            self._step_trace = []
            self._last_run_id = None

    def get_state(self) -> dict[str, object]:
        return {
            "calls": self._calls,
            "last_run_id": self._last_run_id,
            "step_trace": list(self._step_trace),
            "script_length": len(self._script),
            "strict": self._strict,
        }


def build_default_transfer_script() -> list[AgentTurnResponse]:
    """Reference script used by local middleware demos and tests."""

    return [
        AgentTurnResponse.from_tool_call(
            name="get_balance",
            arguments={"account_id": "acct_A"},
            operation_type=OperationType.READ,
            rationale="Check source account before transfer",
        ),
        AgentTurnResponse.from_tool_call(
            name="create_transfer",
            arguments={"from": "acct_A", "to": "acct_B", "amount": 25},
            operation_type=OperationType.WRITE,
            rationale="Execute transfer",
        ),
        AgentTurnResponse.from_tool_call(
            name="get_transfer_status",
            arguments={"transfer_ref": "latest"},
            operation_type=OperationType.READ,
            rationale="Confirm transfer status",
        ),
        AgentTurnResponse.from_final_answer(
            answer="Transfer workflow complete.",
            stop_reason="completed",
        ),
    ]


def load_script_from_history_payload(payload: dict[str, Any]) -> list[AgentTurnResponse]:
    """Load scripted responses from a history/scenario payload object.

    Expected shape:
    - payload["steps"]: list of step objects
    - each step has kind=tool_call|final and corresponding fields
    """
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise ValueError("history payload must include a list field 'steps'")

    script: list[AgentTurnResponse] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"step[{index}] must be an object")

        kind = step.get("kind")
        if kind == DecisionKind.TOOL_CALL.value:
            operation_type = _parse_operation_type(step.get("operation_type", "READ"))
            arguments = step.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError(f"step[{index}].arguments must be an object")
            script.append(
                AgentTurnResponse.from_tool_call(
                    name=_require_str(step, "name", f"step[{index}]"),
                    arguments=arguments,
                    operation_type=operation_type,
                    rationale=step.get("rationale"),
                )
            )
            continue

        if kind == DecisionKind.FINAL.value:
            script.append(
                AgentTurnResponse.from_final_answer(
                    answer=_require_str(step, "answer", f"step[{index}]"),
                    stop_reason=str(step.get("stop_reason", "completed")),
                )
            )
            continue

        raise ValueError(
            f"step[{index}].kind must be '{DecisionKind.TOOL_CALL.value}' "
            f"or '{DecisionKind.FINAL.value}'"
        )

    return script


def build_request_records_from_script(
    script: Sequence[AgentTurnResponse],
    *,
    initial_status: RequestStatus = RequestStatus.NOT_SENT,
) -> list[RequestRecord]:
    """Build recovery-compatible request records for tool-call steps only."""

    requests: list[RequestRecord] = []
    for step_index, step in enumerate(script):
        if step.tool_call is None:
            continue
        requests.append(
            RequestRecord(
                request_id=f"request-step-{step_index}",
                status=initial_status,
                payload={
                    "step_index": step_index,
                    "tool_name": step.tool_call.name,
                    "arguments": step.tool_call.arguments,
                    "operation_type": step.tool_call.operation_type.value,
                    "rationale": step.tool_call.rationale,
                },
            )
        )
    return requests


def build_request_records_from_history_payload(
    payload: dict[str, Any],
    *,
    initial_status: RequestStatus = RequestStatus.NOT_SENT,
) -> list[RequestRecord]:
    """Build recovery-compatible request records from a history payload."""

    return build_request_records_from_script(
        load_script_from_history_payload(payload),
        initial_status=initial_status,
    )


def build_plan_items_from_history_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Build logging-service plan items from a history payload."""

    script = load_script_from_history_payload(payload)
    items: list[dict[str, Any]] = []
    for index, step in enumerate(script):
        title = step.tool_call.rationale if step.tool_call is not None else step.final.answer if step.final is not None else f"step {index}"
        item: dict[str, Any] = {"title": title}
        if index > 0:
            item["depends_on"] = [f"step_{index - 1}"]
        items.append(item)
    return items


def build_plan_source_text_from_history_payload(payload: dict[str, Any]) -> str:
    """Build a numbered plan summary from a history payload."""

    plan_items = build_plan_items_from_history_payload(payload)
    return "\n".join(f"{index}. {item['title']}" for index, item in enumerate(plan_items, start=1))


def load_script_from_history_file(file_path: str | Path) -> list[AgentTurnResponse]:
    """Load a deterministic script from a local JSON history/scenario file."""
    payload = load_history_payload(file_path)
    return load_script_from_history_payload(payload)


def load_messages_from_history_payload(payload: dict[str, Any]) -> list[ChatMessage]:
    """Load chat messages from optional history payload conversation entries."""
    conversation = payload.get("conversation", [])
    if not isinstance(conversation, list):
        raise ValueError("history payload field 'conversation' must be a list")

    messages: list[ChatMessage] = []
    for index, item in enumerate(conversation):
        if not isinstance(item, dict):
            raise ValueError(f"conversation[{index}] must be an object")
        raw_role = _require_str(item, "role", f"conversation[{index}]")
        content = _require_str(item, "content", f"conversation[{index}]")
        if raw_role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(
                f"conversation[{index}].role must be one of system,user,assistant,tool"
            )
        role = cast(MessageRole, raw_role)
        name = item.get("name")
        tool_call_id = item.get("tool_call_id")
        if name is not None and not isinstance(name, str):
            raise ValueError(f"conversation[{index}].name must be a string when provided")
        if tool_call_id is not None and not isinstance(tool_call_id, str):
            raise ValueError(
                f"conversation[{index}].tool_call_id must be a string when provided"
            )
        messages.append(
            ChatMessage(
                role=role,
                content=content,
                name=name,
                tool_call_id=tool_call_id,
            )
        )

    return messages


def load_history_payload(file_path: str | Path) -> dict[str, Any]:
    """Load and validate a history/scenario JSON object from disk."""
    path = Path(file_path)
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid history JSON at {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("history file payload must be a JSON object")
    return payload


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scripted mock agent CLI")
    parser.add_argument(
        "--history-file",
        required=True,
        help="Path to history/scenario JSON file in the expected schema.",
    )
    parser.add_argument(
        "--step-index",
        type=int,
        default=0,
        help="Step index to generate (default: 0).",
    )
    parser.add_argument(
        "--run-id",
        default="mock-agent-cli-run",
        help="Run id used in AgentContext.",
    )
    parser.add_argument(
        "--allow-step-overflow",
        action="store_true",
        help="Return final fallback when step index exceeds script length.",
    )
    parser.add_argument(
        "--show-script-info",
        action="store_true",
        help="Print script and history metadata before output.",
    )
    return parser


def _response_to_dict(response: AgentTurnResponse) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": response.kind.value,
        "model_name": response.model_name,
    }
    if response.raw_output is not None:
        payload["raw_output"] = response.raw_output

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

    return payload


def run_cli(argv: list[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    try:
        history_payload = load_history_payload(args.history_file)
        script = load_script_from_history_payload(history_payload)
        messages = load_messages_from_history_payload(history_payload)

        if args.show_script_info:
            script_info = {
                "scenario_id": history_payload.get("scenario_id"),
                "description": history_payload.get("description"),
                "script_steps": len(script),
                "conversation_messages": len(messages),
            }
            print(json.dumps(script_info, separators=(",", ":"), sort_keys=True))

        agent = ScriptedMockAgent(
            script=script,
            strict=not args.allow_step_overflow,
        )
        request = AgentTurnRequest(
            messages=messages,
            available_tools=[],
            context=AgentContext(run_id=args.run_id, step_index=args.step_index),
        )
        response = agent.generate_turn(request)
        print(json.dumps(_response_to_dict(response), separators=(",", ":"), sort_keys=True))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2


def main() -> int:
    return run_cli()


def _parse_operation_type(raw: Any) -> OperationType:
    if isinstance(raw, OperationType):
        return raw
    if isinstance(raw, str):
        normalized = raw.strip().upper()
        if normalized == OperationType.READ.value:
            return OperationType.READ
        if normalized == OperationType.WRITE.value:
            return OperationType.WRITE
    raise ValueError("operation_type must be READ or WRITE")


def _require_str(obj: dict[str, Any], key: str, ctx: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{ctx}.{key} must be a non-empty string")
    return value


def _validate_jsonable(value: Any, *, field_name: str) -> None:
    try:
        json.dumps(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be JSON-serializable") from exc


if __name__ == "__main__":
    raise SystemExit(main())
