"""Real LLM adapter contract and lightweight HTTP implementation.

This module keeps the same turn-based interface as the mock agent so runtime
code can swap providers without changing orchestration logic.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from . import mock_agent


class RealLLMAPIError(RuntimeError):
    """Raised when a remote LLM adapter cannot complete a turn."""


class HttpJSONLLMAdapter(mock_agent.LocalAgentLLMAPI):
    """Minimal HTTP adapter that returns normalized AgentTurnResponse objects.

    Expected remote response schema:
    {
      "kind": "tool_call" | "final",
      "model_name": "optional",
      "raw_output": "optional",
      "tool_call": {
        "name": "...",
        "arguments": {...},
        "operation_type": "READ" | "WRITE",
        "rationale": "optional"
      },
      "final": {
        "answer": "...",
        "stop_reason": "optional"
      }
    }
    """

    def __init__(
        self,
        *,
        endpoint: str,
        model_name: str,
        api_key_env: str | None = None,
        timeout_seconds: int = 30,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if not endpoint.strip():
            raise ValueError("endpoint must be a non-empty string")
        if not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")

        self._endpoint = endpoint
        self._model_name = model_name
        self._api_key_env = api_key_env
        self._timeout_seconds = timeout_seconds
        self._extra_headers = dict(extra_headers or {})
        self._calls = 0
        self._last_status: str | None = None

    def generate_turn(
        self,
        request: mock_agent.AgentTurnRequest,
    ) -> mock_agent.AgentTurnResponse:
        self._calls += 1
        payload = {
            "model": self._model_name,
            "request": _request_to_payload(request),
        }

        try:
            response_payload = self._post_json(payload)
            response = _parse_turn_response(response_payload, default_model=self._model_name)
            self._last_status = "ok"
            return response
        except RealLLMAPIError:
            self._last_status = "error"
            raise

    def reset(self, run_id: str | None = None) -> None:
        del run_id
        self._calls = 0
        self._last_status = None

    def get_state(self) -> dict[str, object]:
        return {
            "adapter": "http-json",
            "endpoint": self._endpoint,
            "model_name": self._model_name,
            "calls": self._calls,
            "last_status": self._last_status,
        }

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            **self._extra_headers,
        }

        if self._api_key_env is not None:
            token = os.getenv(self._api_key_env)
            if token is None or not token.strip():
                raise RealLLMAPIError(
                    f"missing API key environment variable: {self._api_key_env}"
                )
            headers["Authorization"] = f"Bearer {token}"

        req = urllib_request.Request(
            self._endpoint,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib_request.urlopen(req, timeout=self._timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RealLLMAPIError(f"LLM HTTP error {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:
            raise RealLLMAPIError(f"LLM request failed: {exc.reason}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RealLLMAPIError("LLM response is not valid JSON") from exc

        if not isinstance(parsed, dict):
            raise RealLLMAPIError("LLM response must be a JSON object")
        return parsed


def _request_to_payload(request: mock_agent.AgentTurnRequest) -> dict[str, Any]:
    return {
        "messages": [asdict(message) for message in request.messages],
        "available_tools": [asdict(tool) for tool in request.available_tools],
        "context": asdict(request.context),
        "max_output_tokens": request.max_output_tokens,
        "temperature": request.temperature,
    }


def _parse_turn_response(
    payload: dict[str, Any],
    *,
    default_model: str,
) -> mock_agent.AgentTurnResponse:
    kind = payload.get("kind")
    model_name = payload.get("model_name")
    raw_output = payload.get("raw_output")
    resolved_model = str(model_name) if isinstance(model_name, str) and model_name else default_model

    if kind == mock_agent.DecisionKind.TOOL_CALL.value:
        tool_call = payload.get("tool_call")
        if not isinstance(tool_call, dict):
            raise RealLLMAPIError("tool_call response requires object field 'tool_call'")

        name = tool_call.get("name")
        arguments = tool_call.get("arguments")
        operation_type = tool_call.get("operation_type", mock_agent.OperationType.READ.value)
        rationale = tool_call.get("rationale")

        if not isinstance(name, str) or not name.strip():
            raise RealLLMAPIError("tool_call.name must be a non-empty string")
        if not isinstance(arguments, dict):
            raise RealLLMAPIError("tool_call.arguments must be an object")
        if rationale is not None and not isinstance(rationale, str):
            raise RealLLMAPIError("tool_call.rationale must be a string when provided")

        try:
            parsed_operation_type = _parse_operation_type(operation_type)
        except ValueError as exc:
            raise RealLLMAPIError(str(exc)) from exc

        return mock_agent.AgentTurnResponse.from_tool_call(
            name=name,
            arguments=arguments,
            operation_type=parsed_operation_type,
            rationale=rationale,
            model_name=resolved_model,
            raw_output=raw_output if isinstance(raw_output, str) else None,
        )

    if kind == mock_agent.DecisionKind.FINAL.value:
        final = payload.get("final")
        if not isinstance(final, dict):
            raise RealLLMAPIError("final response requires object field 'final'")
        answer = final.get("answer")
        stop_reason = final.get("stop_reason", "completed")

        if not isinstance(answer, str) or not answer.strip():
            raise RealLLMAPIError("final.answer must be a non-empty string")
        if not isinstance(stop_reason, str) or not stop_reason.strip():
            raise RealLLMAPIError("final.stop_reason must be a non-empty string")

        return mock_agent.AgentTurnResponse.from_final_answer(
            answer=answer,
            stop_reason=stop_reason,
            model_name=resolved_model,
            raw_output=raw_output if isinstance(raw_output, str) else None,
        )

    raise RealLLMAPIError("response.kind must be 'tool_call' or 'final'")


def _parse_operation_type(raw: Any) -> mock_agent.OperationType:
    if isinstance(raw, mock_agent.OperationType):
        return raw
    if isinstance(raw, str):
        normalized = raw.strip().upper()
        if normalized == mock_agent.OperationType.READ.value:
            return mock_agent.OperationType.READ
        if normalized == mock_agent.OperationType.WRITE.value:
            return mock_agent.OperationType.WRITE
    raise ValueError("tool_call.operation_type must be READ or WRITE")