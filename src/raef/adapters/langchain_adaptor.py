"""LangChain-oriented adapter helpers for RAEF logging.

This module focuses on logging-relevant behavior:
- tool metadata gates (read-only/concurrency-safe/permission checks),
- replay-safe identity helpers for tool executions/messages.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable

from ..logging_service import LoggingService
from ..txn_manager import TransactionManager, build_execution_id as txn_build_execution_id


class ToolPermissionError(RuntimeError):
    """Raised when a tool permission gate denies a call."""


@dataclass(frozen=True)
class ToolPolicy:
    """Metadata gates used by middleware policy and logging."""

    is_read_only: bool
    is_concurrency_safe: bool = False
    requires_permission: bool = False
    permission_label: str | None = None


def build_execution_id(
    *,
    run_id: str,
    plan_item_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    invocation_key: str | None = None,
) -> str:
    """Build deterministic execution id for replay-safe tool identity."""

    return txn_build_execution_id(
        run_id=run_id,
        plan_item_id=plan_item_id,
        tool_name=tool_name,
        arguments=arguments,
        invocation_key=invocation_key,
    )


def build_message_id(
    *,
    run_id: str,
    role: str,
    content: str,
    index_hint: int,
) -> str:
    """Build deterministic message id for replay-safe transcript identity."""

    raw = f"{run_id}|{role}|{index_hint}|{content}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return f"msg_{digest}"


def wrap_langchain_tool(
    *,
    run_id: str,
    plan_item_id: str,
    tool_name: str,
    tool_fn: Callable[[dict[str, Any]], dict[str, Any]],
    logging_service: LoggingService,
    policy: ToolPolicy,
    permission_check: Callable[[dict[str, Any]], bool] | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Wrap one tool function with RAEF logging + policy gates.

    Returned callable is framework-agnostic, but can be used directly as the
    body for a LangChain tool.
    """

    operation_type = "READ" if policy.is_read_only else "WRITE"

    def wrapped(arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be a dictionary")

        if policy.requires_permission:
            granted = False
            if permission_check is not None:
                granted = bool(permission_check(arguments))
            if not granted:
                label = policy.permission_label or tool_name
                raise ToolPermissionError(f"permission denied for tool: {label}")

        manager = TransactionManager(logging_service, dispatch_reads=True)
        result = manager.execute_callable(
            run_id=run_id,
            plan_item_id=plan_item_id,
            tool_name=tool_name,
            request_payload=arguments,
            operation_type=operation_type,
            tool_fn=tool_fn,
        )
        if result.exception is not None:
            raise result.exception
        if result.response_payload is None:
            raise RuntimeError(f"missing response payload for execution_id={result.execution_id}")
        return result.response_payload

    return wrapped
