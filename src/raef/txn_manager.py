"""Transaction manager module for RAEF.

Intercept tool calls, assign execution identifiers, guard against duplicate
requests, and coordinate log-before-send behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import logging
from typing import Any, Protocol

from .agent_state import ExternalResultRecord, ExternalResultStatus
from .logging_service import LoggingService
from .models import ExecutionStatus, OperationType

logger = logging.getLogger(__name__)


class AmbiguousToolError(RuntimeError):
    """Raised when a dispatched tool call may already have executed remotely."""


class TransactionDisposition(str, Enum):
    """High-level outcome for one logical tool invocation."""

    SUCCEEDED = "succeeded"
    REUSED = "reused"
    PENDING_RECOVERY = "pending_recovery"
    FAILED = "failed"


@dataclass(slots=True)
class TransactionResult:
    """Return value from the transaction manager."""

    execution_id: str
    disposition: TransactionDisposition
    record: ExternalResultRecord
    exception: Exception | None = None

    @property
    def response_payload(self) -> dict[str, Any] | None:
        return self.record.response_payload

    @property
    def execution_status(self) -> ExecutionStatus:
        return self.record.execution_status

    @property
    def result_status(self) -> ExternalResultStatus:
        return self.record.result_status


class ToolAdapterProtocol(Protocol):
    """Adapter contract for invoking a tool under transaction control."""

    def invoke(
        self,
        *,
        tool_name: str,
        request_payload: dict[str, Any],
        execution_id: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Execute the tool call and return a dictionary payload."""


class FunctionToolAdapter:
    """Adapter that wraps a simple ``tool_fn(arguments) -> dict`` callable."""

    def __init__(self, tool_fn: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.tool_fn = tool_fn

    def invoke(
        self,
        *,
        tool_name: str,
        request_payload: dict[str, Any],
        execution_id: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        del tool_name, execution_id, timeout_s
        return self.tool_fn(request_payload)


def build_execution_id(
    *,
    run_id: str,
    plan_item_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    invocation_key: str | None = None,
) -> str:
    """Build deterministic execution id for replay-safe tool identity."""

    if not isinstance(arguments, dict):
        raise ValueError("arguments must be a dictionary")
    canonical_args = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    raw_parts = [run_id, plan_item_id, tool_name]
    if invocation_key is not None:
        raw_parts.append(invocation_key)
    raw_parts.append(canonical_args)
    raw = "|".join(raw_parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"exec_{digest}"


class TransactionManager:
    """Live-path coordinator for one logical tool invocation."""

    _REUSABLE_SUCCESS_STATUSES = {
        ExecutionStatus.ACKED,
        ExecutionStatus.VERIFIED_COMMITTED,
    }
    _PENDING_RECOVERY_STATUSES = {
        ExecutionStatus.INTENT_LOGGED,
        ExecutionStatus.DISPATCHED,
        ExecutionStatus.VERIFIED_NOT_FOUND,
    }

    def __init__(self, logging_service: LoggingService, *, dispatch_reads: bool = False) -> None:
        self.logging_service = logging_service
        self.dispatch_reads = dispatch_reads # if the read only calls also assign a DISPATCHED state

    def execute_tool(
        self,
        *,
        run_id: str,
        plan_item_id: str,
        tool_name: str,
        request_payload: dict[str, Any],
        operation_type: OperationType | str,
        adapter: ToolAdapterProtocol,
        invocation_key: str | None = None,
        timeout_s: float | None = None,
        idempotency_supported: bool = False,
    ) -> TransactionResult:
        """Execute one tool call with durable log-before-send behavior."""

        normalized_operation = _normalize_operation_type(operation_type)
        execution_id = build_execution_id(
            run_id=run_id,
            plan_item_id=plan_item_id,
            tool_name=tool_name,
            arguments=request_payload,
            invocation_key=invocation_key,
        )

        # A verified_not_found record is the one prior state that is proven
        # safe to re-execute: recovery probed the target and found no commit.
        is_verified_replay = False
        existing = self.logging_service.get_external_result(execution_id)
        if existing is not None:
            if existing.execution_status != ExecutionStatus.VERIFIED_NOT_FOUND:
                logger.debug(
                    "reusing existing execution: execution_id=%s status=%s",
                    execution_id,
                    existing.execution_status.value,
                )
                return self._handle_existing(existing)
            is_verified_replay = True
            logger.info(
                "replaying verified_not_found execution: execution_id=%s tool=%s",
                execution_id,
                tool_name,
            )

        if not is_verified_replay:
            self.logging_service.record_tool_intent(
                run_id=run_id,
                plan_item_id=plan_item_id,
                execution_id=execution_id,
                tool_name=tool_name,
                operation_type=normalized_operation.value,
                request_payload=request_payload,
            )

            current = self.logging_service.get_external_result(execution_id)
            if current is None:
                raise RuntimeError(f"tool intent missing after logging for execution_id={execution_id}")
            if current.execution_status != ExecutionStatus.INTENT_LOGGED:
                return self._handle_existing(current)

            # verified_not_found -> dispatched is not a legal transition, so
            # the dispatch marker is only written on first execution.
            if normalized_operation == OperationType.WRITE or self.dispatch_reads:
                self.logging_service.record_tool_dispatch(run_id=run_id, execution_id=execution_id)

        try:
            response_payload = adapter.invoke(
                tool_name=tool_name,
                request_payload=request_payload,
                execution_id=execution_id if normalized_operation == OperationType.WRITE and idempotency_supported else None,
                timeout_s=timeout_s,
            )
            if not isinstance(response_payload, dict):
                raise ValueError("tool output must be a dictionary")
        except AmbiguousToolError as exc:
            if normalized_operation == OperationType.WRITE:
                logger.warning(
                    "ambiguous write marked pending_recovery: execution_id=%s tool=%s error=%s",
                    execution_id,
                    tool_name,
                    exc,
                )
                # Replays of verified_not_found stay in that state on repeat
                # ambiguity (verified_not_found -> dispatched is illegal); the
                # next recovery pass re-verifies against the target.
                ambiguous_status = (
                    ExecutionStatus.VERIFIED_NOT_FOUND if is_verified_replay else ExecutionStatus.DISPATCHED
                )
                record = self.logging_service.record_tool_result(
                    run_id=run_id,
                    execution_id=execution_id,
                    result_status=ExternalResultStatus.TIMEOUT.value,
                    response_payload=None,
                    error_message=str(exc),
                    tool_name=tool_name,
                    operation_type=normalized_operation.value,
                    request_payload=request_payload,
                    plan_item_id=plan_item_id,
                    execution_status=ambiguous_status.value,
                    clear_pending=False,
                )
                return TransactionResult(
                    execution_id=execution_id,
                    disposition=TransactionDisposition.PENDING_RECOVERY,
                    record=record,
                    exception=exc,
                )

            record = self.logging_service.record_tool_result(
                run_id=run_id,
                execution_id=execution_id,
                result_status=ExternalResultStatus.ERROR.value,
                response_payload=None,
                error_message=str(exc),
                tool_name=tool_name,
                operation_type=normalized_operation.value,
                request_payload=request_payload,
                plan_item_id=plan_item_id,
                execution_status=ExecutionStatus.FAILED.value,
            )
            return TransactionResult(
                execution_id=execution_id,
                disposition=TransactionDisposition.FAILED,
                record=record,
                exception=exc,
            )
        except Exception as exc:
            logger.error(
                "tool execution failed: execution_id=%s tool=%s error=%s",
                execution_id,
                tool_name,
                exc,
            )
            record = self.logging_service.record_tool_result(
                run_id=run_id,
                execution_id=execution_id,
                result_status=ExternalResultStatus.ERROR.value,
                response_payload=None,
                error_message=str(exc),
                tool_name=tool_name,
                operation_type=normalized_operation.value,
                request_payload=request_payload,
                plan_item_id=plan_item_id,
                execution_status=ExecutionStatus.FAILED.value,
            )
            return TransactionResult(
                execution_id=execution_id,
                disposition=TransactionDisposition.FAILED,
                record=record,
                exception=exc,
            )

        logger.info(
            "tool execution acked: execution_id=%s tool=%s operation=%s replayed=%s",
            execution_id,
            tool_name,
            normalized_operation.value,
            is_verified_replay,
        )
        record = self.logging_service.record_tool_result(
            run_id=run_id,
            execution_id=execution_id,
            result_status=ExternalResultStatus.OK.value,
            response_payload=response_payload,
            tool_name=tool_name,
            operation_type=normalized_operation.value,
            request_payload=request_payload,
            plan_item_id=plan_item_id,
            execution_status=ExecutionStatus.ACKED.value,
        )
        return TransactionResult(
            execution_id=execution_id,
            disposition=TransactionDisposition.SUCCEEDED,
            record=record,
        )

    def execute_callable(
        self,
        *,
        run_id: str,
        plan_item_id: str,
        tool_name: str,
        request_payload: dict[str, Any],
        operation_type: OperationType | str,
        tool_fn: Callable[[dict[str, Any]], dict[str, Any]],
        invocation_key: str | None = None,
        timeout_s: float | None = None,
        idempotency_supported: bool = False,
    ) -> TransactionResult:
        """Convenience wrapper for simple callable-based tools."""

        return self.execute_tool(
            run_id=run_id,
            plan_item_id=plan_item_id,
            tool_name=tool_name,
            request_payload=request_payload,
            operation_type=operation_type,
            adapter=FunctionToolAdapter(tool_fn),
            invocation_key=invocation_key,
            timeout_s=timeout_s,
            idempotency_supported=idempotency_supported,
        )

    def _handle_existing(self, record: ExternalResultRecord) -> TransactionResult:
        if record.execution_status in self._REUSABLE_SUCCESS_STATUSES:
            return TransactionResult(
                execution_id=record.execution_id,
                disposition=TransactionDisposition.REUSED,
                record=record,
            )
        if record.execution_status == ExecutionStatus.FAILED:
            return TransactionResult(
                execution_id=record.execution_id,
                disposition=TransactionDisposition.FAILED,
                record=record,
            )
        if record.execution_status in self._PENDING_RECOVERY_STATUSES:
            return TransactionResult(
                execution_id=record.execution_id,
                disposition=TransactionDisposition.PENDING_RECOVERY,
                record=record,
            )
        raise ValueError(f"unsupported execution status: {record.execution_status.value}")


def _normalize_operation_type(raw: OperationType | str) -> OperationType:
    if isinstance(raw, OperationType):
        return raw
    return OperationType(raw.strip().upper())
