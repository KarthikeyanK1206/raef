from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime

from ...agent_state import ExternalResultRecord, ExternalResultStatus
from ...logging_service import LoggingService
from ...models import ExecutionStatus, OperationType
from ..common import (
    DEFAULT_WAIT_SECONDS,
    RecoveryAction,
    RecoveryDecision,
    WriteVerifierProtocol,
)
from .strategy import RuntimeRecoveryStrategy

logger = logging.getLogger(__name__)


class RecoveryCoordinator:
    """Runtime recovery coordinator backed by LoggingService.

    When a ``verifier`` is provided, ambiguous WRITE executions that exceed
    their waiting window are resolved by probing the target system instead of
    being handed off: a proven commit becomes ``MARK_COMMITTED`` (and its
    durable result is reconciled into the context), while a proven absence
    becomes ``REPLAY`` with the record moved to ``verified_not_found`` so the
    transaction manager can safely re-execute it. Inconclusive verification
    falls back to the human/policy handoff path.
    """

    def __init__(
        self,
        logging_service: LoggingService,
        *,
        strategy: RuntimeRecoveryStrategy | None = None,
        default_wait_seconds: float = DEFAULT_WAIT_SECONDS,
        on_handoff: Callable[[RecoveryDecision], None] | None = None,
        verifier: WriteVerifierProtocol | None = None,
    ) -> None:
        self.logging_service = logging_service
        self.strategy = strategy or RuntimeRecoveryStrategy(default_wait_seconds=default_wait_seconds)
        self.on_handoff = on_handoff
        self.verifier = verifier
        self._handoff_emitted: set[str] = set()

    def recover_run(
        self,
        run_id: str,
        *,
        now: datetime | None = None,
    ) -> list[RecoveryDecision]:
        now = now or datetime.now(UTC)

        context = self.logging_service.context_service.load_context(run_id)
        if context is None:
            return []

        records = {
            record.execution_id: record
            for record in self.logging_service.list_external_results(run_id)
        }
        pending_ids = set(self.logging_service.list_pending_executions(run_id))
        candidate_ids = sorted(set(records.keys()) | pending_ids)

        decisions: list[RecoveryDecision] = []
        for execution_id in candidate_ids:
            record = records.get(execution_id)
            if record is None:
                decision = RecoveryDecision(
                    run_id=run_id,
                    execution_id=execution_id,
                    action=RecoveryAction.HANDOFF_RETRY_OR_ABANDON,
                    reason="pending execution is missing durable external result",
                    execution_status=None,
                )
                logger.warning(
                    "recovery handoff: run_id=%s execution_id=%s reason=%s",
                    run_id,
                    execution_id,
                    decision.reason,
                )
                self._emit_handoff_if_needed(decision)
                decisions.append(decision)
                continue

            decision = self.strategy.decide(
                run_id=run_id,
                execution_id=execution_id,
                record=record,
                now=now,
            )

            if (
                decision.action == RecoveryAction.HANDOFF_RETRY_OR_ABANDON
                and self.verifier is not None
                and record.operation_type == OperationType.WRITE.value
                and record.execution_status == ExecutionStatus.DISPATCHED
            ):
                decision, record = self._verify_ambiguous_write(
                    run_id=run_id,
                    execution_id=execution_id,
                    record=record,
                    fallback=decision,
                )

            logger.info(
                "recovery decision: run_id=%s execution_id=%s action=%s reason=%s",
                run_id,
                execution_id,
                decision.action.value,
                decision.reason,
            )

            if decision.action == RecoveryAction.MARK_COMMITTED:
                self._reconcile_tool_message(run_id=run_id, record=record)
            elif decision.action == RecoveryAction.HANDOFF_RETRY_OR_ABANDON:
                self._emit_handoff_if_needed(decision)

            decisions.append(decision)

        return decisions

    def _verify_ambiguous_write(
        self,
        *,
        run_id: str,
        execution_id: str,
        record: ExternalResultRecord,
        fallback: RecoveryDecision,
    ) -> tuple[RecoveryDecision, ExternalResultRecord]:
        assert self.verifier is not None
        try:
            verification = self.verifier.verify_write(
                execution_id,
                tool_name=record.tool_name,
                args=record.request_payload,
            )
        except Exception as exc:
            logger.warning(
                "verification failed: run_id=%s execution_id=%s error=%s",
                run_id,
                execution_id,
                exc,
            )
            return fallback, record

        if verification.committed is True:
            updated = self.logging_service.record_tool_result(
                run_id=run_id,
                execution_id=execution_id,
                result_status=ExternalResultStatus.OK.value,
                response_payload=_response_payload_from_verification(verification),
                execution_status=ExecutionStatus.VERIFIED_COMMITTED.value,
            )
            return (
                RecoveryDecision(
                    run_id=run_id,
                    execution_id=execution_id,
                    action=RecoveryAction.MARK_COMMITTED,
                    reason=f"target verification confirmed commit: {verification.reason}",
                    execution_status=ExecutionStatus.VERIFIED_COMMITTED,
                ),
                updated,
            )

        if verification.committed is False:
            updated = self.logging_service.record_tool_result(
                run_id=run_id,
                execution_id=execution_id,
                result_status=ExternalResultStatus.UNKNOWN.value,
                response_payload=None,
                execution_status=ExecutionStatus.VERIFIED_NOT_FOUND.value,
                clear_pending=False,
            )
            return (
                RecoveryDecision(
                    run_id=run_id,
                    execution_id=execution_id,
                    action=RecoveryAction.REPLAY,
                    reason=f"target verification found no commit: {verification.reason}",
                    execution_status=ExecutionStatus.VERIFIED_NOT_FOUND,
                ),
                updated,
            )

        return fallback, record

    def _emit_handoff_if_needed(self, decision: RecoveryDecision) -> None:
        if self.on_handoff is None:
            return
        key = (
            f"{decision.run_id}:{decision.execution_id}:{decision.action.value}:"
            f"{decision.execution_status.value if decision.execution_status else 'none'}:"
            f"{decision.wait_until.isoformat() if decision.wait_until else 'none'}"
        )
        if key in self._handoff_emitted:
            return
        self._handoff_emitted.add(key)
        self.on_handoff(decision)

    def _reconcile_tool_message(self, *, run_id: str, record: ExternalResultRecord) -> None:
        if record.response_payload is None:
            return
        context = self.logging_service.context_service.load_context(run_id)
        if context is None:
            return
        for message in context.messages:
            if message.role == "tool" and message.tool_call_id == record.execution_id:
                return

        content = json.dumps(record.response_payload, sort_keys=True, separators=(",", ":"))
        self.logging_service.record_context_message(
            run_id=run_id,
            role="tool",
            content=content,
            name=record.tool_name,
            tool_call_id=record.execution_id,
            meta={
                "source": "recovery",
                "reconciled": True,
                "execution_status": record.execution_status.value,
            },
        )


def _response_payload_from_verification(verification: object) -> dict[str, object]:
    """Shape a durable response payload from whatever the verifier observed."""

    record = getattr(verification, "record", None)
    receipt = getattr(verification, "receipt", None)
    payload: dict[str, object] = {"verified": True}
    if isinstance(receipt, dict):
        payload["receipt"] = receipt
    if isinstance(record, dict):
        result = record.get("result")
        if isinstance(result, dict):
            payload["result"] = result
        payload.setdefault("target_record", record)
    return payload
