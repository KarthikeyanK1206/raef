from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ...agent_state import ExternalResultRecord
from ...models import ExecutionStatus
from ..common import (
    DEFAULT_WAIT_SECONDS,
    RecoveryAction,
    RecoveryDecision,
    is_committed_status,
    resolve_wait_deadline,
)


@dataclass(slots=True)
class RuntimeRecoveryStrategy:
    """Runtime decision engine for one recovered execution record."""

    default_wait_seconds: float = DEFAULT_WAIT_SECONDS

    def decide(
        self,
        *,
        run_id: str,
        execution_id: str,
        record: ExternalResultRecord,
        now: datetime,
    ) -> RecoveryDecision:
        status = record.execution_status

        if status == ExecutionStatus.INTENT_LOGGED:
            return RecoveryDecision(
                run_id=run_id,
                execution_id=execution_id,
                action=RecoveryAction.REPLAY,
                reason="intent recorded but dispatch not observed",
                execution_status=status,
            )

        if status == ExecutionStatus.DISPATCHED:
            wait_seconds, deadline = resolve_wait_deadline(
                started_at=record.updated_at,
                request_payload=record.request_payload,
                fallback_seconds=self.default_wait_seconds,
            )
            if now < deadline:
                return RecoveryDecision(
                    run_id=run_id,
                    execution_id=execution_id,
                    action=RecoveryAction.RESUME_WAITING,
                    reason="in-step execution still inside waiting window",
                    execution_status=status,
                    wait_timeout_seconds=wait_seconds,
                    wait_until=deadline,
                )
            return RecoveryDecision(
                run_id=run_id,
                execution_id=execution_id,
                action=RecoveryAction.HANDOFF_RETRY_OR_ABANDON,
                reason="in-step execution exceeded waiting window",
                execution_status=status,
                wait_timeout_seconds=wait_seconds,
                wait_until=deadline,
            )

        if status == ExecutionStatus.VERIFIED_NOT_FOUND:
            return RecoveryDecision(
                run_id=run_id,
                execution_id=execution_id,
                action=RecoveryAction.REPLAY,
                reason="target verification indicates missing commit",
                execution_status=status,
            )

        if is_committed_status(status):
            return RecoveryDecision(
                run_id=run_id,
                execution_id=execution_id,
                action=RecoveryAction.MARK_COMMITTED,
                reason="execution already committed",
                execution_status=status,
            )

        if status == ExecutionStatus.FAILED:
            return RecoveryDecision(
                run_id=run_id,
                execution_id=execution_id,
                action=RecoveryAction.HANDOFF_RETRY_OR_ABANDON,
                reason="execution previously failed and requires policy handoff",
                execution_status=status,
            )

        return RecoveryDecision(
            run_id=run_id,
            execution_id=execution_id,
            action=RecoveryAction.HANDOFF_RETRY_OR_ABANDON,
            reason=f"unhandled status={status.value}",
            execution_status=status,
        )
