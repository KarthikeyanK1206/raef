"""Commit verifier module for RAEF.

Define the verification path that determines whether a tool action was already
performed at the target system and whether replay is safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import ExecutionStatus
from .tools.mock_target import MockTargetService


@dataclass(frozen=True)
class VerificationDecision:
    """Verification result for one execution id."""

    execution_id: str
    execution_status: ExecutionStatus
    decision: str
    reason: str
    committed: bool | None
    record: dict[str, Any] | None = None
    receipt: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "execution_status": self.execution_status.value,
            "decision": self.decision,
            "reason": self.reason,
            "committed": self.committed,
            "record": self.record,
            "receipt": self.receipt,
        }


class MockTargetVerifier:
    """Verifier that checks the local mock target for committed writes."""

    def __init__(self, target: MockTargetService) -> None:
        self.target = target

    def verify_write(
        self,
        execution_id: str,
        *,
        tool_name: str = "",
        args: dict[str, Any] | None = None,
    ) -> VerificationDecision:
        if not execution_id:
            raise ValueError("execution_id is required")

        try:
            record = self.target.get_action_by_execution_id(execution_id)
        except Exception as exc:  # pragma: no cover - defensive integration path
            return VerificationDecision(
                execution_id=execution_id,
                execution_status=ExecutionStatus.DISPATCHED,
                decision="retry_later",
                reason=f"target lookup failed: {exc}",
                committed=None,
            )

        if isinstance(record, dict):
            receipt = _extract_receipt(record)
            return VerificationDecision(
                execution_id=execution_id,
                execution_status=ExecutionStatus.VERIFIED_COMMITTED,
                decision="skip_replay",
                reason="execution id already exists in mock target action log",
                committed=True,
                record=record,
                receipt=receipt,
            )

        payload = args.get("payload", {}) if isinstance(args, dict) else {}
        if isinstance(payload, dict):
            key = payload.get("key")
            expected_value = payload.get("value")
            if isinstance(key, str) and key:
                observed = self.target.query_state("get_value", {"key": key}).get("result")
                if observed == expected_value and expected_value is not None:
                    return VerificationDecision(
                        execution_id=execution_id,
                        execution_status=ExecutionStatus.VERIFIED_COMMITTED,
                        decision="skip_replay",
                        reason="domain state already matches the expected payload",
                        committed=True,
                        record={"payload": payload, "observed_value": observed, "tool_name": tool_name},
                    )

        return VerificationDecision(
            execution_id=execution_id,
            execution_status=ExecutionStatus.VERIFIED_NOT_FOUND,
            decision="safe_to_replay",
            reason="mock target has no record for the execution id",
            committed=False,
        )


def verify_mock_target_write(
    target: MockTargetService,
    execution_id: str,
    *,
    tool_name: str = "",
    args: dict[str, Any] | None = None,
) -> VerificationDecision:
    """Convenience helper for verifying one mock-target write."""

    return MockTargetVerifier(target).verify_write(execution_id, tool_name=tool_name, args=args)


def _extract_receipt(record: dict[str, Any]) -> dict[str, Any] | None:
    receipt = record.get("receipt")
    if isinstance(receipt, dict):
        return receipt
    result = record.get("result")
    if isinstance(result, dict):
        nested = result.get("receipt")
        if isinstance(nested, dict):
            return nested
    return None
