"""Recovery shared models and helpers for RAEF."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Protocol

from ..models import ExecutionStatus


DEFAULT_WAIT_SECONDS = 5.0


class WriteVerification(Protocol):
    """Result contract a write verifier must satisfy.

    ``committed`` is tri-state: True (proof of commit), False (proof of
    absence), or None (inconclusive; the caller must not replay).
    """

    @property
    def committed(self) -> bool | None: ...

    @property
    def reason(self) -> str: ...

    @property
    def record(self) -> dict[str, Any] | None: ...

    @property
    def receipt(self) -> dict[str, Any] | None: ...


class WriteVerifierProtocol(Protocol):
    """Target-side commit probe used to resolve ambiguous writes."""

    def verify_write(
        self,
        execution_id: str,
        *,
        tool_name: str = "",
        args: dict[str, Any] | None = None,
    ) -> WriteVerification: ...


class RecoveryAction(str, Enum):
    """Recovery actions returned by runtime recovery."""

    RESUME_WAITING = "resume_waiting"
    REPLAY = "replay"
    HANDOFF_RETRY_OR_ABANDON = "handoff_retry_or_abandon"
    MARK_COMMITTED = "mark_committed"


@dataclass(slots=True)
class RecoveryDecision:
    """One recovery decision for a specific execution id."""

    run_id: str
    execution_id: str
    action: RecoveryAction
    reason: str
    execution_status: ExecutionStatus | None = None
    wait_timeout_seconds: float | None = None
    wait_until: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def is_committed_status(status: ExecutionStatus) -> bool:
    return status in {ExecutionStatus.ACKED, ExecutionStatus.VERIFIED_COMMITTED}


def parse_call_timeout_seconds(payload: dict[str, Any] | None) -> float | None:
    """Parse call timeout from tool request payload when present."""

    if payload is None or not isinstance(payload, dict):
        return None

    direct_keys_seconds = (
        "call_timeout_seconds",
        "timeout_seconds",
        "timeout_s",
    )
    for key in direct_keys_seconds:
        parsed = _parse_positive_float(payload.get(key))
        if parsed is not None:
            return parsed

    direct_keys_ms = ("call_timeout_ms", "timeout_ms")
    for key in direct_keys_ms:
        parsed_ms = _parse_positive_float(payload.get(key))
        if parsed_ms is not None:
            return parsed_ms / 1000.0

    parsed_generic = _parse_positive_float(payload.get("timeout"))
    if parsed_generic is not None:
        return parsed_generic

    for nested_key in ("meta", "metadata", "config", "options"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            nested_timeout = parse_call_timeout_seconds(nested)
            if nested_timeout is not None:
                return nested_timeout
    return None


def resolve_wait_deadline(
    *,
    started_at: datetime,
    request_payload: dict[str, Any] | None,
    fallback_seconds: float = DEFAULT_WAIT_SECONDS,
) -> tuple[float, datetime]:
    timeout_seconds = parse_call_timeout_seconds(request_payload)
    resolved = timeout_seconds if timeout_seconds is not None else fallback_seconds
    if resolved <= 0:
        resolved = fallback_seconds
    return resolved, started_at + timedelta(seconds=resolved)


def _parse_positive_float(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if value > 0 else None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        return value if value > 0 else None
    return None
