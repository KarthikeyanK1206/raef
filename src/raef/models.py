"""Shared data models for RAEF.

Define the core schemas for execution records, write-ahead log entries,
recovery status, and verifier results.
"""

from __future__ import annotations

from enum import Enum


class OperationType(str, Enum):
    """Operation type for tool calls."""

    READ = "READ"
    WRITE = "WRITE"


class ExecutionStatus(str, Enum):
    """Lifecycle status for one execution id across logging/txn/recovery.
    This is for the tool call level
    """

    INTENT_LOGGED = "intent_logged"
    DISPATCHED = "dispatched"
    ACKED = "acked"
    VERIFIED_COMMITTED = "verified_committed"
    VERIFIED_NOT_FOUND = "verified_not_found"
    FAILED = "failed"


def normalize_execution_status(raw: str) -> ExecutionStatus:
    """Parse status from string with strict normalization."""

    return ExecutionStatus(raw.strip().lower())
