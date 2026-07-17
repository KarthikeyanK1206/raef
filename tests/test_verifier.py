"""Tests for the verifier module."""

from __future__ import annotations

from pathlib import Path

from raef.models import ExecutionStatus
from raef.tools.mock_target import JsonKVStore, MockTargetService
from raef.verifier import MockTargetVerifier, verify_mock_target_write


def test_mock_target_verifier_returns_committed_for_known_execution_id(tmp_path: Path) -> None:
    target = MockTargetService(JsonKVStore(tmp_path / "target.json"))
    target.apply_idempotent_action(
        action_name="set_value",
        payload={"key": "orders/order-1", "value": {"status": "ordered"}},
        execution_id="exec-verified-1",
    )

    decision = MockTargetVerifier(target).verify_write(
        "exec-verified-1",
        tool_name="apply-action",
        args={"payload": {"key": "orders/order-1", "value": {"status": "ordered"}}},
    )

    assert decision.execution_status == ExecutionStatus.VERIFIED_COMMITTED
    assert decision.committed is True
    assert decision.receipt is not None


def test_mock_target_verifier_returns_not_found_for_missing_execution_id(tmp_path: Path) -> None:
    target = MockTargetService(JsonKVStore(tmp_path / "target.json"))

    decision = verify_mock_target_write(
        target,
        "exec-missing-1",
        tool_name="apply-action",
        args={"payload": {"key": "orders/order-2", "value": {"status": "ordered"}}},
    )

    assert decision.execution_status == ExecutionStatus.VERIFIED_NOT_FOUND
    assert decision.committed is False
    assert decision.decision == "safe_to_replay"


def test_mock_target_verifier_can_fall_back_to_domain_state_match(tmp_path: Path) -> None:
    target = MockTargetService(JsonKVStore(tmp_path / "target.json"))
    target.apply_action(
        action_name="set_value",
        payload={"key": "lights/dorm_a/status", "value": {"value": "off"}},
    )

    decision = MockTargetVerifier(target).verify_write(
        "exec-unknown",
        tool_name="apply-action",
        args={"payload": {"key": "lights/dorm_a/status", "value": {"value": "off"}}},
    )

    assert decision.execution_status == ExecutionStatus.VERIFIED_COMMITTED
    assert decision.committed is True
    assert "expected payload" in decision.reason
