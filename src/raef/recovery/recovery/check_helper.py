from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CheckRecoveryResult:
    """Result envelope for handoff controllers that support check-before-retry."""

    should_retry: bool
    found_previous_result: bool
    reason: str
    request_id: str | None = None
    check_request_payload: dict[str, Any] | None = None
    check_response_payload: dict[str, Any] | None = None
    recovered_response_payload: dict[str, Any] | None = None


def recover_with_check_api(
    *,
    request: Any,
    check_request_payload: Mapping[str, Any] | None,
    call_check_api: Callable[[dict[str, Any]], dict[str, Any]],
    response_extractor: Callable[[dict[str, Any]], tuple[bool, dict[str, Any] | None]] | None = None,
    request_id_field: str = "execution_id",
) -> CheckRecoveryResult:
    """Try recovering by querying remote request status before retrying.

    This helper is for upload/handoff controllers. It first attempts to resolve
    a prior result through a remote "check request" API. If no committed result
    can be confirmed, caller should fall back to its own retry policy.
    """

    if not request_id_field:
        raise ValueError("request_id_field must be a non-empty string")

    request_id = _extract_request_id(request)
    payload = dict(check_request_payload or {})
    if request_id is not None:
        payload.setdefault(request_id_field, request_id)

    extractor = response_extractor or _default_response_extractor
    try:
        check_response = call_check_api(payload)
    except Exception as exc:
        return CheckRecoveryResult(
            should_retry=True,
            found_previous_result=False,
            reason=f"check api call failed: {exc}",
            request_id=request_id,
            check_request_payload=payload,
        )

    if not isinstance(check_response, dict):
        return CheckRecoveryResult(
            should_retry=True,
            found_previous_result=False,
            reason="check api returned non-dict response",
            request_id=request_id,
            check_request_payload=payload,
        )

    found, recovered = extractor(check_response)
    if found and recovered is not None:
        return CheckRecoveryResult(
            should_retry=False,
            found_previous_result=True,
            reason="reused previous committed result from check api",
            request_id=request_id,
            check_request_payload=payload,
            check_response_payload=check_response,
            recovered_response_payload=recovered,
        )
    return CheckRecoveryResult(
        should_retry=True,
        found_previous_result=False,
        reason="check api did not confirm a reusable committed result",
        request_id=request_id,
        check_request_payload=payload,
        check_response_payload=check_response,
        recovered_response_payload=recovered,
    )


def _extract_request_id(request: Any) -> str | None:
    candidates = ("execution_id", "request_id", "id")

    if isinstance(request, Mapping):
        for key in candidates:
            value = request.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    for key in candidates:
        value = getattr(request, key, None)
        if isinstance(value, str) and value:
            return value
    return None


def _default_response_extractor(response: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
    found = response.get("found")
    if isinstance(found, bool):
        if not found:
            return False, None
        return True, _extract_payload_from_record(response.get("record")) or _extract_dict_payload(response.get("result"))

    state = response.get("state")
    if isinstance(state, str) and state.lower() == "committed":
        payload = (
            _extract_payload_from_record(response.get("record"))
            or _extract_dict_payload(response.get("result"))
            or _extract_dict_payload(response)
        )
        return True, payload

    return False, None


def _extract_payload_from_record(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    return _extract_dict_payload(raw.get("result")) or _extract_dict_payload(raw.get("receipt")) or dict(raw)


def _extract_dict_payload(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    return None
