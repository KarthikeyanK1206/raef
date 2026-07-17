"""Extended mock request store built on the recovery request model.

This keeps the recovery data shape (`RequestRecord`, `RequestStatus`) while
adding a few helper methods that make the local mock-agent flows easier to run
and inspect.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RequestStatus(Enum):
    NOT_SENT = "not_sent"
    IN_STEP = "in_step"
    REPLIED = "replied"


@dataclass(frozen=True)
class RequestRecord:
    request_id: str
    status: RequestStatus
    payload: dict[str, Any]


class MockStore:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    def load_requests(self):
        return self.data.get("requests", [])


class MockStoreMod(MockStore):
    """Small request-store helper for local scripted agent workflows."""

    def __init__(
        self,
        requests: list[RequestRecord] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.data["requests"] = list(requests or [])
        self.data["metadata"] = dict(metadata or {})

    def write_request(self, req: RequestRecord) -> None:
        requests = self.load_requests()
        for index, existing in enumerate(requests):
            if existing.request_id == req.request_id:
                requests[index] = req
                self.data["requests"] = requests
                return
        requests.append(req)
        self.data["requests"] = requests

    def replace_requests(self, requests: list[RequestRecord]) -> None:
        self.data["requests"] = list(requests)

    def get_request(self, request_id: str) -> RequestRecord | None:
        for req in self.load_requests():
            if req.request_id == request_id:
                return req
        return None

    def get_request_by_step_index(self, step_index: int) -> RequestRecord | None:
        for req in self.load_requests():
            payload = req.payload if isinstance(req.payload, dict) else {}
            if payload.get("step_index") == step_index:
                return req
        return None

    def mark_request_status(self, request_id: str, status: RequestStatus) -> RequestRecord:
        req = self.get_request(request_id)
        if req is None:
            raise KeyError(f"unknown request_id: {request_id}")
        updated = RequestRecord(request_id=req.request_id, status=status, payload=req.payload)
        self.write_request(updated)
        return updated

    def mark_step_status(self, step_index: int, status: RequestStatus) -> RequestRecord:
        req = self.get_request_by_step_index(step_index)
        if req is None:
            raise KeyError(f"unknown step_index: {step_index}")
        return self.mark_request_status(req.request_id, status)

    def set_metadata(self, **items: Any) -> None:
        metadata = dict(self.data.get("metadata", {}))
        metadata.update(items)
        self.data["metadata"] = metadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": dict(self.data.get("metadata", {})),
            "requests": [
                {
                    "request_id": req.request_id,
                    "status": req.status.value,
                    "payload": req.payload,
                }
                for req in self.load_requests()
            ],
        }
