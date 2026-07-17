"""Core JSON KV-backed mock target service.

This module provides the shared in-process service logic used by both the CLI
and MCP wrappers.

It intentionally exposes several remote-service capability tiers so the rest
of the project can test different recovery / exactly-once assumptions:
- an idempotent API keyed by execution_id,
- a RIFL-style API with explicit probe support by execution_id,
- queryable-but-not-exactly-once APIs:
  - distinguishable object writes keyed by domain key,
  - non-distinguishable counter increments.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def default_store_path() -> Path:
    """Return the default local JSON store path inside the workspace data dir."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "mock_target_store.json"


def parse_json_object(raw: str, *, field_name: str = "request") -> dict[str, Any]:
    """Parse and validate a JSON object payload."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {field_name} JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


def require_str(obj: dict[str, Any], key: str) -> str:
    """Require a non-empty string field in an object."""
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def require_dict(obj: dict[str, Any], key: str) -> dict[str, Any]:
    """Require an object field in an object."""
    value = obj.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


class IdempotencyMode(str, Enum):
    """Legacy execution-id handling strategy for apply_action()."""

    NON_IDEMPOTENT = "non_idempotent"
    IDEMPOTENT = "idempotent"


class RemoteApiLevel(str, Enum):
    """Capability tier exposed by the mock remote service."""

    LEGACY = "legacy"
    IDEMPOTENT = "idempotent"
    RIFL = "rifl"
    QUERYABLE_DISTINGUISHABLE = "queryable_distinguishable"
    QUERYABLE_COUNTER = "queryable_counter"


class JsonKVStore:
    """Small JSON-backed storage with several logical maps.

    - action_log: action/execution keyed log of committed work
    - domain_state: free-form legacy key-value map
    - idempotent_requests: request-id keyed deduplicated writes
    - rifl_requests: request-id keyed probeable executions
    - distinguishable_state: queryable object state by domain key
    - distinguishable_history: append-only object history by domain key
    - counters: non-distinguishable aggregate counters
    """

    def __init__(self, store_path: Path) -> None:
        self.store_path = store_path

    def init(self) -> None:
        if self.store_path.exists():
            return
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "version": 3,
                "created_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
            },
            "stats": {
                "total_commits": 0,
                "idempotent_commits": 0,
                "rifl_commits": 0,
                "distinguishable_commits": 0,
                "counter_commits": 0,
            },
            "action_log": {},
            "domain_state": {},
            "idempotent_requests": {},
            "rifl_requests": {},
            "distinguishable_state": {},
            "distinguishable_history": {},
            "counters": {},
        }
        self._atomic_write(payload)

    def load(self) -> dict[str, Any]:
        self.init()
        with self.store_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self._ensure_shape(data)
        return data

    def save(self, data: dict[str, Any]) -> None:
        self._ensure_shape(data)
        data["meta"]["updated_at"] = _utc_now_iso()
        self._atomic_write(data)

    def _atomic_write(self, data: dict[str, Any]) -> None:
        tmp_path = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        tmp_path.replace(self.store_path)

    @staticmethod
    def _ensure_shape(data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise ValueError("store payload must be an object")
        data.setdefault("meta", {})
        data.setdefault("stats", {})
        data.setdefault("action_log", {})
        data.setdefault("domain_state", {})
        data.setdefault("idempotent_requests", {})
        data.setdefault("rifl_requests", {})
        data.setdefault("distinguishable_state", {})
        data.setdefault("distinguishable_history", {})
        data.setdefault("counters", {})
        for key in [
            "stats",
            "action_log",
            "domain_state",
            "idempotent_requests",
            "rifl_requests",
            "distinguishable_state",
            "distinguishable_history",
            "counters",
        ]:
            if not isinstance(data[key], dict):
                raise ValueError(f"{key} must be an object")
        stats = data["stats"]
        for key in [
            "total_commits",
            "idempotent_commits",
            "rifl_commits",
            "distinguishable_commits",
            "counter_commits",
        ]:
            stats.setdefault(key, 0)
            if not isinstance(stats[key], int):
                raise ValueError(f"stats.{key} must be an integer")


class MockTargetService:
    """Core in-process service for local CLI/MCP adapters."""

    def __init__(
        self,
        store: JsonKVStore,
        *,
        idempotency_mode: IdempotencyMode = IdempotencyMode.NON_IDEMPOTENT,
    ) -> None:
        self.store = store
        self.idempotency_mode = idempotency_mode

    def apply_action(
        self,
        action_name: str,
        payload: dict[str, Any],
        execution_id: str | None = None,
    ) -> dict[str, Any]:
        """Legacy action API kept for compatibility with existing wrappers."""
        if not action_name:
            raise ValueError("action_name is required")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")

        if self.idempotency_mode == IdempotencyMode.IDEMPOTENT:
            return self.apply_idempotent_action(
                action_name=action_name,
                payload=payload,
                execution_id=execution_id or str(uuid4()),
            )

        data = self.store.load()
        exec_id = execution_id or str(uuid4())
        receipt = _build_receipt(execution_id=exec_id)
        self._apply_to_domain_state(data["domain_state"], action_name, payload, receipt)

        action_log: dict[str, Any] = data["action_log"]
        existing = action_log.get(exec_id)
        attempt_count = 1
        receipts = [receipt]
        if isinstance(existing, dict):
            attempt_count = int(existing.get("attempt_count", 1)) + 1
            prior_receipts = existing.get("receipts")
            if isinstance(prior_receipts, list):
                receipts = [*prior_receipts, receipt]
            else:
                prior_receipt = existing.get("receipt")
                if isinstance(prior_receipt, dict):
                    receipts = [prior_receipt, receipt]

        record = {
            "api_level": RemoteApiLevel.LEGACY.value,
            "execution_id": exec_id,
            "action_name": action_name,
            "payload": payload,
            "receipt": receipt,
            "receipts": receipts,
            "attempt_count": attempt_count,
            "status": "committed",
            "updated_at": _utc_now_iso(),
        }
        action_log[exec_id] = record
        _increment_stat(data, "total_commits")
        self.store.save(data)
        return {
            "receipt": receipt,
            "idempotent_hit": False,
            "api_level": RemoteApiLevel.LEGACY.value,
        }

    def apply_idempotent_action(
        self,
        action_name: str,
        payload: dict[str, Any],
        execution_id: str,
    ) -> dict[str, Any]:
        """Deduplicate by execution id and return the original receipt on replay."""
        if not action_name:
            raise ValueError("action_name is required")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        if not execution_id:
            raise ValueError("execution_id is required")

        data = self.store.load()
        requests: dict[str, Any] = data["idempotent_requests"]
        existing = requests.get(execution_id)
        if isinstance(existing, dict):
            if existing.get("action_name") != action_name or existing.get("payload") != payload:
                raise ValueError("execution_id already used with a different action or payload")
            return {
                "receipt": existing.get("receipt"),
                "idempotent_hit": True,
                "api_level": RemoteApiLevel.IDEMPOTENT.value,
            }

        receipt = _build_receipt(execution_id=execution_id)
        self._apply_to_domain_state(data["domain_state"], action_name, payload, receipt)
        record = {
            "api_level": RemoteApiLevel.IDEMPOTENT.value,
            "execution_id": execution_id,
            "action_name": action_name,
            "payload": payload,
            "receipt": receipt,
            "status": "committed",
            "updated_at": _utc_now_iso(),
        }
        requests[execution_id] = record
        data["action_log"][execution_id] = record
        _increment_stat(data, "total_commits")
        _increment_stat(data, "idempotent_commits")
        self.store.save(data)
        return {
            "receipt": receipt,
            "idempotent_hit": False,
            "api_level": RemoteApiLevel.IDEMPOTENT.value,
        }

    def apply_rifl_action(
        self,
        action_name: str,
        payload: dict[str, Any],
        execution_id: str,
    ) -> dict[str, Any]:
        """RIFL-style request API with durable probeable execution records."""
        if not action_name:
            raise ValueError("action_name is required")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        if not execution_id:
            raise ValueError("execution_id is required")

        data = self.store.load()
        requests: dict[str, Any] = data["rifl_requests"]
        existing = requests.get(execution_id)
        if isinstance(existing, dict):
            if existing.get("action_name") != action_name or existing.get("payload") != payload:
                raise ValueError("execution_id already used with a different action or payload")
            return {
                "receipt": existing.get("receipt"),
                "rifl_hit": True,
                "state": existing.get("state", "committed"),
                "api_level": RemoteApiLevel.RIFL.value,
            }

        receipt = _build_receipt(execution_id=execution_id)
        self._apply_to_domain_state(data["domain_state"], action_name, payload, receipt)
        record = {
            "api_level": RemoteApiLevel.RIFL.value,
            "execution_id": execution_id,
            "action_name": action_name,
            "payload": payload,
            "receipt": receipt,
            "state": "committed",
            "result": {"receipt": receipt},
            "updated_at": _utc_now_iso(),
        }
        requests[execution_id] = record
        data["action_log"][execution_id] = record
        _increment_stat(data, "total_commits")
        _increment_stat(data, "rifl_commits")
        self.store.save(data)
        return {
            "receipt": receipt,
            "rifl_hit": False,
            "state": "committed",
            "api_level": RemoteApiLevel.RIFL.value,
        }

    def probe_rifl_execution(self, execution_id: str) -> dict[str, Any]:
        """Probe RIFL request state by execution id."""
        if not execution_id:
            raise ValueError("execution_id is required")
        data = self.store.load()
        record = data["rifl_requests"].get(execution_id)
        if record is None:
            return {
                "found": False,
                "state": "not_found",
                "record": None,
            }
        return {
            "found": True,
            "state": str(record.get("state", "committed")),
            "record": record,
        }

    def apply_queryable_distinguishable(
        self,
        action_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply a domain-keyed action that can be queried later by business key.

        This API does not deduplicate by execution id. Recovery can inspect the
        resulting object state by key, but duplicates are still possible.
        """
        if not action_name:
            raise ValueError("action_name is required")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        key = require_str(payload, "key")

        data = self.store.load()
        state: dict[str, Any] = data["distinguishable_state"]
        history_by_key: dict[str, Any] = data["distinguishable_history"]
        action_id = str(uuid4())
        receipt = _build_receipt(action_id=action_id)
        entity = self._apply_to_distinguishable_state(state, action_name, payload, receipt)

        history = history_by_key.setdefault(key, [])
        if not isinstance(history, list):
            raise ValueError("distinguishable_history entries must be lists")
        history.append(
            {
                "action_id": action_id,
                "action_name": action_name,
                "payload": payload,
                "receipt": receipt,
            }
        )

        record = {
            "api_level": RemoteApiLevel.QUERYABLE_DISTINGUISHABLE.value,
            "action_id": action_id,
            "action_name": action_name,
            "payload": payload,
            "query_key": key,
            "receipt": receipt,
            "status": "committed",
            "updated_at": _utc_now_iso(),
        }
        data["action_log"][action_id] = record
        _increment_stat(data, "total_commits")
        _increment_stat(data, "distinguishable_commits")
        self.store.save(data)
        return {
            "receipt": receipt,
            "query_key": key,
            "entity": entity,
            "api_level": RemoteApiLevel.QUERYABLE_DISTINGUISHABLE.value,
        }

    def increment_counter(self, counter_name: str, delta: int | float = 1) -> dict[str, Any]:
        """Apply a non-distinguishable aggregate update like a counter increment."""
        if not counter_name:
            raise ValueError("counter_name is required")
        if not isinstance(delta, (int, float)):
            raise ValueError("delta must be numeric")

        data = self.store.load()
        counters: dict[str, Any] = data["counters"]
        current = counters.get(counter_name, 0)
        if not isinstance(current, (int, float)):
            raise ValueError("existing counter is not numeric")
        new_value = current + delta
        counters[counter_name] = new_value

        action_id = str(uuid4())
        receipt = _build_receipt(action_id=action_id)
        data["action_log"][action_id] = {
            "api_level": RemoteApiLevel.QUERYABLE_COUNTER.value,
            "action_id": action_id,
            "counter_name": counter_name,
            "delta": delta,
            "receipt": receipt,
            "status": "committed",
            "updated_at": _utc_now_iso(),
        }
        _increment_stat(data, "total_commits")
        _increment_stat(data, "counter_commits")
        self.store.save(data)
        return {
            "receipt": receipt,
            "counter_name": counter_name,
            "value": new_value,
            "api_level": RemoteApiLevel.QUERYABLE_COUNTER.value,
        }

    def get_action_by_execution_id(self, execution_id: str) -> dict[str, Any] | None:
        if not execution_id:
            raise ValueError("execution_id is required")
        data = self.store.load()
        action_log: dict[str, Any] = data["action_log"]
        record = action_log.get(execution_id)
        if isinstance(record, dict):
            return record
        return data["idempotent_requests"].get(execution_id) or data["rifl_requests"].get(execution_id)

    def query_state(self, query_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not query_name:
            raise ValueError("query_name is required")
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")

        data = self.store.load()
        domain_state: dict[str, Any] = data["domain_state"]
        distinguishable_state: dict[str, Any] = data["distinguishable_state"]
        history_by_key: dict[str, Any] = data["distinguishable_history"]
        counters: dict[str, Any] = data["counters"]

        if query_name == "get_value":
            key = require_str(payload, "key")
            return {"result": domain_state.get(key)}

        if query_name == "list_keys":
            prefix = payload.get("prefix")
            keys = sorted(domain_state.keys())
            if isinstance(prefix, str):
                keys = [k for k in keys if k.startswith(prefix)]
            return {"result": keys}

        if query_name == "dump_state":
            return {"result": domain_state}

        if query_name == "get_action_count":
            return {"result": int(data["stats"].get("total_commits", 0))}

        if query_name == "get_distinguishable_value":
            key = require_str(payload, "key")
            return {"result": distinguishable_state.get(key)}

        if query_name == "list_distinguishable_keys":
            return {"result": sorted(distinguishable_state.keys())}

        if query_name == "get_distinguishable_history":
            key = require_str(payload, "key")
            history = history_by_key.get(key)
            if not isinstance(history, list):
                history = []
            return {"result": history}

        if query_name == "get_counter":
            counter_name = require_str(payload, "counter_name")
            return {"result": counters.get(counter_name, 0)}

        if query_name == "dump_counters":
            return {"result": counters}

        if query_name == "get_idempotent_record":
            execution_id = require_str(payload, "execution_id")
            return {"result": data["idempotent_requests"].get(execution_id)}

        if query_name == "get_rifl_record":
            execution_id = require_str(payload, "execution_id")
            return {"result": data["rifl_requests"].get(execution_id)}

        raise ValueError(f"unsupported query_name: {query_name}")

    def health(self) -> dict[str, Any]:
        data = self.store.load()
        return {
            "status": "ok",
            "store_path": str(self.store.store_path),
            "actions": int(data["stats"].get("total_commits", 0)),
            "keys": len(data["domain_state"]),
            "default_legacy_idempotency_mode": self.idempotency_mode.value,
            "idempotent_requests": len(data["idempotent_requests"]),
            "rifl_requests": len(data["rifl_requests"]),
            "distinguishable_keys": len(data["distinguishable_state"]),
            "counters": len(data["counters"]),
            "capabilities": [
                RemoteApiLevel.IDEMPOTENT.value,
                RemoteApiLevel.RIFL.value,
                RemoteApiLevel.QUERYABLE_DISTINGUISHABLE.value,
                RemoteApiLevel.QUERYABLE_COUNTER.value,
            ],
        }

    @staticmethod
    def _apply_to_domain_state(
        domain_state: dict[str, Any],
        action_name: str,
        payload: dict[str, Any],
        receipt: dict[str, Any],
    ) -> None:
        if action_name == "set_value":
            key = require_str(payload, "key")
            domain_state[key] = payload.get("value")
            return

        if action_name == "delete_value":
            key = require_str(payload, "key")
            domain_state.pop(key, None)
            return

        if action_name == "increment_value":
            key = require_str(payload, "key")
            delta = payload.get("delta", 1)
            if not isinstance(delta, (int, float)):
                raise ValueError("delta must be numeric")
            current = domain_state.get(key, 0)
            if not isinstance(current, (int, float)):
                raise ValueError("existing value is not numeric")
            domain_state[key] = current + delta
            return

        history = domain_state.setdefault("_applied_actions", [])
        if not isinstance(history, list):
            raise ValueError("_applied_actions must be a list")
        history.append(
            {
                "action_name": action_name,
                "payload": payload,
                "receipt": {
                    "execution_id": receipt.get("execution_id"),
                    "action_id": receipt["action_id"],
                },
            }
        )

    @staticmethod
    def _apply_to_distinguishable_state(
        state: dict[str, Any],
        action_name: str,
        payload: dict[str, Any],
        receipt: dict[str, Any],
    ) -> dict[str, Any]:
        key = require_str(payload, "key")
        if action_name == "delete_value":
            entity = {
                "key": key,
                "exists": False,
                "deleted": True,
                "last_action_name": action_name,
                "last_receipt": receipt,
                "updated_at": _utc_now_iso(),
            }
            state[key] = entity
            return entity

        entity = {
            "key": key,
            "exists": True,
            "deleted": False,
            "value": payload.get("value"),
            "payload": payload,
            "last_action_name": action_name,
            "last_receipt": receipt,
            "updated_at": _utc_now_iso(),
        }
        state[key] = entity
        return entity


def handle_mcp_tool_call(
    service: MockTargetService,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Utility helper for MCP adapters to route tool names to service calls."""
    if tool_name == "mock_target_apply_action":
        return service.apply_action(
            action_name=require_str(arguments, "action_name"),
            payload=require_dict(arguments, "payload"),
            execution_id=arguments.get("execution_id"),
        )
    if tool_name == "mock_target_apply_idempotent_action":
        return service.apply_idempotent_action(
            action_name=require_str(arguments, "action_name"),
            payload=require_dict(arguments, "payload"),
            execution_id=require_str(arguments, "execution_id"),
        )
    if tool_name == "mock_target_apply_rifl_action":
        return service.apply_rifl_action(
            action_name=require_str(arguments, "action_name"),
            payload=require_dict(arguments, "payload"),
            execution_id=require_str(arguments, "execution_id"),
        )
    if tool_name == "mock_target_probe_rifl_execution":
        return service.probe_rifl_execution(require_str(arguments, "execution_id"))
    if tool_name == "mock_target_apply_queryable_distinguishable":
        return service.apply_queryable_distinguishable(
            action_name=require_str(arguments, "action_name"),
            payload=require_dict(arguments, "payload"),
        )
    if tool_name == "mock_target_increment_counter":
        return service.increment_counter(
            counter_name=require_str(arguments, "counter_name"),
            delta=arguments.get("delta", 1),
        )
    if tool_name == "mock_target_query_state":
        return service.query_state(
            query_name=require_str(arguments, "query_name"),
            payload=require_dict(arguments, "payload"),
        )
    if tool_name == "mock_target_get_action":
        record = service.get_action_by_execution_id(require_str(arguments, "execution_id"))
        return {"found": record is not None, "record": record}
    if tool_name == "mock_target_health":
        return service.health()
    raise ValueError(f"unsupported MCP tool: {tool_name}")


def _build_receipt(*, execution_id: str | None = None, action_id: str | None = None) -> dict[str, Any]:
    return {
        "execution_id": execution_id,
        "action_id": action_id or str(uuid4()),
        "status": "committed",
        "committed_at": _utc_now_iso(),
    }


def _increment_stat(data: dict[str, Any], key: str) -> None:
    stats: dict[str, Any] = data["stats"]
    stats[key] = int(stats.get(key, 0)) + 1
