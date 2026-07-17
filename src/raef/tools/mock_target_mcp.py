"""MCP interface for the JSON KV-backed mock target service.

This follows the mainstream Python MCP pattern: use FastMCP when available and
register typed tools that wrap the core service methods.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from .mock_target import (
    IdempotencyMode,
    JsonKVStore,
    MockTargetService,
    default_store_path,
)

try:
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover - optional dependency
    FastMCP = None


def _default_store_path() -> Path:
    return default_store_path()


def create_mcp_server(
    store_path: Path | None = None,
    *,
    idempotency_mode: IdempotencyMode = IdempotencyMode.NON_IDEMPOTENT,
) -> Any:
    """Create and configure a FastMCP server instance.

    Returns a FastMCP object when the optional `mcp` package is installed.
    """
    if FastMCP is None:
        raise RuntimeError(
            "The `mcp` package is not installed. Install it to run MCP stdio server."
        )

    resolved_path = store_path or _default_store_path()
    service = MockTargetService(
        JsonKVStore(resolved_path),
        idempotency_mode=idempotency_mode,
    )
    server = FastMCP("raef-mock-target")

    @server.tool(name="mock_target_apply_action")
    def mock_target_apply_action(
        action_name: str,
        payload: dict[str, Any],
        execution_id: str | None = None,
    ) -> dict[str, Any]:
        """Apply one action through the legacy compatibility API."""
        return service.apply_action(
            action_name=action_name,
            payload=payload,
            execution_id=execution_id,
        )

    @server.tool(name="mock_target_apply_idempotent_action")
    def mock_target_apply_idempotent_action(
        action_name: str,
        payload: dict[str, Any],
        execution_id: str,
    ) -> dict[str, Any]:
        """Apply one idempotent action keyed by execution id."""
        return service.apply_idempotent_action(
            action_name=action_name,
            payload=payload,
            execution_id=execution_id,
        )

    @server.tool(name="mock_target_apply_rifl_action")
    def mock_target_apply_rifl_action(
        action_name: str,
        payload: dict[str, Any],
        execution_id: str,
    ) -> dict[str, Any]:
        """Apply one RIFL-style action keyed by execution id."""
        return service.apply_rifl_action(
            action_name=action_name,
            payload=payload,
            execution_id=execution_id,
        )

    @server.tool(name="mock_target_probe_rifl_execution")
    def mock_target_probe_rifl_execution(execution_id: str) -> dict[str, Any]:
        """Probe a RIFL execution by execution id."""
        return service.probe_rifl_execution(execution_id)

    @server.tool(name="mock_target_apply_queryable_distinguishable")
    def mock_target_apply_queryable_distinguishable(
        action_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply a queryable-but-not-exactly-once action by domain key."""
        return service.apply_queryable_distinguishable(action_name=action_name, payload=payload)

    @server.tool(name="mock_target_increment_counter")
    def mock_target_increment_counter(counter_name: str, delta: int | float = 1) -> dict[str, Any]:
        """Apply a non-distinguishable aggregate counter increment."""
        return service.increment_counter(counter_name=counter_name, delta=delta)

    @server.tool(name="mock_target_query_state")
    def mock_target_query_state(
        query_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Query current domain state from the JSON KV store."""
        return service.query_state(query_name=query_name, payload=payload)

    @server.tool(name="mock_target_get_action")
    def mock_target_get_action(execution_id: str) -> dict[str, Any]:
        """Fetch an action log record by execution id."""
        record = service.get_action_by_execution_id(execution_id)
        return {"found": record is not None, "record": record}

    @server.tool(name="mock_target_health")
    def mock_target_health() -> dict[str, Any]:
        """Return health and basic counters."""
        return service.health()

    return server


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mock target MCP server (stdio)")
    parser.add_argument(
        "--store-path",
        default=str(_default_store_path()),
        help="Path to JSON KV store.",
    )
    parser.add_argument(
        "--idempotency-mode",
        choices=[mode.value for mode in IdempotencyMode],
        default=IdempotencyMode.NON_IDEMPOTENT.value,
        help="Legacy write handling mode for repeated execution_id values.",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if FastMCP is None:
        print(
            "Missing dependency: install `mcp` to run this MCP server",
            file=sys.stderr,
        )
        return 2

    server = create_mcp_server(
        Path(args.store_path),
        idempotency_mode=IdempotencyMode(args.idempotency_mode),
    )
    server.run(transport="stdio")
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
