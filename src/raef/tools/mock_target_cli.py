"""CLI interface for the JSON KV-backed mock target service.

This module is intended for tool-runtime integration where a middleware invokes
an external program and exchanges JSON through arguments/stdin/stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .mock_target import (
    IdempotencyMode,
    JsonKVStore,
    MockTargetService,
    default_store_path,
    parse_json_object,
    require_dict,
    require_str,
)

EXIT_OK = 0
EXIT_INVALID_REQUEST = 2
EXIT_NOT_FOUND = 3
EXIT_TRANSIENT_FAILURE = 4
EXIT_INTERNAL_ERROR = 5


def _default_store_path() -> Path:
    return default_store_path()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mock target CLI interface")
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

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_cmd = subparsers.add_parser("init-store", help="Initialize store file")
    init_cmd.set_defaults(needs_request=False)

    health_cmd = subparsers.add_parser("health", help="Service health")
    health_cmd.set_defaults(needs_request=False)

    apply_cmd = subparsers.add_parser("apply-action", help="Apply one action through the legacy API")
    apply_cmd.set_defaults(needs_request=True)
    _add_request_options(apply_cmd)

    apply_idempotent = subparsers.add_parser("apply-idempotent", help="Apply one idempotent action")
    apply_idempotent.set_defaults(needs_request=True)
    _add_request_options(apply_idempotent)

    apply_rifl = subparsers.add_parser("apply-rifl", help="Apply one RIFL-style action")
    apply_rifl.set_defaults(needs_request=True)
    _add_request_options(apply_rifl)

    probe_rifl = subparsers.add_parser("probe-rifl", help="Probe RIFL execution by execution id")
    probe_rifl.set_defaults(needs_request=True)
    _add_request_options(probe_rifl)

    apply_queryable = subparsers.add_parser(
        "apply-queryable-distinguishable",
        help="Apply a queryable-but-not-exactly-once keyed action",
    )
    apply_queryable.set_defaults(needs_request=True)
    _add_request_options(apply_queryable)

    increment_counter = subparsers.add_parser("increment-counter", help="Increment a non-distinguishable counter")
    increment_counter.set_defaults(needs_request=True)
    _add_request_options(increment_counter)

    query_cmd = subparsers.add_parser("query-state", help="Query domain state")
    query_cmd.set_defaults(needs_request=True)
    _add_request_options(query_cmd)

    get_cmd = subparsers.add_parser("get-action", help="Get action by execution id")
    get_cmd.set_defaults(needs_request=True)
    _add_request_options(get_cmd)

    return parser


def _add_request_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--request-json", help="Inline JSON request object")
    parser.add_argument("--request-file", help="Path to a JSON request file")
    parser.add_argument(
        "--request-stdin",
        action="store_true",
        help="Read JSON request object from stdin",
    )


def _load_request(args: argparse.Namespace) -> dict[str, Any]:
    sources = [bool(args.request_json), bool(args.request_file), bool(args.request_stdin)]
    if sum(sources) != 1:
        raise ValueError(
            "Provide exactly one request source: --request-json, --request-file, or --request-stdin"
        )

    if args.request_json:
        raw = args.request_json
    elif args.request_file:
        raw = Path(args.request_file).read_text(encoding="utf-8")
    else:
        raw = sys.stdin.read()

    return parse_json_object(raw, field_name="request")


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


def run(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    service = MockTargetService(
        JsonKVStore(Path(args.store_path)),
        idempotency_mode=IdempotencyMode(args.idempotency_mode),
    )

    try:
        if args.command == "init-store":
            service.store.init()
            _print_json({"status": "ok", "store_path": str(service.store.store_path)})
            return EXIT_OK

        if args.command == "health":
            _print_json(service.health())
            return EXIT_OK

        request = _load_request(args)

        if args.command == "apply-action":
            response = service.apply_action(
                action_name=require_str(request, "action_name"),
                payload=require_dict(request, "payload"),
                execution_id=request.get("execution_id"),
            )
            _print_json(response)
            return EXIT_OK

        if args.command == "apply-idempotent":
            response = service.apply_idempotent_action(
                action_name=require_str(request, "action_name"),
                payload=require_dict(request, "payload"),
                execution_id=require_str(request, "execution_id"),
            )
            _print_json(response)
            return EXIT_OK

        if args.command == "apply-rifl":
            response = service.apply_rifl_action(
                action_name=require_str(request, "action_name"),
                payload=require_dict(request, "payload"),
                execution_id=require_str(request, "execution_id"),
            )
            _print_json(response)
            return EXIT_OK

        if args.command == "probe-rifl":
            response = service.probe_rifl_execution(require_str(request, "execution_id"))
            _print_json(response)
            return EXIT_OK if response.get("found") else EXIT_NOT_FOUND

        if args.command == "apply-queryable-distinguishable":
            response = service.apply_queryable_distinguishable(
                action_name=require_str(request, "action_name"),
                payload=require_dict(request, "payload"),
            )
            _print_json(response)
            return EXIT_OK

        if args.command == "increment-counter":
            counter_name = require_str(request, "counter_name")
            response = service.increment_counter(
                counter_name=counter_name,
                delta=request.get("delta", 1),
            )
            _print_json(response)
            return EXIT_OK

        if args.command == "query-state":
            response = service.query_state(
                query_name=require_str(request, "query_name"),
                payload=require_dict(request, "payload"),
            )
            _print_json(response)
            return EXIT_OK

        if args.command == "get-action":
            record = service.get_action_by_execution_id(require_str(request, "execution_id"))
            if record is None:
                _print_json({"found": False, "record": None})
                return EXIT_NOT_FOUND
            _print_json({"found": True, "record": record})
            return EXIT_OK

        print(f"unsupported command: {args.command}", file=sys.stderr)
        return EXIT_INVALID_REQUEST

    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_INVALID_REQUEST
    except TimeoutError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_TRANSIENT_FAILURE
    except Exception as exc:  # pragma: no cover
        print(f"internal error: {exc}", file=sys.stderr)
        return EXIT_INTERNAL_ERROR


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
