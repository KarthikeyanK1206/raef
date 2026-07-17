"""Public decorators for lightweight RAEF runtime integration."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, Literal, ParamSpec, TypeVar, cast

from ..logging_service import LoggingService


P = ParamSpec("P")
R = TypeVar("R")


def with_logging_service(
    *,
    data_root: str | Path = "./data/raef_runtime",
    checkpoint_every_n_events: int = 0,
    wal_backend: Literal["sqlite"] = "sqlite",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Inject a LoggingService when the wrapped function is called.

    Wrapped function requirements:
    - Accept a keyword argument named logging_service.
    - Optionally pass an explicit logging_service when calling; if provided,
      no auto-created instance is injected.
    """

    data_path = Path(data_root)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            params = dict(kwargs)
            if params.get("logging_service") is None:
                params["logging_service"] = LoggingService.with_data_root(
                    data_path,
                    checkpoint_every_n_events=checkpoint_every_n_events,
                    wal_backend=wal_backend,
                )
            return func(*args, **cast(Any, params))

        return wrapper

    return decorator


def ensure_run_started(
    *,
    default_plan_source_text: str,
    default_plan_items: list[dict[str, Any]],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Ensure run context exists before entering wrapped function.

    Wrapped function requirements:
    - Accept keyword arguments run_id and logging_service.
    - Optionally accept initial_messages; defaults to one user message.
    """

    if not default_plan_items:
        raise ValueError("default_plan_items cannot be empty")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            params = dict(kwargs)
            run_id = params.get("run_id")
            logging_service = params.get("logging_service")

            if not isinstance(run_id, str) or not run_id.strip():
                raise ValueError("run_id must be provided as a non-empty string")
            if not isinstance(logging_service, LoggingService):
                raise ValueError("logging_service must be provided")

            planner_state = logging_service.planner_service.load_plan(run_id)
            if planner_state is None:
                initial_messages = params.get("initial_messages")
                if initial_messages is None:
                    initial_messages = [
                        {
                            "role": "user",
                            "content": "start",
                        }
                    ]
                if not isinstance(initial_messages, list):
                    raise ValueError("initial_messages must be a list of message objects")
                logging_service.start_run(
                    run_id=run_id,
                    initial_messages=initial_messages,
                    plan_source_text=default_plan_source_text,
                    plan_items=default_plan_items,
                )

            return func(*args, **cast(Any, params))

        return wrapper

    return decorator
