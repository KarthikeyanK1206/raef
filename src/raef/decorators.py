"""Compatibility shim for moved adapter decorators."""

from __future__ import annotations

from .adapters.decorators import ensure_run_started, with_logging_service

__all__ = [
    "with_logging_service",
    "ensure_run_started",
]
