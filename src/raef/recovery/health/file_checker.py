from __future__ import annotations

from pathlib import Path
import time
from typing import Callable

from .base import HealthChecker


class FileHealthChecker(HealthChecker):
    """File-based health checker without network dependency.

    Supports two modes:
    1) probe_fn mode: call a local function that returns truthy when healthy.
    2) heartbeat file mode: treat process as healthy if file mtime is recent.
    """

    def __init__(
        self,
        *,
        probe_fn: Callable[[], bool] | None = None,
        heartbeat_file: str | Path | None = None,
        stale_after_seconds: float = 5.0,
        clock_fn: Callable[[], float] | None = None,
    ) -> None:
        if probe_fn is None and heartbeat_file is None:
            raise ValueError("Either probe_fn or heartbeat_file must be provided")
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be > 0")

        self.probe_fn = probe_fn
        self.heartbeat_file = Path(heartbeat_file) if heartbeat_file is not None else None
        self.stale_after_seconds = float(stale_after_seconds)
        self._clock_fn = clock_fn or time.time

    def is_alive(self) -> bool:
        if self.probe_fn is not None:
            try:
                return bool(self.probe_fn())
            except Exception:
                return False

        assert self.heartbeat_file is not None
        if not self.heartbeat_file.exists():
            return False
        try:
            mtime_epoch = self.heartbeat_file.stat().st_mtime
        except OSError:
            return False

        now_epoch = self._clock_fn()
        return (now_epoch - mtime_epoch) <= self.stale_after_seconds
