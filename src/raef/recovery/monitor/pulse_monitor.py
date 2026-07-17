from __future__ import annotations

import time
from collections.abc import Callable


class PulseMonitor:
    def __init__(
        self,
        health_checker,
        recovery_handler,
        *,
        poll_interval_seconds: float = 1.0,
        recovery_cooldown_seconds: float = 1.0,
        time_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        self.health_checker = health_checker
        self.recovery_handler = recovery_handler
        self.poll_interval_seconds = max(0.01, float(poll_interval_seconds))
        self.recovery_cooldown_seconds = max(0.0, float(recovery_cooldown_seconds))
        self._time_fn = time_fn or time.monotonic
        self._sleep_fn = sleep_fn or time.sleep
        self._last_recovery_at: float | None = None

    def run(self, *, max_loops: int | None = None):
        loops = 0
        while True:
            if max_loops is not None and loops >= max_loops:
                return
            loops += 1

            alive = self.health_checker.is_alive()
            if not alive:
                now = self._time_fn()
                if self._last_recovery_at is None or (
                    now - self._last_recovery_at >= self.recovery_cooldown_seconds
                ):
                    self.recovery_handler.handle()
                    self._last_recovery_at = now
            self._sleep_fn(self.poll_interval_seconds)
