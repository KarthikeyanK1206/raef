"""Soft observability helpers for local logging module workflows.

The SQLite runtime remains the canonical store. This module writes best-effort
JSON mirrors that make local inspection easier without affecting correctness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


class RunStateReader(Protocol):
    """Small reader contract used by the soft JSON mirror."""

    def rebuild_state(self, run_id: str, *, snapshot_type: str = "full") -> dict[str, Any]:
        """Return the current run bundle."""


class SoftJSONObservability:
    """Best-effort JSON mirror for local run inspection."""

    def __init__(self, root: Path, *, reader: RunStateReader) -> None:
        self.root = root
        self.reader = reader
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def record_run(self, run_id: str) -> None:
        payload = self.reader.rebuild_state(run_id)
        self._write_json(self.runs_dir / f"{run_id}.json", payload)

    def flush(self) -> None:
        """Soft JSON writes are synchronous, so flush is currently a no-op."""

    def close(self) -> None:
        """No background workers are active, so close is currently a no-op."""

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp_path.replace(path)
