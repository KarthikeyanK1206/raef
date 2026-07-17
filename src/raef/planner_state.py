"""Planner state list management for RAEF.

This module implements explicit plan state tracking so code can:
- create and persist plan items,
- update item status,
- track active/cursor position,
- attach LLM outputs and tool execution references.

Is mainly for workflow's future planning

This is event driven so the runner would get a item to run, execute it and
then update the idem with the interface.

TODO: this planning currently always read and writes to the disk,
later will use a hybrid in memory stragety

"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Protocol

from .runtime_store import SQLiteRuntimeStore
from .utils import (
    normalize_str_list,
    optional_int,
    optional_str,
    parse_datetime,
    require_dict_or_default,
    require_non_empty_str,
    utc_now,
)


# PlannerItem denotes the state of a unit/step in a plan

class PlannerItemStatus(str, Enum):
    """Lifecycle status for one planner item.
    this is for the task/plan level
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"


@dataclass(slots=True)
class PlannerItem:
    """One explicit plan step maintained by middleware state."""

    plan_item_id: str
    title: str
    description: str = ""
    status: PlannerItemStatus = PlannerItemStatus.PENDING
    sequence_index: int = 0
    depends_on: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    llm_output: str | None = None
    llm_output_history: list[str] = field(default_factory=list)
    tool_call_refs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    updated_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_item_id": self.plan_item_id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "sequence_index": self.sequence_index,
            "depends_on": list(self.depends_on),
            "acceptance_criteria": list(self.acceptance_criteria),
            "llm_output": self.llm_output,
            "llm_output_history": list(self.llm_output_history),
            "tool_call_refs": list(self.tool_call_refs),
            "notes": list(self.notes),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlannerItem":
        sequence_index = int(payload.get("sequence_index", 0))
        if sequence_index < 0:
            raise ValueError("sequence_index must be non-negative")
        return cls(
            plan_item_id=require_non_empty_str(payload, "plan_item_id"),
            title=require_non_empty_str(payload, "title"),
            description=str(payload.get("description", "")),
            status=PlannerItemStatus(str(payload.get("status", PlannerItemStatus.PENDING.value))),
            sequence_index=sequence_index,
            depends_on=normalize_str_list(payload.get("depends_on", []), field_name="depends_on"),
            acceptance_criteria=normalize_str_list(
                payload.get("acceptance_criteria", []),
                field_name="acceptance_criteria",
            ),
            llm_output=optional_str(payload.get("llm_output")),
            llm_output_history=normalize_str_list(
                payload.get("llm_output_history", []),
                field_name="llm_output_history",
            ),
            tool_call_refs=normalize_str_list(
                payload.get("tool_call_refs", []),
                field_name="tool_call_refs",
            ),
            notes=normalize_str_list(payload.get("notes", []), field_name="notes"),
            updated_at=parse_datetime(payload.get("updated_at")),
        )


# PlannerState denotes the entire plan state

@dataclass(slots=True)
class PlannerState:
    """Persisted plan state for one run."""

    run_id: str
    version: int
    plan_source_text: str
    items: list[PlannerItem]
    plan_schema: dict[str, Any] = field(default_factory=dict)
    active_item_id: str | None = None
    cursor_index: int | None = None
    updated_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "version": self.version,
            "plan_source_text": self.plan_source_text,
            "items": [item.to_dict() for item in self.items],
            "plan_schema": self.plan_schema,
            "active_item_id": self.active_item_id,
            "cursor_index": self.cursor_index,
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlannerState":
        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list):
            raise ValueError("items must be a list")

        items = [PlannerItem.from_dict(item) for item in raw_items]
        state = cls(
            run_id=require_non_empty_str(payload, "run_id"),
            version=int(payload.get("version", 1)),
            plan_source_text=str(payload.get("plan_source_text", "")),
            items=items,
            plan_schema=_normalize_plan_schema(require_dict_or_default(payload.get("plan_schema")), items=items),
            active_item_id=optional_str(payload.get("active_item_id")),
            cursor_index=optional_int(payload.get("cursor_index"), "cursor_index"),
            updated_at=parse_datetime(payload.get("updated_at")),
        )
        state._validate_integrity()
        return state

    def _validate_integrity(self) -> None:
        if self.version < 1:
            raise ValueError("version must be >= 1")
        ids = [item.plan_item_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("plan item ids must be unique")
        if self.active_item_id is not None and self.active_item_id not in set(ids):
            raise ValueError("active_item_id does not exist in items")
        if self.cursor_index is not None:
            if self.cursor_index < 0 or self.cursor_index >= len(self.items):
                raise ValueError("cursor_index out of range")


# interface for plan persistence
class PlannerStateRepository(Protocol):
    """Persistence interface for planner state."""

    def load(self, run_id: str) -> PlannerState | None:
        """Load planner state for run_id if present."""

    def save(self, state: PlannerState) -> PlannerState:
        """Persist planner state and return persisted object."""

    def flush(self) -> None:
        """Persist any dirty in-memory state."""


# persist plan via sqlite
class SQLitePlannerStateRepository:
    """SQLite-backed planner state repository."""

    def __init__(
        self,
        store: SQLiteRuntimeStore | None = None,
        db_path: Path | None = None,
        *,
        busy_timeout_ms: int = 5000,
    ) -> None:
        if store is None:
            if db_path is None:
                repo_root = Path(__file__).resolve().parents[2]
                db_path = repo_root / "data" / "raef_runtime" / "runtime.sqlite"
            store = SQLiteRuntimeStore(db_path, busy_timeout_ms=busy_timeout_ms)
        self.store = store

    def load(self, run_id: str) -> PlannerState | None:
        state = self.store.load_planner_state(run_id)
        if state is None:
            return None
        return PlannerState.from_dict(state.to_dict())

    def save(self, state: PlannerState) -> PlannerState:
        state.updated_at = utc_now()
        state._validate_integrity()
        self.store.save_planner_state(state)
        return PlannerState.from_dict(state.to_dict())

    def flush(self) -> None:
        self.store.flush()


# plan management, controls the plan to transit accoding to the state machine
class PlannerStateService:
    """Application-facing planner API for explicit code-driven plan updates."""

    _ALLOWED_TRANSITIONS: dict[PlannerItemStatus, set[PlannerItemStatus]] = {
        PlannerItemStatus.PENDING: {
            PlannerItemStatus.IN_PROGRESS,
            PlannerItemStatus.BLOCKED,
            PlannerItemStatus.DONE,
            PlannerItemStatus.FAILED,
        },
        PlannerItemStatus.IN_PROGRESS: {
            PlannerItemStatus.BLOCKED,
            PlannerItemStatus.DONE,
            PlannerItemStatus.FAILED,
            PlannerItemStatus.PENDING,
        },
        PlannerItemStatus.BLOCKED: {
            PlannerItemStatus.PENDING,
            PlannerItemStatus.IN_PROGRESS,
            PlannerItemStatus.FAILED,
        },
        PlannerItemStatus.DONE: set(),
        PlannerItemStatus.FAILED: {PlannerItemStatus.PENDING, PlannerItemStatus.IN_PROGRESS},
    }

    def __init__(self, repository: PlannerStateRepository | None = None) -> None:
        self.repository = repository or SQLitePlannerStateRepository()

    def create_plan(
        self,
        run_id: str,
        plan_source_text: str,
        items: list[dict[str, Any]],
        *,
        plan_schema: dict[str, Any] | None = None,
    ) -> PlannerState:
        if not items:
            raise ValueError("items cannot be empty")

        normalized_schema = _normalize_plan_schema(plan_schema or {"steps": items})

        planner_items: list[PlannerItem] = []
        seen_ids: set[str] = set()
        for index, raw in enumerate(items):
            item_id = str(raw.get("plan_item_id") or f"step_{index}").strip()
            if not item_id:
                raise ValueError(f"item at index {index} has an empty plan_item_id")
            if item_id in seen_ids:
                raise ValueError(f"duplicate plan_item_id: {item_id}")
            seen_ids.add(item_id)

            title = str(raw.get("title", "")).strip()
            if not title:
                raise ValueError(f"item {item_id} must have non-empty title")

            sequence_index = int(raw.get("sequence_index", index))
            if sequence_index < 0:
                raise ValueError(f"item {item_id} must have non-negative sequence_index")

            planner_items.append(
                PlannerItem(
                    plan_item_id=item_id,
                    title=title,
                    description=str(raw.get("description", "")),
                    status=_parse_status(raw.get("status", PlannerItemStatus.PENDING.value)),
                    sequence_index=sequence_index,
                    depends_on=normalize_str_list(raw.get("depends_on", []), field_name="depends_on"),
                    acceptance_criteria=normalize_str_list(
                        raw.get("acceptance_criteria", []),
                        field_name="acceptance_criteria",
                    ),
                    llm_output=optional_str(raw.get("llm_output")),
                )
            )

        planner_items.sort(key=lambda item: item.sequence_index)
        state = PlannerState(
            run_id=run_id,
            version=1,
            plan_source_text=plan_source_text,
            items=planner_items,
            plan_schema=_normalize_plan_schema(normalized_schema, items=planner_items),
        )
        first = self.next_runnable_item_from_state(state)
        if first is not None:
            state.active_item_id = first.plan_item_id
            state.cursor_index = first.sequence_index
        return self.repository.save(state)

    # Expected format of a llm plan as text:
    # 1. Search available flights from Los Angeles to New York
    # 2. Compare prices and flight durations
    # 3. Select the best flight option
    # 4. Enter passenger details
    # 5. Confirm and book the ticket
    #
    # Example prompt:
    # You are a planning assistant for an AI agent.
    # Given a user task, break it down into a clear, step-by-step execution plan.
    # Output only the plan, one step per line, preferably in numbered format.
    #
    # TODO: later we could generate the plan via a structured plans format with
    # goal, steps, constraints, and verification sections.

    def create_plan_from_text(self, run_id: str, plan_source_text: str) -> PlannerState:
        """Create a plan by parsing structured or numbered model output text."""
        plan_schema = _parse_plan_source_to_schema(plan_source_text)
        parsed_items = plan_schema["steps"]
        return self.create_plan(
            run_id=run_id,
            plan_source_text=plan_source_text,
            items=parsed_items,
            plan_schema=plan_schema,
        )

    # load plan from storage
    def load_plan(self, run_id: str) -> PlannerState | None:
        return self.repository.load(run_id)

    # state mutation: set PlannerState to run_id and plan_item_id
    def set_active_item(self, run_id: str, plan_item_id: str) -> PlannerState:
        state = self._load_required(run_id)
        item = self._find_item(state, plan_item_id)
        state.active_item_id = item.plan_item_id
        state.cursor_index = item.sequence_index
        item.updated_at = utc_now()
        return self._bump_and_save(state)

    # Call example:
    # item = planner.next_runnable_item(run_id)
    # planner.update_item_status(run_id, item.plan_item_id, "in_progress")
    # result = run_tool(...)
    # planner.update_item_status(run_id, item.plan_item_id, "done")

    # updates the status of a plan step
    def update_item_status(
        self,
        run_id: str,
        plan_item_id: str,
        status: str,
        note: str | None = None,
    ) -> PlannerState:
        state = self._load_required(run_id)
        item = self._find_item(state, plan_item_id)
        next_status = _parse_status(status)

        if next_status != item.status:
            allowed = self._ALLOWED_TRANSITIONS[item.status]  # cannot proceed to invalid state
            if next_status not in allowed:
                raise ValueError(
                    f"illegal planner status transition: {item.status.value} -> {next_status.value}"
                )
            item.status = next_status
            item.updated_at = utc_now()

        if note is not None and note.strip():
            item.notes.append(note.strip())
            item.updated_at = utc_now()
        # DONE at this PlannerItem
        if item.status == PlannerItemStatus.DONE and state.active_item_id == item.plan_item_id:
            next_item = self.next_runnable_item_from_state(state)
            if next_item is not None:
                state.active_item_id = next_item.plan_item_id
                state.cursor_index = next_item.sequence_index

        return self._bump_and_save(state)

    # Stores reasoning/output for this step
    def attach_llm_output(self, run_id: str, plan_item_id: str, llm_output: str) -> PlannerState:
        state = self._load_required(run_id)
        item = self._find_item(state, plan_item_id)
        text = llm_output.strip()
        if not text:
            raise ValueError("llm_output cannot be empty")
        item.llm_output = text
        item.llm_output_history.append(text)
        item.updated_at = utc_now()
        return self._bump_and_save(state)

    # References a planner step to execution ID
    def attach_tool_call_ref(self, run_id: str, plan_item_id: str, execution_id: str) -> PlannerState:
        state = self._load_required(run_id)
        item = self._find_item(state, plan_item_id)
        ref = execution_id.strip()
        if not ref:
            raise ValueError("execution_id cannot be empty")
        if ref not in item.tool_call_refs:
            item.tool_call_refs.append(ref)
            item.updated_at = utc_now()
        return self._bump_and_save(state)

    def next_runnable_item(self, run_id: str) -> PlannerItem | None:
        state = self._load_required(run_id)
        return self.next_runnable_item_from_state(state)

    # Scheduler:
    # Finds the next step that is pending, has all dependencies satisfied,
    # and respects the explicit sequence ordering.

    def next_runnable_item_from_state(self, state: PlannerState) -> PlannerItem | None:
        done_ids = {
            item.plan_item_id for item in state.items if item.status == PlannerItemStatus.DONE
        }
        ordered_items = sorted(state.items, key=lambda item: item.sequence_index)

        for item in ordered_items:
            if item.status != PlannerItemStatus.PENDING:
                continue
            if all(dep in done_ids for dep in item.depends_on):
                return item
        return None

    # loads from the repository
    def _load_required(self, run_id: str) -> PlannerState:
        state = self.repository.load(run_id)
        if state is None:
            raise ValueError(f"planner state not found for run_id={run_id}")
        return state

    @staticmethod
    # find item by ID
    def _find_item(state: PlannerState, plan_item_id: str) -> PlannerItem:
        for item in state.items:
            if item.plan_item_id == plan_item_id:
                return item
        raise ValueError(f"plan item not found: {plan_item_id}")

    # persistence state with version
    def _bump_and_save(self, state: PlannerState) -> PlannerState:
        state.version += 1
        state.updated_at = utc_now()
        return self.repository.save(state)

    def flush(self) -> None:
        self.repository.flush()


# parse the given llm text and parse the plan, is subject to how we define the plan
def _parse_plan_source_to_schema(plan_source_text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(plan_source_text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        return _normalize_plan_schema(parsed)

    lines = [line.rstrip() for line in plan_source_text.splitlines() if line.strip()]
    if not lines:
        return _normalize_plan_schema({"steps": [{"title": "Execute task"}]})

    goal = ""
    steps: list[dict[str, Any]] = []
    constraints: list[str] = []
    verification: list[str] = []
    current_section = "steps"

    for line in lines:
        text = line.strip()
        lower = text.lower()
        if lower.startswith("goal:"):
            goal = text.split(":", 1)[1].strip()
            current_section = "goal"
            continue
        if lower.startswith("steps:"):
            current_section = "steps"
            continue
        if lower.startswith("constraints:"):
            current_section = "constraints"
            continue
        if lower.startswith("verification:"):
            current_section = "verification"
            continue

        if re.match(r"^\d+[.)]\s+", text) or text.startswith(("- ", "* ")):
            item_text = re.sub(r"^\d+[.)]\s+", "", text)
            item_text = item_text[2:].strip() if item_text.startswith(("- ", "* ")) else item_text.strip()
            if not item_text:
                continue
            if current_section == "constraints":
                constraints.append(item_text)
            elif current_section == "verification":
                verification.append(item_text)
            else:
                steps.append({"title": item_text})
            continue

        if current_section == "goal" and not goal:
            goal = text
        elif current_section == "constraints":
            constraints.append(text)
        elif current_section == "verification":
            verification.append(text)
        else:
            steps.append({"title": text})

    return _normalize_plan_schema(
        {
            "goal": goal,
            "steps": steps,
            "constraints": constraints,
            "verification": verification,
            "source_format": "text",
        }
    )


def _normalize_plan_schema(
    raw_schema: dict[str, Any],
    *,
    items: list[PlannerItem] | None = None,
) -> dict[str, Any]:
    steps_raw = raw_schema.get("steps")
    normalized_steps: list[dict[str, Any]] = []
    if steps_raw is None and items is not None:
        steps_raw = [item.to_dict() for item in items]
    if not isinstance(steps_raw, list):
        raise ValueError("plan schema steps must be a list")

    for index, raw_step in enumerate(steps_raw):
        if isinstance(raw_step, str):
            raw_step = {"title": raw_step}
        if not isinstance(raw_step, dict):
            raise ValueError("each plan schema step must be a string or object")
        title = str(raw_step.get("title", "")).strip()
        if not title:
            raise ValueError(f"plan step at index {index} must have non-empty title")
        normalized_steps.append(
            {
                "plan_item_id": str(raw_step.get("plan_item_id") or f"step_{index}").strip(),
                "title": title,
                "description": str(raw_step.get("description", "")),
                "depends_on": normalize_str_list(raw_step.get("depends_on", []), field_name="depends_on"),
                "acceptance_criteria": normalize_str_list(
                    raw_step.get("acceptance_criteria", []),
                    field_name="acceptance_criteria",
                ),
                "sequence_index": int(raw_step.get("sequence_index", index)),
            }
        )

    return {
        "goal": str(raw_schema.get("goal", "")).strip(),
        "constraints": normalize_str_list(raw_schema.get("constraints", []), field_name="constraints"),
        "verification": normalize_str_list(raw_schema.get("verification", []), field_name="verification"),
        "source_format": str(raw_schema.get("source_format", "normalized")).strip() or "normalized",
        "steps": normalized_steps,
    }


def _parse_status(raw: Any) -> PlannerItemStatus:
    if isinstance(raw, PlannerItemStatus):
        return raw
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        return PlannerItemStatus(normalized)
    raise ValueError("invalid planner item status")
