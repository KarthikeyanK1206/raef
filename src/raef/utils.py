"""Shared utility helpers for RAEF persistence and validation."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_datetime(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str) and raw.strip():
        return datetime.fromisoformat(raw)
    return utc_now()


def require_non_empty_str(obj_or_value: Any, key_or_field: str) -> str:
    if isinstance(obj_or_value, dict):
        value = obj_or_value.get(key_or_field)
    else:
        value = obj_or_value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key_or_field} must be a non-empty string")
    return value.strip()


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError("optional string field has invalid type")


def require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def require_dict_or_default(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("value must be an object")
    return value


def optional_int(value: Any, field_name: str = "value") -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    raise ValueError(f"{field_name} must be an integer when provided")


def normalize_str_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")

    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} must contain only strings")
        text = item.strip()
        if text:
            result.append(text)
    return result


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", value)


def read_json_object(file_path: Path) -> dict[str, Any]:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{file_path} must contain a JSON object")
    return payload


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def json_size_bytes(payload: Any) -> int:
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def durable_write_text(file_path: Path, text: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    tmp_path.replace(file_path)
    _fsync_directory(file_path.parent)



def durable_write_json(file_path: Path, payload: dict[str, Any]) -> None:
    durable_write_text(file_path, stable_json_dumps(payload))



def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
