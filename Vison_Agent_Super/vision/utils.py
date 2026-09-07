"""Small shared helpers for the Super vision pipeline."""

from __future__ import annotations

import re
from typing import Any


def log(message: str) -> None:
    print(message, flush=True)


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, (str, int))]


def mapping_selected_ids(mapping: dict[str, Any]) -> list[str]:
    keys = (
        "annotation_ids",
        "parameter_text_ids",
        "description_text_ids",
        "leader_ids",
        "arrowhead_ids",
        "target_ids",
    )
    selected: list[str] = ["F0"]
    for key in keys:
        selected.extend(string_list(mapping.get(key)))
    return list(dict.fromkeys(selected))


def safe_fai_name(value: Any) -> str:
    if value is None:
        return "unknown"
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return cleaned or "unknown"
