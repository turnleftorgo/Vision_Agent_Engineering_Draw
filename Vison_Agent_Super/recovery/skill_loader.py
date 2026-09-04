"""Load and validate the prompt-only FAI recovery skill."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .models import RecoverySkill


FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


def load_recovery_skill(path: Path) -> RecoverySkill:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Recovery skill not found: {resolved}")
    raw = resolved.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(raw)
    if not match:
        raise ValueError("Recovery SKILL.md must contain YAML frontmatter")

    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip("\"'")
    name = metadata.get("name", "")
    description = metadata.get("description", "")
    body = match.group(2).strip()
    if name != "fai-crop-recovery":
        raise ValueError(f"Unexpected recovery skill name: {name!r}")
    if not description:
        raise ValueError("Recovery skill description is empty")
    if not body:
        raise ValueError("Recovery skill body is empty")

    return RecoverySkill(
        path=resolved,
        name=name,
        description=description,
        body=body,
        sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )
