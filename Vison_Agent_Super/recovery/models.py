"""Immutable data contracts shared by the crop recovery modules."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class CropBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def ordered(self) -> "CropBox":
        return CropBox(
            min(self.x1, self.x2),
            min(self.y1, self.y2),
            max(self.x1, self.x2),
            max(self.y1, self.y2),
        )

    @property
    def width(self) -> float:
        box = self.ordered()
        return max(0.0, box.x2 - box.x1)

    @property
    def height(self) -> float:
        box = self.ordered()
        return max(0.0, box.y2 - box.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        box = self.ordered()
        return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)

    def clamp(self, width: int, height: int) -> "CropBox":
        box = self.ordered()
        return CropBox(
            max(0.0, min(float(width), box.x1)),
            max(0.0, min(float(height), box.y1)),
            max(0.0, min(float(width), box.x2)),
            max(0.0, min(float(height), box.y2)),
        ).ordered()

    def expand(self, left: float, top: float, right: float, bottom: float) -> "CropBox":
        box = self.ordered()
        return CropBox(
            box.x1 - max(0.0, left),
            box.y1 - max(0.0, top),
            box.x2 + max(0.0, right),
            box.y2 + max(0.0, bottom),
        )

    def translate(self, dx: float, dy: float) -> "CropBox":
        box = self.ordered()
        return CropBox(box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy)

    def intersection_area(self, other: "CropBox") -> float:
        left = max(self.ordered().x1, other.ordered().x1)
        top = max(self.ordered().y1, other.ordered().y1)
        right = min(self.ordered().x2, other.ordered().x2)
        bottom = min(self.ordered().y2, other.ordered().y2)
        return max(0.0, right - left) * max(0.0, bottom - top)

    def contains_box(self, other: "CropBox", margin: float = 0.0) -> bool:
        box = self.ordered()
        target = other.ordered()
        return (
            target.x1 >= box.x1 + margin
            and target.y1 >= box.y1 + margin
            and target.x2 <= box.x2 - margin
            and target.y2 <= box.y2 - margin
        )

    def to_int_tuple(self) -> tuple[int, int, int, int]:
        box = self.ordered()
        return (
            int(math.floor(box.x1)),
            int(math.floor(box.y1)),
            int(math.ceil(box.x2)),
            int(math.ceil(box.y2)),
        )

    def to_list(self) -> list[int]:
        return list(self.to_int_tuple())

    @classmethod
    def from_values(cls, values: Any) -> "CropBox":
        if hasattr(values, "x1"):
            return cls(
                float(values.x1), float(values.y1), float(values.x2), float(values.y2)
            )
        sequence = list(values)
        if len(sequence) != 4:
            raise ValueError("CropBox requires four coordinates")
        return cls(*(float(value) for value in sequence))

    @classmethod
    def union(cls, boxes: list["CropBox"]) -> Optional["CropBox"]:
        usable = [box.ordered() for box in boxes if box.area > 0]
        if not usable:
            return None
        return cls(
            min(box.x1 for box in usable),
            min(box.y1 for box in usable),
            max(box.x2 for box in usable),
            max(box.y2 for box in usable),
        )


@dataclass(frozen=True)
class ExpansionNorm:
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0

    def total(self) -> int:
        return self.left + self.top + self.right + self.bottom

    def to_dict(self) -> dict[str, int]:
        return {
            "left_norm": self.left,
            "top_norm": self.top,
            "right_norm": self.right,
            "bottom_norm": self.bottom,
        }


@dataclass(frozen=True)
class RecoveryAction:
    action: str
    valid: bool
    candidate_valid: bool
    missing: tuple[str, ...]
    arguments: ExpansionNorm
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "valid": self.valid,
            "candidate_valid": self.candidate_valid,
            "missing": list(self.missing),
            "arguments": self.arguments.to_dict(),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class RecoveryRecommendation:
    candidate_valid: bool
    crop_complete: bool
    missing: tuple[str, ...]
    expand: ExpansionNorm
    confidence: float
    role: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_valid": self.candidate_valid,
            "crop_complete": self.crop_complete,
            "missing": list(self.missing),
            "expand": {
                "left": self.expand.left,
                "top": self.expand.top,
                "right": self.expand.right,
                "bottom": self.expand.bottom,
            },
            "confidence": self.confidence,
            "role": self.role,
        }


@dataclass(frozen=True)
class RecoveryConfig:
    max_turns: int = 6
    max_format_retries: int = 2
    max_subagents: int = 3
    max_content_bytes: int = 4096
    max_direction_norm: int = 500
    max_crop_area_ratio: float = 0.45
    max_crop_growth: float = 12.0
    recovery_max_tokens: int = 8192
    subagent_confidence_threshold: float = 0.60
    context_fraction: float = 0.50
    max_image_edge: int = 2400


@dataclass(frozen=True)
class EvidenceBox:
    id: str
    kind: str
    bbox: CropBox
    selected: bool = False


@dataclass(frozen=True)
class RecoverySkill:
    path: Path
    name: str
    description: str
    body: str
    sha256: str


@dataclass
class RecoveryObservation:
    turn: int
    crop_bbox: CropBox
    context_bbox: CropBox
    crop_image: Any
    context_image: Any
    boundary_locked: dict[str, bool]
    crop_scale: float = 1.0
    context_scale: float = 1.0


@dataclass(frozen=True)
class AgentCall:
    content: str
    reasoning: str
    finish_reason: str
    elapsed_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_metadata(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "finish_reason": self.finish_reason,
            "elapsed_seconds": self.elapsed_seconds,
            "content_bytes": len(self.content.encode("utf-8")),
            "content_chars": len(self.content),
            "reasoning_chars": len(self.reasoning),
            "error": self.error or None,
        }


@dataclass(frozen=True)
class ExpansionResult:
    before: CropBox
    after: CropBox
    requested_norm: ExpansionNorm
    applied_pixels: dict[str, float]
    boundary_locked: dict[str, bool]
    changed: bool
    area_cap: float
    limited_by: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": self.before.to_list(),
            "after": self.after.to_list(),
            "requested_norm": self.requested_norm.to_dict(),
            "applied_pixels": {
                key: round(value, 3) for key, value in self.applied_pixels.items()
            },
            "boundary_locked": self.boundary_locked,
            "changed": self.changed,
            "area_cap": round(self.area_cap, 3),
            "limited_by": list(self.limited_by),
        }


@dataclass
class RecoveryTurn:
    turn: int
    crop_before: CropBox
    crop_after: CropBox
    action: Optional[RecoveryAction]
    format_attempts: list[dict[str, Any]] = field(default_factory=list)
    expansion: Optional[ExpansionResult] = None
    score_before: float = 0.0
    score_after: float = 0.0
    used_subagents: bool = False
    subagents: dict[str, Any] = field(default_factory=dict)
    decision_source: str = "main"

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "crop_before": self.crop_before.to_list(),
            "crop_after": self.crop_after.to_list(),
            "action": self.action.to_dict() if self.action else None,
            "format_attempts": self.format_attempts,
            "expansion": self.expansion.to_dict() if self.expansion else None,
            "score_before": round(self.score_before, 4),
            "score_after": round(self.score_after, 4),
            "used_subagents": self.used_subagents,
            "subagents": self.subagents,
            "decision_source": self.decision_source,
        }


@dataclass
class RecoveryResult:
    status: str
    final_crop: CropBox
    best_crop: CropBox
    turns: list[RecoveryTurn]
    rejected: bool
    valid: bool
    best_score: float
    history_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "final_crop": self.final_crop.to_list(),
            "best_crop": self.best_crop.to_list(),
            "rejected": self.rejected,
            "valid": self.valid,
            "best_score": round(self.best_score, 4),
            "history_path": self.history_path,
            "turns": [turn.to_dict() for turn in self.turns],
        }
