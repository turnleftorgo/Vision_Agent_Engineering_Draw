"""Geometry and evidence contracts used by the Super vision stages."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def ordered(self) -> "BBox":
        return BBox(
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
    def center(self) -> tuple[float, float]:
        box = self.ordered()
        return ((box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def clamp(self, width: int, height: int) -> "BBox":
        box = self.ordered()
        return BBox(
            max(0.0, min(float(width), box.x1)),
            max(0.0, min(float(height), box.y1)),
            max(0.0, min(float(width), box.x2)),
            max(0.0, min(float(height), box.y2)),
        ).ordered()

    def expand(self, left: float, top: float, right: float, bottom: float) -> "BBox":
        box = self.ordered()
        return BBox(box.x1 - left, box.y1 - top, box.x2 + right, box.y2 + bottom)

    def translate(self, dx: float, dy: float) -> "BBox":
        box = self.ordered()
        return BBox(box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy)

    def intersection_area(self, other: "BBox") -> float:
        a = self.ordered()
        b = other.ordered()
        width = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
        height = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
        return width * height

    def iou(self, other: "BBox") -> float:
        intersection = self.intersection_area(other)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        box = self.ordered()
        return (
            box.x1 - margin <= x <= box.x2 + margin
            and box.y1 - margin <= y <= box.y2 + margin
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
    def union(cls, boxes: Iterable["BBox"]) -> Optional["BBox"]:
        values = [box.ordered() for box in boxes if box is not None and box.area > 0]
        if not values:
            return None
        return cls(
            min(box.x1 for box in values),
            min(box.y1 for box in values),
            max(box.x2 for box in values),
            max(box.y2 for box in values),
        )


@dataclass
class Primitive:
    id: str
    kind: str
    bbox: BBox
    source: str
    text: str = ""
    confidence: Optional[float] = None
    points: list[tuple[int, int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def prompt_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": self.id,
            "type": self.kind,
            "bbox": self.bbox.to_list(),
            "source": self.source,
        }
        if self.text:
            record["text"] = self.text
        if self.confidence is not None:
            record["confidence"] = round(float(self.confidence), 3)
        if self.points:
            record["points"] = [list(point) for point in self.points]
        return record
