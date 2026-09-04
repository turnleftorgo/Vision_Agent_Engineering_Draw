"""Compact one-image semantic association for the Super FAI pipeline."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

import FAI_DET_CROP_3 as core
from recovery.agents import call_structured_agent


SEMANTIC_FIELDS = {
    "marker_id",
    "candidate_valid",
    "fai_number",
    "spc_letter",
    "annotation_ids",
    "text_ids",
    "leader_path_ids",
    "arrowhead_ids",
    "target_ids",
    "complete",
    "missing",
    "confidence",
}

MISSING_VALUES = [
    "fai_number",
    "annotation",
    "text",
    "leader",
    "arrowhead",
    "target",
    "not_fai_marker",
]

SEMANTIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(SEMANTIC_FIELDS),
    "properties": {
        "marker_id": {"type": "string", "enum": ["F0"]},
        "candidate_valid": {"type": "boolean"},
        "fai_number": {"type": ["string", "null"], "maxLength": 24},
        "spc_letter": {"type": ["string", "null"], "maxLength": 16},
        "annotation_ids": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string"},
        },
        "text_ids": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string"},
        },
        "leader_path_ids": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string"},
        },
        "arrowhead_ids": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string"},
        },
        "target_ids": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string"},
        },
        "complete": {"type": "boolean"},
        "missing": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string", "enum": MISSING_VALUES},
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


@dataclass(frozen=True)
class LeaderPath:
    id: str
    segment_ids: tuple[str, ...]
    segments: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    bbox: core.BBox
    score: float


@dataclass
class CompactSemanticEvidence:
    image: Image.Image
    records: list[dict[str, Any]]
    visible_primitives: list[core.Primitive]
    paths: list[LeaderPath]

    @property
    def path_segments(self) -> dict[str, list[str]]:
        return {path.id: list(path.segment_ids) for path in self.paths}


def build_compact_selection_image(
    evidence: CompactSemanticEvidence,
    selected_ids: Iterable[str],
    selected_path_ids: Iterable[str],
) -> Image.Image:
    """Highlight Qwen's choices without restoring the discarded overlay noise."""
    image = evidence.image.copy()
    draw = ImageDraw.Draw(image)
    selected = set(selected_ids)
    selected_paths = set(selected_path_ids)
    stroke = max(3, round(min(image.size) * 0.003))
    for item in evidence.visible_primitives:
        if item.id not in selected:
            continue
        color = (255, 0, 0) if item.id == "F0" else (0, 150, 0)
        draw.rectangle(item.bbox.to_int_tuple(), outline=color, width=stroke)
    for path in evidence.paths:
        if path.id not in selected_paths:
            continue
        for segment in path.segments:
            draw.line(segment, fill=(0, 150, 0), width=stroke)
    return image


def _bbox_gap(first: core.BBox, second: core.BBox) -> float:
    a = first.ordered()
    b = second.ordered()
    dx = max(a.x1 - b.x2, b.x1 - a.x2, 0.0)
    dy = max(a.y1 - b.y2, b.y1 - a.y2, 0.0)
    return math.hypot(dx, dy)


def _line_length(segment: tuple[tuple[int, int], tuple[int, int]]) -> float:
    return math.dist(segment[0], segment[1])


def _line_bbox(segment: tuple[tuple[int, int], tuple[int, int]]) -> core.BBox:
    first, second = segment
    return core.BBox(
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[0], second[0]) + 1,
        max(first[1], second[1]) + 1,
    )


def _point_segment_distance(
    point: tuple[int, int],
    segment: tuple[tuple[int, int], tuple[int, int]],
) -> float:
    px, py = point
    (x1, y1), (x2, y2) = segment
    dx = x2 - x1
    dy = y2 - y1
    length_squared = dx * dx + dy * dy
    if length_squared <= 0:
        return math.hypot(px - x1, py - y1)
    ratio = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_squared))
    return math.hypot(px - (x1 + ratio * dx), py - (y1 + ratio * dy))


def _segment_gap(
    first: tuple[tuple[int, int], tuple[int, int]],
    second: tuple[tuple[int, int], tuple[int, int]],
) -> float:
    return min(
        _point_segment_distance(first[0], second),
        _point_segment_distance(first[1], second),
        _point_segment_distance(second[0], first),
        _point_segment_distance(second[1], first),
    )


def _direction_difference(
    first: tuple[tuple[int, int], tuple[int, int]],
    second: tuple[tuple[int, int], tuple[int, int]],
) -> float:
    def angle(segment: tuple[tuple[int, int], tuple[int, int]]) -> float:
        value = math.degrees(
            math.atan2(
                segment[1][1] - segment[0][1],
                segment[1][0] - segment[0][0],
            )
        )
        return value % 180.0

    difference = abs(angle(first) - angle(second))
    return min(difference, 180.0 - difference)


def _projected_overlap_ratio(
    first: tuple[tuple[int, int], tuple[int, int]],
    second: tuple[tuple[int, int], tuple[int, int]],
) -> float:
    dx = first[1][0] - first[0][0]
    dy = first[1][1] - first[0][1]
    length = math.hypot(dx, dy)
    if length <= 0:
        return 0.0
    ux, uy = dx / length, dy / length

    def interval(
        segment: tuple[tuple[int, int], tuple[int, int]],
    ) -> tuple[float, float]:
        values = [point[0] * ux + point[1] * uy for point in segment]
        return min(values), max(values)

    a1, a2 = interval(first)
    b1, b2 = interval(second)
    overlap = max(0.0, min(a2, b2) - max(a1, b1))
    shorter = min(a2 - a1, b2 - b1)
    return overlap / shorter if shorter > 0 else 0.0


def _deduplicate_lines(
    lines: list[tuple[core.Primitive, tuple[tuple[int, int], tuple[int, int]]]],
    tolerance: float,
) -> list[tuple[core.Primitive, tuple[tuple[int, int], tuple[int, int]]]]:
    retained: list[tuple[core.Primitive, tuple[tuple[int, int], tuple[int, int]]]] = []
    for item in sorted(lines, key=lambda pair: _line_length(pair[1]), reverse=True):
        duplicate = any(
            _direction_difference(item[1], existing[1]) <= 6.0
            and _segment_gap(item[1], existing[1]) <= tolerance
            and _projected_overlap_ratio(item[1], existing[1]) >= 0.55
            for existing in retained
        )
        if not duplicate:
            retained.append(item)
    return retained


def _inside_annotation_geometry(
    segment: tuple[tuple[int, int], tuple[int, int]],
    annotation_boxes: list[core.BBox],
) -> bool:
    for box in annotation_boxes:
        margin = max(5.0, min(20.0, math.hypot(box.width, box.height) * 0.025))
        if not all(box.contains_point(*point, margin=margin) for point in segment):
            continue
        if _line_length(segment) <= math.hypot(box.width, box.height) * 1.05:
            return True
    return False


def _segments_connect(
    first: tuple[tuple[int, int], tuple[int, int]],
    second: tuple[tuple[int, int], tuple[int, int]],
    snap: float,
) -> bool:
    endpoint_gap = min(math.dist(a, b) for a in first for b in second)
    if endpoint_gap <= snap:
        return True
    return (
        _direction_difference(first, second) >= 12.0
        and _segment_gap(first, second) <= snap
    )


def _inside_text_stroke(
    segment: tuple[tuple[int, int], tuple[int, int]],
    text_boxes: list[core.BBox],
) -> bool:
    midpoint = (
        round((segment[0][0] + segment[1][0]) / 2),
        round((segment[0][1] + segment[1][1]) / 2),
    )
    length = _line_length(segment)
    for box in text_boxes:
        expanded = box.expand(3, 3, 3, 3)
        if expanded.contains_point(*midpoint) and length <= max(
            box.width * 1.25, box.height * 7.0
        ):
            return True
    return False


def _annotation_hint(
    annotation: core.Primitive,
    text_primitives: list[core.Primitive],
) -> str:
    nearby = sorted(
        text_primitives, key=lambda item: _bbox_gap(item.bbox, annotation.bbox)
    )
    values = [
        item.text.strip()
        for item in nearby[:4]
        if item.text.strip()
        and _bbox_gap(item.bbox, annotation.bbox) <= max(20.0, annotation.bbox.height)
    ]
    return " | ".join(values)[:180]


def _is_heading(text: str) -> bool:
    normalized = re.sub(r"[^A-Z]", "", text.upper())
    return any(
        word in normalized
        for word in (
            "DIMENSIONS",
            "DATUMS",
            "GENERALNOTES",
            "NOTES",
            "DETAIL",
            "SECTION",
        )
    )


def select_annotation_candidates(
    primitives: list[core.Primitive],
    marker: core.Primitive,
    *,
    limit: int = 8,
) -> list[core.Primitive]:
    text = [item for item in primitives if item.kind == "ocr_text"]
    scored: list[tuple[float, core.Primitive]] = []
    for item in primitives:
        if item.kind != "annotation":
            continue
        overlap = item.bbox.intersection_area(marker.bbox) / max(item.bbox.area, 1.0)
        if overlap >= 0.50:
            continue
        hint = _annotation_hint(item, text)
        if _is_heading(hint):
            continue
        distance = _bbox_gap(item.bbox, marker.bbox)
        numeric_bonus = 90.0 if re.search(r"\d|[±Ø⌀]", hint) else 0.0
        guided_bonus = 35.0 if "LocateAnything" in item.source else 0.0
        size_penalty = math.sqrt(max(item.bbox.area, 1.0)) * 0.025
        score = numeric_bonus + guided_bonus - distance * 0.18 - size_penalty
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:limit]]


def build_leader_paths(
    primitives: list[core.Primitive],
    anchors: list[core.Primitive],
    terminal_candidates: list[core.Primitive],
    roi: Image.Image,
    *,
    max_paths: int = 12,
    max_segments_per_path: int = 12,
) -> list[LeaderPath]:
    text_boxes = [item.bbox for item in primitives if item.kind == "ocr_text"]
    annotation_boxes = [item.bbox for item in anchors if item.kind == "annotation"]
    line_items: list[
        tuple[core.Primitive, tuple[tuple[int, int], tuple[int, int]]]
    ] = []
    for item in primitives:
        if item.kind != "leader_segment" or len(item.points) < 2:
            continue
        segment = (item.points[0], item.points[-1])
        if _inside_text_stroke(segment, text_boxes) or _inside_annotation_geometry(
            segment, annotation_boxes
        ):
            continue
        line_items.append((item, segment))
    if not line_items:
        return []

    diagonal = math.hypot(roi.width, roi.height)
    snap = max(10.0, diagonal * 0.007)
    line_items = _deduplicate_lines(line_items, max(4.0, snap * 0.35))
    adjacency: list[set[int]] = [set() for _ in line_items]
    for first in range(len(line_items)):
        for second in range(first + 1, len(line_items)):
            if _segments_connect(line_items[first][1], line_items[second][1], snap):
                adjacency[first].add(second)
                adjacency[second].add(first)

    components: list[list[int]] = []
    unseen = set(range(len(line_items)))
    while unseen:
        seed = unseen.pop()
        component = [seed]
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current]:
                if neighbor not in unseen:
                    continue
                unseen.remove(neighbor)
                component.append(neighbor)
                frontier.append(neighbor)
        components.append(component)

    anchor_boxes = [item.bbox for item in anchors]
    terminal_boxes = [item.bbox for item in terminal_candidates]
    ranked: list[tuple[float, list[int]]] = []
    for component in components:
        component_boxes = [_line_bbox(line_items[index][1]) for index in component]
        union = core.BBox.union(component_boxes)
        if union is None:
            continue
        anchor_distance = min(
            (_bbox_gap(union, box) for box in anchor_boxes), default=float("inf")
        )
        terminal_distance = min(
            (_bbox_gap(union, box) for box in terminal_boxes), default=float("inf")
        )
        if anchor_distance > diagonal * 0.16:
            continue
        total_length = sum(_line_length(line_items[index][1]) for index in component)
        terminal_bonus = 100.0 if terminal_distance <= diagonal * 0.04 else 0.0
        branch_penalty = max(0, len(component) - max_segments_per_path) * 12.0
        score = (
            total_length * 0.06
            + terminal_bonus
            - anchor_distance * 0.3
            - branch_penalty
        )
        ranked.append((score, component))
    ranked.sort(key=lambda pair: pair[0], reverse=True)

    paths: list[LeaderPath] = []
    for score, component in ranked[:max_paths]:
        component.sort(
            key=lambda index: _line_length(line_items[index][1]), reverse=True
        )
        chosen = component[:max_segments_per_path]
        segments = tuple(line_items[index][1] for index in chosen)
        union = core.BBox.union([_line_bbox(segment) for segment in segments])
        if union is None:
            continue
        paths.append(
            LeaderPath(
                id=f"LP{len(paths)}",
                segment_ids=tuple(line_items[index][0].id for index in chosen),
                segments=segments,
                bbox=union,
                score=score,
            )
        )
    return paths


def _limited_by_reference(
    primitives: Iterable[core.Primitive],
    references: list[core.BBox],
    limit: int,
) -> list[core.Primitive]:
    values = list(primitives)
    values.sort(
        key=lambda item: min(
            (_bbox_gap(item.bbox, box) for box in references), default=0.0
        )
    )
    return values[:limit]


def _draw_label(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x, y = position
    bounds = draw.textbbox((x, y), label)
    draw.rectangle(
        (bounds[0] - 1, bounds[1] - 1, bounds[2] + 1, bounds[3] + 1), fill="white"
    )
    draw.text((x, y), label, fill=color)


def build_compact_semantic_evidence(
    roi: Image.Image,
    primitives: list[core.Primitive],
) -> CompactSemanticEvidence:
    marker = next(item for item in primitives if item.id == "F0")
    annotations = select_annotation_candidates(primitives, marker)
    anchor_boxes = [marker.bbox] + [item.bbox for item in annotations]
    terminal_pool = [
        item
        for item in primitives
        if item.kind in {"arrowhead", "triangle", "target_part"}
    ]
    paths = build_leader_paths(primitives, [marker] + annotations, terminal_pool, roi)
    path_boxes = [path.bbox for path in paths]
    references = anchor_boxes + path_boxes
    arrows = _limited_by_reference(
        (item for item in primitives if item.kind in {"arrowhead", "triangle"}),
        references,
        10,
    )
    targets = _limited_by_reference(
        (item for item in primitives if item.kind == "target_part"), references, 8
    )
    texts = _limited_by_reference(
        (
            item
            for item in primitives
            if item.kind == "ocr_text"
            and item.text.strip()
            and not _is_heading(item.text)
        ),
        anchor_boxes,
        16,
    )

    image = roi.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    stroke = max(2, round(min(image.size) * 0.0015))
    colors = {
        "fai_marker": (255, 0, 0),
        "annotation": (255, 140, 0),
        "arrowhead": (0, 190, 220),
        "triangle": (0, 150, 150),
        "target_part": (0, 170, 0),
    }
    visible = [marker] + annotations + arrows + targets
    for item in visible:
        color = colors[item.kind]
        draw.rectangle(item.bbox.to_int_tuple(), outline=color, width=stroke)
        _draw_label(
            draw, (round(item.bbox.x1) + 2, round(item.bbox.y1) + 2), item.id, color
        )
    for path in paths:
        for segment in path.segments:
            draw.line(segment, fill=(0, 80, 255), width=stroke)
        longest = max(path.segments, key=_line_length)
        midpoint = (
            round((longest[0][0] + longest[1][0]) / 2),
            round((longest[0][1] + longest[1][1]) / 2),
        )
        _draw_label(draw, midpoint, path.id, (0, 80, 255))

    records: list[dict[str, Any]] = [{"id": "F0", "type": "fai_marker"}]
    records.extend(
        {
            "id": item.id,
            "type": "annotation",
            **({"text_hint": hint} if (hint := _annotation_hint(item, texts)) else {}),
        }
        for item in annotations
    )
    records.extend(
        {"id": item.id, "type": "ocr_text", "text": item.text[:180]} for item in texts
    )
    records.extend(
        {"id": path.id, "type": "leader_path", "segment_count": len(path.segments)}
        for path in paths
    )
    records.extend({"id": item.id, "type": item.kind} for item in arrows + targets)
    return CompactSemanticEvidence(image, records, visible, paths)


def compact_semantic_prompt(records: list[dict[str, Any]]) -> str:
    evidence_json = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return f"""You are the semantic evidence selector for one FAI annotation group.

You receive exactly ONE engineering-drawing image. It uses compact labels:
- F0: selected FAI marker, red
- A*: annotation candidates, orange
- T*: OCR text records listed below; no text boxes are drawn
- LP*: merged leader-path candidates, blue
- H*: LocateAnything arrowheads, cyan
- G*: OpenCV arrowheads, teal
- R*: touched target-feature candidates, green

Select only evidence IDs belonging to F0. Work internally in this order:
1. Verify F0 contains literal FAI and an inspection number.
2. Select its annotation and text.
3. Select only connected leader paths from that annotation.
4. Select their terminal arrowheads and smallest touched target features.

Rules:
- Select IDs only. Never output or invent coordinates.
- Every selected ID must exist in EVIDENCE and match its ID family.
- Proximity alone does not prove association.
- Exclude neighboring FAI/SPC groups, headings such as DIMENSIONS, unrelated
  dimensions, datum/section/grid symbols, and ordinary part geometry.
- For duplicates, select the smallest candidate that fully contains the item.
- H* and G* are alternative proposals; do not select all overlaps.
- Leave absent categories empty and report them in missing.
- complete=true only for one coherent group containing its readable marker,
  annotation/text, all applicable leaders, terminal arrows, and target parts.
- Return exactly one schema-conforming JSON object with no prose or Markdown.

EVIDENCE:
{evidence_json}"""


def _string_list(value: Any, allowed: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(
        dict.fromkeys(
            item for item in value if isinstance(item, str) and item in allowed
        )
    )


def run_compact_semantic_mapping(
    client: Any,
    model: str,
    evidence: CompactSemanticEvidence,
    raw_output_path: Path,
    metadata_path: Path,
    *,
    max_tokens: int = 8192,
) -> dict[str, Any]:
    call = call_structured_agent(
        client,
        model,
        system_prompt=(
            "Perform one bounded evidence-selection task. Reason internally, but "
            "put only the final JSON object in message.content."
        ),
        user_prompt=compact_semantic_prompt(evidence.records),
        images=[evidence.image],
        schema_name="fai_compact_semantic_mapping_v5",
        schema=SEMANTIC_SCHEMA,
        max_tokens=max_tokens,
    )
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.write_text(call.content, encoding="utf-8")
    metadata = call.to_metadata()
    metadata.update(
        {
            "image_count": 1,
            "evidence_record_count": len(evidence.records),
            "leader_path_count": len(evidence.paths),
            "structured_output": "json_schema",
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if call.error or call.finish_reason == "length":
        return {
            "marker_id": "F0",
            "candidate_valid": True,
            "complete": False,
            "missing": ["semantic_protocol"],
            "confidence": 0.0,
            "semantic_error": call.error or f"finish_reason:{call.finish_reason}",
        }
    cleaned = re.sub(r"<think>.*?</think>", "", call.content, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return {
            "marker_id": "F0",
            "candidate_valid": True,
            "complete": False,
            "missing": ["semantic_protocol"],
            "confidence": 0.0,
            "semantic_error": f"invalid_json:{exc}",
        }
    if not isinstance(parsed, dict) or set(parsed) != SEMANTIC_FIELDS:
        return {
            "marker_id": "F0",
            "candidate_valid": True,
            "complete": False,
            "missing": ["semantic_protocol"],
            "confidence": 0.0,
            "semantic_error": "invalid_fields",
        }

    ids_by_type: dict[str, set[str]] = {}
    for record in evidence.records:
        ids_by_type.setdefault(str(record["type"]), set()).add(str(record["id"]))
    annotation_ids = _string_list(
        parsed.get("annotation_ids"), ids_by_type.get("annotation", set())
    )
    text_ids = _string_list(parsed.get("text_ids"), ids_by_type.get("ocr_text", set()))
    path_ids = _string_list(
        parsed.get("leader_path_ids"), ids_by_type.get("leader_path", set())
    )
    arrow_ids = _string_list(
        parsed.get("arrowhead_ids"),
        ids_by_type.get("arrowhead", set()) | ids_by_type.get("triangle", set()),
    )
    target_ids = _string_list(
        parsed.get("target_ids"), ids_by_type.get("target_part", set())
    )
    expanded_line_ids = list(
        dict.fromkeys(
            segment_id
            for path_id in path_ids
            for segment_id in evidence.path_segments.get(path_id, [])
        )
    )
    missing = [
        item
        for item in parsed.get("missing", [])
        if isinstance(item, str) and item in MISSING_VALUES
    ]
    return {
        "marker_id": "F0",
        "candidate_valid": bool(parsed.get("candidate_valid", True)),
        "fai_number": parsed.get("fai_number"),
        "spc_letter": parsed.get("spc_letter"),
        "annotation_ids": annotation_ids,
        "text_ids": text_ids,
        "parameter_text_ids": text_ids,
        "description_text_ids": [],
        "leader_path_ids": path_ids,
        "leader_ids": expanded_line_ids,
        "arrowhead_ids": arrow_ids,
        "target_ids": target_ids,
        "complete": bool(parsed.get("complete", False)),
        "missing": missing,
        "confidence": float(parsed.get("confidence", 0.0)),
        "compact_semantic": True,
    }
