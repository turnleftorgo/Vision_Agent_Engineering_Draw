#!/usr/bin/env python3
"""
FAI Detection and Complete Crop Pipeline (Version 4)

Version 4 keeps the multi-detector evidence pipeline from Version 3, but makes
the semantic-association boundary explicit:

* one Qwen semantic-association request per FAI candidate;
* Qwen selects existing evidence IDs and returns semantic values only;
* Qwen never returns crop coordinates or fallback boxes;
* Python validates every selected ID and computes the minimum enclosing crop;
* the crops directory contains a Qwen-selection visualization and the minimum
  enclosing crop for every structurally valid mapping;
* a structurally valid but incomplete mapping continues into crop validation,
  where its semantic ``missing`` list is treated as visual recovery context;
* normal validated output is never produced from an invalid semantic mapping.

The stable detection and geometry helpers remain in FAI_DET_CROP_3.py. This
file owns the Version 4 prompt, response contract, crop construction, artifact
layout, validation loop, and manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI
from PIL import Image, ImageDraw

import FAI_DET_CROP_3 as v3


VERSION = 4
DEFAULT_MAPPING_MAX_TOKENS = 16384
DEFAULT_VALIDATION_ATTEMPTS = 3
DEFAULT_MAX_LINE_EVIDENCE = 40

ID_FIELDS = (
    "annotation_ids",
    "parameter_text_ids",
    "description_text_ids",
    "leader_ids",
    "arrowhead_ids",
    "target_ids",
)

ALLOWED_KINDS = {
    "annotation_ids": {"annotation"},
    "parameter_text_ids": {"ocr_text", "annotation"},
    "description_text_ids": {"ocr_text", "annotation"},
    "leader_ids": {"leader_segment"},
    "arrowhead_ids": {"arrowhead", "triangle"},
    "target_ids": {"target_part"},
}

SEMANTIC_MAPPING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "marker_id": {"type": "string", "const": "F0"},
        "fai_number": {"type": ["string", "null"]},
        "spc_letter": {"type": ["string", "null"]},
        "annotation_ids": {"type": "array", "items": {"type": "string"}},
        "parameter_text_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "description_text_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "leader_ids": {"type": "array", "items": {"type": "string"}},
        "arrowhead_ids": {"type": "array", "items": {"type": "string"}},
        "target_ids": {"type": "array", "items": {"type": "string"}},
        "parameter_values": {
            "type": "array",
            "items": {"type": "string"},
        },
        "measurement_description": {"type": ["string", "null"]},
        "is_range_measurement": {"type": "boolean"},
        "complete": {"type": "boolean"},
        "missing": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": [
        "marker_id",
        "fai_number",
        "spc_letter",
        "annotation_ids",
        "parameter_text_ids",
        "description_text_ids",
        "leader_ids",
        "arrowhead_ids",
        "target_ids",
        "parameter_values",
        "measurement_description",
        "is_range_measurement",
        "complete",
        "missing",
        "confidence",
    ],
    "additionalProperties": False,
}


def semantic_mapping_prompt(primitives: list[v3.Primitive]) -> str:
    records = [primitive.prompt_record() for primitive in primitives]
    evidence_json = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return f"""You are the single semantic-association stage of an engineering-drawing FAI pipeline.

You receive two images of the same candidate ROI:
- IMAGE 1 is the clean ROI.
- IMAGE 2 is the complete evidence overlay.

Evidence IDs in IMAGE 2:
- F*: selected FAI marker (red)
- A*: measurement/tolerance annotation proposal (orange)
- T*: OCR text line (purple)
- H*: LocateAnything leader-arrowhead proposal (cyan)
- L*: OpenCV line or merged line-path proposal (blue)
- G*: OpenCV triangular-arrowhead proposal (teal)
- R*: touched local component or part proposal (green)

The selected marker is exactly F0. Associate exactly one coherent annotation group
with F0. Nearby FAI, SPC, datum, section, grid, dimension, and part evidence may belong
to other groups. Select them only when IMAGE 1 shows that they belong to F0's
measurement annotation or connected leader path.

Select the minimum sufficient set of existing evidence IDs that completely represents
F0's group:
1. Read the FAI number inside F0 and return it as a string, preserving leading zeros.
2. Read the associated SPC letter only if F0 has one.
3. Select the A* annotation regions belonging to F0.
4. Select all T*/A* parameter, tolerance, feature-control-frame, and directly
   associated description evidence belonging to F0.
5. Select every non-duplicate L* segment needed for the complete connected path from
   the annotation to its terminal arrowhead or terminal feature.
6. Select the real terminal arrowheads from H* or G*. Exclude 100-percent sampling
   triangles, text glyphs, circle fragments, hatch fragments, and ordinary corners.
7. Select the smallest useful R* region touched by each selected terminal arrowhead.

OpenCV L* proposals can be leaders, ordinary part outlines, dimension lines, table
borders, or hatch lines. Select an L* ID only when it belongs to F0's connected
measurement path. Do not select duplicate IDs for the same physical line segment.

Use IMAGE 1 to read semantics and verify physical connections. Use IMAGE 2 and the
Evidence dictionary to identify the matching IDs. OCR can be wrong; trust the visible
drawing when OCR and IMAGE 1 disagree.

Coordinate ownership rule:
- Output no bbox, point, polygon, pixel coordinate, normalized coordinate, crop
  coordinate, padding, fallback box, or boundary adjustment.
- Never alter an ID or the geometry behind it.
- Every selected ID must exist verbatim in the Evidence dictionary.
- If required evidence has no usable existing ID, leave that list empty, describe the
  missing evidence in `missing`, and set `complete` to false. Never invent an ID.

`measurement_description` is text directly associated with F0's measurement. A section
title such as SECTION A-A is not a measurement description unless the drawing clearly
uses it as part of the selected measurement instruction.

`is_range_measurement` is true only for an explicit range, min/max interval, or limit
pair. Optional SPC or description fields may be null when the drawing genuinely has
none; absence of a genuinely optional field does not make the mapping incomplete.

Set `complete` to true only when the selected IDs and semantic values are sufficient
for Python to construct one complete F0 crop.

Reason internally. After reasoning, response `content` must be exactly one JSON object
matching the required schema. Do not put reasoning, prose, Markdown, code fences, XML,
or coordinate fields in final `content`.

Evidence dictionary:
{evidence_json}

Required final JSON shape:
{{
  "marker_id": "F0",
  "fai_number": null,
  "spc_letter": null,
  "annotation_ids": [],
  "parameter_text_ids": [],
  "description_text_ids": [],
  "leader_ids": [],
  "arrowhead_ids": [],
  "target_ids": [],
  "parameter_values": [],
  "measurement_description": null,
  "is_range_measurement": false,
  "complete": false,
  "missing": [],
  "confidence": 0.0
}}
"""


def crop_validation_prompt(mapping: dict[str, Any]) -> str:
    """Build the crop validator prompt with semantic recovery context.

    ``complete=false`` is not a reason to skip crop validation.  The semantic
    stage's missing list is passed as a set of hypotheses for the validator to
    check against the pixels in the wider context image.
    """
    semantic_context = {
        "fai_number": mapping.get("fai_number"),
        "spc_letter": mapping.get("spc_letter"),
        "parameter_values": v3.string_list(mapping.get("parameter_values")),
        "measurement_description": mapping.get("measurement_description"),
        "selected_component_ids": [
            item for item in selected_ids(mapping) if item != "F0"
        ],
        "semantic_complete": bool(mapping.get("complete", False)),
        "semantic_missing": v3.string_list(mapping.get("missing")),
    }
    context_json = json.dumps(
        semantic_context,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"""Validate a proposed crop from an engineering drawing.

IMAGE 1 is the wider candidate context. The MAGENTA rectangle is the proposed
crop and the RED rectangle is the selected FAI marker.
IMAGE 2 is the proposed crop itself.

The crop is valid only if it fully contains one coherent FAI annotation group:
- the FAI circle and readable inspection number;
- associated SPC letter if present;
- parameter/tolerance/feature-control-frame and description text;
- every connected leader segment;
- every terminal arrowhead;
- a visible local portion of every component feature touched by those arrows.

Reject it if a relevant leader reaches the crop border, an arrowhead is clipped,
the touched feature is absent, or unrelated neighboring annotations dominate.
Inspect IMAGE 1 for connected evidence immediately outside the magenta rectangle.
Do not declare the crop valid merely because IMAGE 2 contains an FAI circle and
some numbers.

The semantic-association result below is diagnostic context, not ground truth.
Its `semantic_missing` entries are visual hypotheses that must be checked in the
images. `semantic_complete=false` does not automatically make the crop invalid:
- if a reported missing item is visible outside the magenta rectangle and belongs
  to this FAI group, return valid=false and request expansion toward it;
- if the supposedly missing item is already sufficiently visible inside IMAGE 2,
  do not expand solely because semantic association failed to select its ID;
- ignore a reported missing item if the pixels show that it belongs to a different
  annotation group.

Semantic-association context:
{context_json}

Expansion values are normalized fractions [0,1000] of the current crop width or
height. For example, bottom=150 means add 15% of the crop height at the bottom.

Return JSON only:
{{
  "valid": true,
  "missing": [],
  "expand_norm": {{"left": 0, "top": 0, "right": 0, "bottom": 0}},
  "confidence": 0.0
}}
"""


def validate_crop_with_semantic_context(
    client: OpenAI,
    model: str,
    context: Image.Image,
    crop: Image.Image,
    mapping: dict[str, Any],
    raw_output_path: Path,
) -> dict[str, Any]:
    response = v3.call_vision_model(
        client,
        model,
        crop_validation_prompt(mapping),
        [context, crop],
        max_tokens=8192,
        temperature=0.1,
    )
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.write_text(response, encoding="utf-8")
    try:
        data = v3.extract_json(response)
    except ValueError:
        return {
            "valid": False,
            "missing": ["validation_json"],
            "expand_norm": {"left": 80, "top": 80, "right": 80, "bottom": 80},
            "confidence": 0.0,
        }
    return data if isinstance(data, dict) else {
        "valid": False,
        "missing": ["validation_object"],
        "expand_norm": {"left": 80, "top": 80, "right": 80, "bottom": 80},
        "confidence": 0.0,
    }


def _message_reasoning_content(message: Any) -> str:
    value = getattr(message, "reasoning_content", None)
    if value is None:
        extra = getattr(message, "model_extra", None)
        if isinstance(extra, dict):
            value = extra.get("reasoning_content")
    return value if isinstance(value, str) else ""


def _usage_value(usage: Any, name: str) -> Optional[int]:
    value = getattr(usage, name, None) if usage is not None else None
    return int(value) if isinstance(value, (int, float)) else None


def request_semantic_mapping(
    client: OpenAI,
    model: str,
    raw_roi: Image.Image,
    overlay: Image.Image,
    primitives: list[v3.Primitive],
    raw_output_path: Path,
    metadata_path: Path,
    max_tokens: int,
    retries: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = semantic_mapping_prompt(primitives)
    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": v3.pil_to_data_url(raw_roi)},
        },
        {
            "type": "image_url",
            "image_url": {"url": v3.pil_to_data_url(overlay)},
        },
    ]
    response: Any = None
    last_error: Optional[Exception] = None
    started = time.monotonic()
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=0.1,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "fai_semantic_mapping_v4",
                        "strict": True,
                        "schema": SEMANTIC_MAPPING_SCHEMA,
                    },
                },
            )
            break
        except Exception as exc:  # local model servers can be temporarily busy
            last_error = exc
            if attempt >= retries:
                raise RuntimeError(
                    "V4 semantic mapping request failed. The Qwen endpoint must "
                    "support request-level JSON Schema output. "
                    f"Last error: {exc}"
                ) from exc
            time.sleep(1.0 + attempt)

    if response is None:
        raise RuntimeError(f"V4 semantic mapping produced no response: {last_error}")

    choice = response.choices[0]
    message = choice.message
    raw_content = message.content or ""
    if not isinstance(raw_content, str):
        raw_content = str(raw_content)
    reasoning = _message_reasoning_content(message)
    usage = getattr(response, "usage", None)
    finish_reason = str(getattr(choice, "finish_reason", "") or "")
    metadata: dict[str, Any] = {
        "finish_reason": finish_reason,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "prompt_tokens": _usage_value(usage, "prompt_tokens"),
        "completion_tokens": _usage_value(usage, "completion_tokens"),
        "total_tokens": _usage_value(usage, "total_tokens"),
        "reasoning_chars": len(reasoning),
        "content_chars": len(raw_content),
        "content_present": bool(raw_content.strip()),
        "structured_output": "json_schema",
        "max_tokens": max_tokens,
    }
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.write_text(raw_content, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if finish_reason != "stop":
        return (
            {
                "marker_id": "F0",
                "complete": False,
                "missing": [f"semantic_finish_reason:{finish_reason or 'unknown'}"],
                "confidence": 0.0,
            },
            {**metadata, "mapping_valid": False},
        )
    if not raw_content.strip():
        return (
            {
                "marker_id": "F0",
                "complete": False,
                "missing": ["semantic_mapping_content"],
                "confidence": 0.0,
            },
            {**metadata, "mapping_valid": False},
        )
    try:
        parsed = v3.extract_json(raw_content)
    except ValueError as exc:
        return (
            {
                "marker_id": "F0",
                "complete": False,
                "missing": ["semantic_mapping_json"],
                "confidence": 0.0,
                "parse_error": str(exc),
            },
            {**metadata, "mapping_valid": False},
        )
    if not isinstance(parsed, dict):
        return (
            {
                "marker_id": "F0",
                "complete": False,
                "missing": ["semantic_mapping_object"],
                "confidence": 0.0,
            },
            {**metadata, "mapping_valid": False},
        )

    mapping, validation_errors = normalize_mapping(parsed, primitives)
    mapping_valid = not validation_errors
    if validation_errors:
        mapping["complete"] = False
        mapping["missing"] = list(
            dict.fromkeys(v3.string_list(mapping.get("missing")) + validation_errors)
        )
    metadata["mapping_valid"] = mapping_valid
    metadata["mapping_validation_errors"] = validation_errors
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return mapping, metadata


def normalize_mapping(
    value: dict[str, Any],
    primitives: list[v3.Primitive],
) -> tuple[dict[str, Any], list[str]]:
    mapping = dict(value)
    errors: list[str] = []
    expected_fields = set(SEMANTIC_MAPPING_SCHEMA["required"])
    actual_fields = set(mapping)
    errors.extend(
        f"missing_field:{name}" for name in sorted(expected_fields - actual_fields)
    )
    errors.extend(
        f"unexpected_field:{name}" for name in sorted(actual_fields - expected_fields)
    )
    if mapping.get("marker_id") != "F0":
        errors.append("marker_id_must_be_F0")
    mapping["marker_id"] = "F0"

    kind_by_id = {primitive.id: primitive.kind for primitive in primitives}
    for field_name, allowed_kinds in ALLOWED_KINDS.items():
        if field_name in mapping and not isinstance(mapping[field_name], list):
            errors.append(f"invalid_type:{field_name}")
        requested = list(dict.fromkeys(v3.string_list(mapping.get(field_name))))
        accepted: list[str] = []
        for primitive_id in requested:
            kind = kind_by_id.get(primitive_id)
            if kind not in allowed_kinds:
                errors.append(f"invalid_{field_name}:{primitive_id}")
                continue
            accepted.append(primitive_id)
        mapping[field_name] = accepted

    number = mapping.get("fai_number")
    if number is not None and not isinstance(number, (str, int, float)):
        errors.append("invalid_type:fai_number")
    if isinstance(number, (int, float)):
        number = str(number)
    if isinstance(number, str):
        number = number.strip()
    if not number:
        number = None
        errors.append("fai_number_unreadable")
    mapping["fai_number"] = number

    spc = mapping.get("spc_letter")
    if spc is not None and not isinstance(spc, str):
        errors.append("invalid_type:spc_letter")
    mapping["spc_letter"] = spc.strip() if isinstance(spc, str) and spc.strip() else None
    if "parameter_values" in mapping and not isinstance(mapping["parameter_values"], list):
        errors.append("invalid_type:parameter_values")
    mapping["parameter_values"] = [
        str(item).strip()
        for item in mapping.get("parameter_values", [])
        if isinstance(item, (str, int, float)) and str(item).strip()
    ]
    description = mapping.get("measurement_description")
    if description is not None and not isinstance(description, str):
        errors.append("invalid_type:measurement_description")
    mapping["measurement_description"] = (
        description.strip()
        if isinstance(description, str) and description.strip()
        else None
    )
    if not isinstance(mapping.get("is_range_measurement"), bool):
        errors.append("invalid_type:is_range_measurement")
    mapping["is_range_measurement"] = bool(mapping.get("is_range_measurement", False))
    if not isinstance(mapping.get("complete"), bool):
        errors.append("invalid_type:complete")
    mapping["complete"] = bool(mapping.get("complete", False))
    if "missing" in mapping and not isinstance(mapping["missing"], list):
        errors.append("invalid_type:missing")
    mapping["missing"] = v3.string_list(mapping.get("missing"))
    try:
        raw_confidence = mapping.get("confidence", 0.0)
        if isinstance(raw_confidence, bool) or not isinstance(raw_confidence, (int, float)):
            raise TypeError(raw_confidence)
        mapping["confidence"] = max(0.0, min(1.0, float(raw_confidence)))
    except (TypeError, ValueError):
        mapping["confidence"] = 0.0
        errors.append("invalid_confidence")

    selected_components = [
        primitive_id
        for field_name in ID_FIELDS
        for primitive_id in mapping[field_name]
    ]
    if not selected_components:
        errors.append("no_components_selected")
    return mapping, list(dict.fromkeys(errors))


def _line_points(primitive: v3.Primitive) -> Optional[tuple[tuple[float, float], tuple[float, float]]]:
    if primitive.kind != "leader_segment" or len(primitive.points) < 2:
        return None
    start, end = primitive.points[0], primitive.points[-1]
    if start == end:
        return None
    return (float(start[0]), float(start[1])), (float(end[0]), float(end[1]))


def _cluster_geometry(
    cluster: list[v3.Primitive],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    longest = max(
        cluster,
        key=lambda item: math.hypot(
            item.points[-1][0] - item.points[0][0],
            item.points[-1][1] - item.points[0][1],
        ),
    )
    p0 = (float(longest.points[0][0]), float(longest.points[0][1]))
    p1 = (float(longest.points[-1][0]), float(longest.points[-1][1]))
    length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    unit = ((p1[0] - p0[0]) / length, (p1[1] - p0[1]) / length)
    points = [
        (float(point[0]), float(point[1]))
        for primitive in cluster
        for point in (primitive.points[0], primitive.points[-1])
    ]
    projections = [
        (point[0] - p0[0]) * unit[0] + (point[1] - p0[1]) * unit[1]
        for point in points
    ]
    low = min(projections)
    high = max(projections)
    start = (p0[0] + low * unit[0], p0[1] + low * unit[1])
    end = (p0[0] + high * unit[0], p0[1] + high * unit[1])
    return p0, unit, start, end


def _line_fits_cluster(
    primitive: v3.Primitive,
    cluster: list[v3.Primitive],
    perpendicular_tolerance: float,
    gap_tolerance: float,
    angle_tolerance_radians: float,
) -> bool:
    candidate = _line_points(primitive)
    if candidate is None:
        return False
    p0, unit, start, end = _cluster_geometry(cluster)
    c0, c1 = candidate
    candidate_length = math.hypot(c1[0] - c0[0], c1[1] - c0[1])
    candidate_unit = (
        (c1[0] - c0[0]) / candidate_length,
        (c1[1] - c0[1]) / candidate_length,
    )
    dot = max(-1.0, min(1.0, abs(unit[0] * candidate_unit[0] + unit[1] * candidate_unit[1])))
    if math.acos(dot) > angle_tolerance_radians:
        return False

    perpendicular = [
        abs((point[0] - p0[0]) * unit[1] - (point[1] - p0[1]) * unit[0])
        for point in (c0, c1)
    ]
    if max(perpendicular) > perpendicular_tolerance:
        return False

    cluster_interval = sorted(
        (
            (start[0] - p0[0]) * unit[0] + (start[1] - p0[1]) * unit[1],
            (end[0] - p0[0]) * unit[0] + (end[1] - p0[1]) * unit[1],
        )
    )
    candidate_interval = sorted(
        (
            (c0[0] - p0[0]) * unit[0] + (c0[1] - p0[1]) * unit[1],
            (c1[0] - p0[0]) * unit[0] + (c1[1] - p0[1]) * unit[1],
        )
    )
    gap = max(
        0.0,
        cluster_interval[0] - candidate_interval[1],
        candidate_interval[0] - cluster_interval[1],
    )
    return gap <= gap_tolerance


def merge_line_evidence(
    primitives: list[v3.Primitive],
    roi: Image.Image,
    max_lines: int,
) -> list[v3.Primitive]:
    lines = [
        primitive
        for primitive in primitives
        if primitive.kind == "leader_segment" and _line_points(primitive) is not None
    ]
    lines.sort(
        key=lambda item: math.hypot(
            item.points[-1][0] - item.points[0][0],
            item.points[-1][1] - item.points[0][1],
        ),
        reverse=True,
    )
    diagonal = math.hypot(roi.width, roi.height)
    perpendicular_tolerance = max(3.0, diagonal * 0.0018)
    gap_tolerance = max(10.0, diagonal * 0.006)
    angle_tolerance = math.radians(4.0)
    clusters: list[list[v3.Primitive]] = []
    for primitive in lines:
        for cluster in clusters:
            if _line_fits_cluster(
                primitive,
                cluster,
                perpendicular_tolerance,
                gap_tolerance,
                angle_tolerance,
            ):
                cluster.append(primitive)
                break
        else:
            clusters.append([primitive])

    merged: list[v3.Primitive] = []
    for cluster in clusters:
        _, _, start, end = _cluster_geometry(cluster)
        integer_start = (int(round(start[0])), int(round(start[1])))
        integer_end = (int(round(end[0])), int(round(end[1])))
        bbox = v3.BBox(
            min(integer_start[0], integer_end[0]),
            min(integer_start[1], integer_end[1]),
            max(integer_start[0], integer_end[0]) + 1,
            max(integer_start[1], integer_end[1]) + 1,
        ).clamp(roi.width, roi.height)
        merged.append(
            v3.Primitive(
                id="",
                kind="leader_segment",
                bbox=bbox,
                source="OpenCV-HoughLinesP-merged",
                points=[integer_start, integer_end],
                metadata={"source_ids": [item.id for item in cluster]},
            )
        )
    merged.sort(
        key=lambda item: math.hypot(
            item.points[-1][0] - item.points[0][0],
            item.points[-1][1] - item.points[0][1],
        ),
        reverse=True,
    )
    merged = merged[:max_lines]
    for index, primitive in enumerate(merged):
        primitive.id = f"L{index}"

    first_line_index = next(
        (
            index
            for index, primitive in enumerate(primitives)
            if primitive.kind == "leader_segment"
        ),
        len(primitives),
    )
    without_lines = [
        primitive for primitive in primitives if primitive.kind != "leader_segment"
    ]
    insert_at = min(first_line_index, len(without_lines))
    return without_lines[:insert_at] + merged + without_lines[insert_at:]


def selected_ids(mapping: dict[str, Any]) -> list[str]:
    values = ["F0"]
    for field_name in ID_FIELDS:
        values.extend(v3.string_list(mapping.get(field_name)))
    return list(dict.fromkeys(values))


def calculate_minimum_enclosing_crop(
    mapping: dict[str, Any],
    primitives: list[v3.Primitive],
    roi_bbox: v3.BBox,
    full_width: int,
    full_height: int,
) -> tuple[v3.BBox, v3.BBox, v3.BBox]:
    by_id = {primitive.id: primitive for primitive in primitives}
    chosen = [by_id[item] for item in selected_ids(mapping) if item in by_id]
    if not chosen or all(primitive.id == "F0" for primitive in chosen):
        raise ValueError("Qwen selected no A/T/L/H/G/R component for F0")

    exact_union = v3.BBox.union([primitive.bbox for primitive in chosen])
    if exact_union is None:
        raise ValueError("Selected evidence has no usable geometry")

    local_diagonal = math.hypot(roi_bbox.width, roi_bbox.height)
    arrow_context = max(24.0, local_diagonal * 0.045)
    crop_boxes = [primitive.bbox for primitive in chosen]
    crop_boxes.extend(
        primitive.bbox.expand(
            arrow_context,
            arrow_context,
            arrow_context,
            arrow_context,
        )
        for primitive in chosen
        if primitive.kind in {"arrowhead", "triangle"}
    )
    contextual_union = v3.BBox.union(crop_boxes)
    if contextual_union is None:
        contextual_union = exact_union
    padding = max(12.0, local_diagonal * 0.018)
    local_crop = contextual_union.expand(
        padding,
        padding,
        padding,
        padding,
    ).clamp(int(roi_bbox.width), int(roi_bbox.height))
    global_crop = local_crop.translate(roi_bbox.x1, roi_bbox.y1).clamp(
        full_width,
        full_height,
    )
    return exact_union, local_crop, global_crop


def build_qwen_selection_overlay(
    raw_roi: Image.Image,
    primitives: list[v3.Primitive],
    mapping: dict[str, Any],
    minimum_local_crop: Optional[v3.BBox] = None,
) -> Image.Image:
    ids = set(selected_ids(mapping))
    chosen = [primitive for primitive in primitives if primitive.id in ids]
    overlay = v3.build_evidence_overlay(raw_roi, chosen)
    if minimum_local_crop is not None:
        draw = ImageDraw.Draw(overlay)
        stroke = max(2, round(min(overlay.size) * 0.004))
        box = minimum_local_crop.to_int_tuple()
        draw.rectangle(box, outline=(220, 0, 220), width=stroke)
        label_y = max(0, box[1] - 18)
        draw.rectangle(
            (box[0], label_y, min(overlay.width, box[0] + 230), label_y + 17),
            fill=(255, 255, 255),
        )
        draw.text(
            (box[0] + 2, label_y + 1),
            "PYTHON MINIMUM CROP",
            fill=(220, 0, 220),
        )
    return overlay


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def mapping_artifact_record(
    mapping: dict[str, Any],
    metadata: dict[str, Any],
    primitives: list[v3.Primitive],
    exact_union: v3.BBox,
    local_crop: v3.BBox,
    global_crop: v3.BBox,
) -> dict[str, Any]:
    by_id = {primitive.id: primitive for primitive in primitives}
    ids = selected_ids(mapping)
    return {
        "mapping": mapping,
        "response_metadata": metadata,
        "selected_evidence_ids": ids,
        "selected_evidence": [
            by_id[item].prompt_record() for item in ids if item in by_id
        ],
        "selected_exact_union_roi_bbox": exact_union.to_list(),
        "minimum_crop_roi_bbox": local_crop.to_list(),
        "minimum_crop_full_image_bbox": global_crop.to_list(),
        "coordinate_owner": "Python",
    }


def write_overview(
    image: Image.Image,
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for record in records:
        marker = record.get("marker_bbox")
        minimum = record.get("minimum_crop_bbox")
        final = record.get("final_crop_bbox")
        if isinstance(marker, list) and len(marker) == 4:
            draw.rectangle(tuple(marker), outline=(255, 0, 0), width=3)
        if isinstance(minimum, list) and len(minimum) == 4:
            draw.rectangle(tuple(minimum), outline=(220, 0, 220), width=3)
        if isinstance(final, list) and len(final) == 4:
            draw.rectangle(tuple(final), outline=(0, 170, 0), width=4)
        if isinstance(minimum, list) and len(minimum) == 4:
            label = f"C{record['candidate_index']} FAI {record.get('fai_number') or '?'}"
            draw.text((minimum[0] + 2, max(0, minimum[1] - 16)), label, fill=(220, 0, 220))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _base_manifest(
    input_path: Path,
    image: Image.Image,
    args: argparse.Namespace,
    candidate_count: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "source": str(input_path),
        "source_size": [image.width, image.height],
        "coordinate_space": "absolute_pixels",
        "candidate_count": candidate_count,
        "models": {
            "locate": args.locate_model,
            "semantic_and_validation": args.qwen_model,
            "endpoint": args.endpoint,
        },
        "pipeline": {
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "mapping_max_tokens": args.mapping_max_tokens,
            "structured_semantic_output": "json_schema",
            "semantic_coordinate_output": False,
            "max_line_evidence": args.max_line_evidence,
            "tesseract_enabled": not args.no_tesseract,
            "verification_enabled": not args.no_verify,
            "validation_attempts": args.validation_attempts,
        },
        "artifacts": {
            "qwen_selected": "Qwen-selected evidence on the full candidate ROI",
            "minimum": "raw minimum enclosing crop computed by Python",
            "minimum_selected": "minimum crop with only Qwen-selected evidence",
            "validated": "final crop after successful crop validation",
        },
        "results": results,
    }


def process_image(args: argparse.Namespace) -> list[dict[str, Any]]:
    input_path = Path(args.image).expanduser().resolve()
    if input_path.suffix.lower() != ".png":
        raise ValueError("Version 4 currently accepts PNG input only.")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    output_dir = Path(args.output).expanduser().resolve()
    raw_dir = output_dir / "raw_responses"
    debug_dir = output_dir / "debug"
    roi_dir = debug_dir / "candidate_rois"
    evidence_dir = debug_dir / "evidence"
    crop_dir = output_dir / "crops"
    failed_dir = output_dir / "failed"
    crop_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(input_path).convert("RGB")
    client = OpenAI(
        base_url=args.endpoint,
        api_key=args.api_key,
        timeout=args.timeout,
    )

    marker_boxes = v3.detect_fai_candidates(
        client,
        args.locate_model,
        image,
        args.tile_size,
        args.tile_overlap,
        raw_dir,
        debug_dir / "tiles" if args.debug else None,
    )
    v3.log("[1/8] OpenCV circle-pair proposals + Qwen magnified validation")
    circle_pairs = v3.detect_circle_pair_candidates(image)
    circle_boxes = v3.qwen_validate_circle_pairs(
        client,
        args.qwen_model,
        image,
        circle_pairs,
        raw_dir / "circle_pair_validation.txt",
        debug_dir / "circle_pair_contact_sheet.png" if args.debug else None,
    )
    marker_boxes = v3.deduplicate_boxes(marker_boxes + circle_boxes)
    if not marker_boxes:
        v3.log("[1/8] Hybrid proposals found no FAI marker; trying full-page Qwen fallback")
        marker_boxes = v3.qwen_fai_fallback(
            client,
            args.qwen_model,
            image,
            raw_dir / "qwen_fai_fallback.txt",
        )
    if not marker_boxes:
        raise RuntimeError("No FAI marker candidates were found.")

    marker_boxes = marker_boxes[: args.max_candidates]
    v3.log(f"[1/8] Found {len(marker_boxes)} deduplicated FAI candidate(s)")
    results: list[dict[str, Any]] = []
    manifest_path = output_dir / "results.json"

    for index, marker_box in enumerate(marker_boxes):
        marker = v3.Primitive(
            id="F0",
            kind="fai_marker",
            bbox=marker_box,
            source="hybrid_fai_proposal",
        )
        roi_bbox = v3.candidate_roi(marker_box, image.width, image.height)
        raw_roi, raw_primitives, _ = v3.create_candidate_evidence(
            client,
            args.locate_model,
            image,
            marker,
            roi_bbox,
            raw_dir,
            index,
            use_tesseract=not args.no_tesseract,
        )
        primitives = merge_line_evidence(
            raw_primitives,
            raw_roi,
            args.max_line_evidence,
        )
        evidence_overlay = v3.build_evidence_overlay(raw_roi, primitives)

        if args.debug:
            roi_dir.mkdir(parents=True, exist_ok=True)
            evidence_dir.mkdir(parents=True, exist_ok=True)
            raw_roi.save(roi_dir / f"candidate_{index:03d}.png")
            evidence_overlay.save(evidence_dir / f"candidate_{index:03d}_evidence.png")
            save_json(
                evidence_dir / f"candidate_{index:03d}_primitives.json",
                [primitive.prompt_record() for primitive in primitives],
            )
            save_json(
                evidence_dir / f"candidate_{index:03d}_line_merge_sources.json",
                {
                    primitive.id: primitive.metadata.get("source_ids", [])
                    for primitive in primitives
                    if primitive.kind == "leader_segment"
                },
            )

        v3.log(f"[4/8] Candidate {index}: one Qwen semantic association")
        mapping_path = raw_dir / f"candidate_{index:03d}_mapping.txt"
        mapping_meta_path = raw_dir / f"candidate_{index:03d}_mapping_meta.json"
        try:
            mapping, mapping_meta = request_semantic_mapping(
                client,
                args.qwen_model,
                raw_roi,
                evidence_overlay,
                primitives,
                mapping_path,
                mapping_meta_path,
                args.mapping_max_tokens,
            )
        except Exception as exc:
            mapping = {
                "marker_id": "F0",
                "complete": False,
                "missing": ["semantic_mapping_request"],
                "confidence": 0.0,
            }
            mapping_meta = {
                "mapping_valid": False,
                "request_error": str(exc),
            }
            save_json(mapping_meta_path, mapping_meta)

        chosen = [item for item in selected_ids(mapping) if item != "F0"]
        v3.log(
            f"[4/8] Candidate {index}: Qwen selected {len(chosen)} component(s): "
            + (", ".join(chosen) if chosen else "none")
        )
        fai_number = mapping.get("fai_number")
        safe_number = v3.safe_fai_name(fai_number)
        named_base = (
            f"FAI{safe_number}_{index:03d}"
            if safe_number != "unknown"
            else f"Candidate{index:03d}"
        )
        record: dict[str, Any] = {
            "candidate_index": index,
            "fai_number": fai_number,
            "marker_bbox": marker_box.to_list(),
            "roi_bbox": roi_bbox.to_list(),
            "mapping": mapping,
            "mapping_response": mapping_meta,
            "selected_evidence_ids": selected_ids(mapping),
            "mapping_complete": bool(mapping.get("complete", False)),
            "status": "semantic_mapping_failed",
        }

        if not mapping_meta.get("mapping_valid", False):
            failure_overlay = build_qwen_selection_overlay(raw_roi, primitives, mapping)
            failure_overlay.save(failed_dir / f"{named_base}_qwen_selection_failed.png")
            save_json(failed_dir / f"{named_base}_mapping_failed.json", record)
            results.append(record)
            save_json(
                manifest_path,
                _base_manifest(input_path, image, args, len(marker_boxes), results),
            )
            v3.log(
                f"[5/8] Candidate {index}: semantic mapping failed; "
                "no normal crop was produced"
            )
            continue

        try:
            exact_union, minimum_local, minimum_global = calculate_minimum_enclosing_crop(
                mapping,
                primitives,
                roi_bbox,
                image.width,
                image.height,
            )
        except ValueError as exc:
            record["status"] = "minimum_crop_failed"
            record["error"] = str(exc)
            save_json(failed_dir / f"{named_base}_minimum_crop_failed.json", record)
            results.append(record)
            save_json(
                manifest_path,
                _base_manifest(input_path, image, args, len(marker_boxes), results),
            )
            continue

        incomplete_tag = "" if mapping.get("complete", False) else "_INCOMPLETE"
        artifact_base = f"{named_base}{incomplete_tag}"
        selection_overlay = build_qwen_selection_overlay(
            raw_roi,
            primitives,
            mapping,
            minimum_local,
        )
        selection_path = crop_dir / f"{artifact_base}_qwen_selected.png"
        minimum_path = crop_dir / f"{artifact_base}_minimum.png"
        minimum_selected_path = crop_dir / f"{artifact_base}_minimum_selected.png"
        selection_json_path = crop_dir / f"{artifact_base}_qwen_selection.json"
        selection_overlay.save(selection_path)
        image.crop(minimum_global.to_int_tuple()).save(minimum_path)
        selection_overlay.crop(minimum_local.to_int_tuple()).save(minimum_selected_path)
        selection_record = mapping_artifact_record(
            mapping,
            mapping_meta,
            primitives,
            exact_union,
            minimum_local,
            minimum_global,
        )
        save_json(selection_json_path, selection_record)
        v3.log(f"[5/8] Candidate {index}: Qwen selection saved: {selection_path}")
        v3.log(f"[6/8] Candidate {index}: Python minimum crop saved: {minimum_path}")

        record.update(
            {
                "status": "mapping_incomplete"
                if not mapping.get("complete", False)
                else "minimum_crop_ready",
                "selected_exact_union_roi_bbox": exact_union.to_list(),
                "minimum_crop_roi_bbox": minimum_local.to_list(),
                "minimum_crop_bbox": minimum_global.to_list(),
                "selection_image_path": str(selection_path),
                "selection_json_path": str(selection_json_path),
                "minimum_crop_path": str(minimum_path),
                "minimum_selected_path": str(minimum_selected_path),
                "validation_history": [],
            }
        )

        initial_mapping_complete = bool(mapping.get("complete", False))
        if not initial_mapping_complete:
            record["recovery_mode"] = "crop_validation_from_incomplete_mapping"
            record["semantic_missing"] = v3.string_list(mapping.get("missing"))
            v3.log(
                f"[7/8] Candidate {index}: Qwen marked mapping incomplete; "
                "continuing to crop validation with semantic missing context"
            )

        current_crop = minimum_global
        validation_history: list[dict[str, Any]] = []
        valid = args.no_verify
        if args.no_verify:
            record["status"] = "verification_skipped"
        else:
            for attempt in range(args.validation_attempts):
                v3.log(
                    f"[7/8] Candidate {index}: Qwen crop validation "
                    f"attempt {attempt + 1}/{args.validation_attempts}"
                )
                crop_image = image.crop(current_crop.to_int_tuple())
                context = v3.build_validation_context(
                    image,
                    roi_bbox,
                    marker_box,
                    current_crop,
                )
                validation = validate_crop_with_semantic_context(
                    client,
                    args.qwen_model,
                    context,
                    crop_image,
                    mapping,
                    raw_dir / f"candidate_{index:03d}_validation_{attempt:02d}.txt",
                )
                validation_history.append(
                    {
                        "attempt": attempt + 1,
                        "crop_bbox": current_crop.to_list(),
                        "result": validation,
                    }
                )
                if validation.get("valid", False):
                    valid = True
                    break
                if attempt + 1 >= args.validation_attempts:
                    break
                expansion = v3.expansion_from_validation(current_crop, validation)
                expanded = current_crop.expand(*expansion).clamp(
                    image.width,
                    image.height,
                )
                if expanded.to_int_tuple() == current_crop.to_int_tuple():
                    break
                current_crop = expanded

        record["validation_history"] = validation_history
        record["final_crop_bbox"] = current_crop.to_list()
        if valid:
            final_path = crop_dir / f"{named_base}_validated.png"
            image.crop(current_crop.to_int_tuple()).save(final_path)
            if args.no_verify:
                record["status"] = "verification_skipped"
            elif initial_mapping_complete:
                record["status"] = "validated"
            else:
                record["status"] = "validated_from_incomplete_mapping"
                record["recovered_from_incomplete_mapping"] = True
            record["final_crop_path"] = str(final_path)
            v3.log(f"[8/8] Candidate {index}: final crop saved: {final_path}")
        else:
            failed_crop_path = failed_dir / f"{named_base}_validation_failed.png"
            image.crop(current_crop.to_int_tuple()).save(failed_crop_path)
            record["status"] = "validation_failed"
            record["failed_crop_path"] = str(failed_crop_path)
            v3.log(
                f"[8/8] Candidate {index}: validation did not pass; "
                f"last attempt saved to {failed_crop_path}"
            )

        results.append(record)
        save_json(
            manifest_path,
            _base_manifest(input_path, image, args, len(marker_boxes), results),
        )

    if args.debug:
        write_overview(image, results, debug_dir / "overview.png")
    save_json(
        manifest_path,
        _base_manifest(input_path, image, args, len(marker_boxes), results),
    )
    v3.log(f"Complete. Results: {manifest_path}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Version 4: detect FAI groups, let Qwen select evidence IDs once, "
            "and let Python save the minimum enclosing crop."
        )
    )
    parser.add_argument("image", help="Path to the input PNG")
    parser.add_argument(
        "-o",
        "--output",
        default="output_v4",
        help="Output directory (default: output_v4)",
    )
    parser.add_argument("--endpoint", default=v3.DEFAULT_ENDPOINT)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LOCAL_VLM_API_KEY", v3.DEFAULT_API_KEY),
    )
    parser.add_argument("--locate-model", default=v3.DEFAULT_LOCATE_MODEL)
    parser.add_argument("--qwen-model", default=v3.DEFAULT_QWEN_MODEL)
    parser.add_argument("--tile-size", type=int, default=1200)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--mapping-max-tokens",
        type=int,
        default=DEFAULT_MAPPING_MAX_TOKENS,
        help="Maximum output tokens for the one semantic-association call",
    )
    parser.add_argument(
        "--max-line-evidence",
        type=int,
        default=DEFAULT_MAX_LINE_EVIDENCE,
        help="Maximum merged OpenCV line records shown to Qwen (default: 40)",
    )
    parser.add_argument(
        "--validation-attempts",
        type=int,
        default=DEFAULT_VALIDATION_ATTEMPTS,
        help="Maximum crop-validation attempts (default: 3)",
    )
    parser.add_argument("--no-tesseract", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.tile_size < 256:
        parser.error("--tile-size must be at least 256")
    if not 0.0 <= args.tile_overlap < 0.9:
        parser.error("--tile-overlap must be in [0.0, 0.9)")
    if args.max_candidates < 1:
        parser.error("--max-candidates must be at least 1")
    if args.mapping_max_tokens < 1024:
        parser.error("--mapping-max-tokens must be at least 1024")
    if args.max_line_evidence < 1:
        parser.error("--max-line-evidence must be at least 1")
    if args.validation_attempts < 1:
        parser.error("--validation-attempts must be at least 1")
    try:
        process_image(args)
    except Exception as exc:
        v3.log(f"ERROR: {exc}")
        if args.debug:
            raise
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
