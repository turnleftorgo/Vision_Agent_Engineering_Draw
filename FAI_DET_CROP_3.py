#!/usr/bin/env python3
"""
FAI Detection and Complete Crop Pipeline (Version 3)

Input:
    A raster PNG engineering drawing.

Pipeline:
    1. LocateAnything finds high-recall FAI marker candidates on overlapping tiles.
    2. Each FAI candidate is expanded into a local high-resolution ROI.
    3. LocateAnything proposes annotation blocks, arrowheads, and touched part regions.
    4. Optional Tesseract OCR and OpenCV line/triangle extraction create local evidence.
    5. Qwen3-VL associates evidence by ID and supplies semantic/OCR fallback information.
    6. Python computes the union crop from the selected evidence.
    7. Qwen3-VL validates crop completeness; Python expands and retries once if needed.

The models propose and associate evidence. PIL/OpenCV retain ownership of the final
pixel coordinates and physical crop operation.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw


DEFAULT_ENDPOINT = "http://127.0.0.1:8001/v1"
DEFAULT_API_KEY = "anything"
DEFAULT_LOCATE_MODEL = "LocateAnything-3B-8bit"
DEFAULT_QWEN_MODEL = "Qwen3.8-27B-MLX-8bit"

FAI_PROMPT = (
    'Locate all the instances that match the following description: '
    'an FAI inspection marker or balloon, consisting of a circle containing '
    'the literal text "FAI" and an identification number. Exclude SPC circles, '
    'datum circles, holes, ordinary circled numbers, and section labels.'
)

ANNOTATION_PROMPT = (
    "Locate all the instances that match the following description: "
    "a rectangular feature-control frame, tolerance table, or mechanical-"
    "drawing measurement annotation block containing numeric dimension or "
    "tolerance values, geometric tolerance symbols, or descriptive note text. "
    "Exclude FAI circles, SPC circles, section titles, and datum labels."
)

ARROW_PROMPT = (
    "Locate all the instances that match the following description: "
    "solid black triangular leader-line arrowheads in a mechanical engineering "
    "drawing. Exclude text characters, filled rectangles, circular markers, "
    "and ordinary part corners."
)

TARGET_PROMPT = (
    "Locate all the instances that match the following description: "
    "a hatched mechanical cross-section, sectioned component, or outlined local "
    "mechanical part near leader-line arrowheads. Include useful local geometry "
    "but exclude unrelated drawing views, title blocks, and annotation tables."
)


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
        b = self.ordered()
        return max(0.0, b.x2 - b.x1)

    @property
    def height(self) -> float:
        b = self.ordered()
        return max(0.0, b.y2 - b.y1)

    @property
    def center(self) -> tuple[float, float]:
        b = self.ordered()
        return ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def clamp(self, width: int, height: int) -> "BBox":
        b = self.ordered()
        return BBox(
            max(0.0, min(float(width), b.x1)),
            max(0.0, min(float(height), b.y1)),
            max(0.0, min(float(width), b.x2)),
            max(0.0, min(float(height), b.y2)),
        ).ordered()

    def expand(self, left: float, top: float, right: float, bottom: float) -> "BBox":
        b = self.ordered()
        return BBox(b.x1 - left, b.y1 - top, b.x2 + right, b.y2 + bottom)

    def translate(self, dx: float, dy: float) -> "BBox":
        b = self.ordered()
        return BBox(b.x1 + dx, b.y1 + dy, b.x2 + dx, b.y2 + dy)

    def intersects(self, other: "BBox") -> bool:
        a = self.ordered()
        b = other.ordered()
        return not (a.x2 < b.x1 or b.x2 < a.x1 or a.y2 < b.y1 or b.y2 < a.y1)

    def intersection_area(self, other: "BBox") -> float:
        a = self.ordered()
        b = other.ordered()
        w = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
        h = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
        return w * h

    def iou(self, other: "BBox") -> float:
        inter = self.intersection_area(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        b = self.ordered()
        return (
            b.x1 - margin <= x <= b.x2 + margin
            and b.y1 - margin <= y <= b.y2 + margin
        )

    def to_int_tuple(self) -> tuple[int, int, int, int]:
        b = self.ordered()
        return (
            int(math.floor(b.x1)),
            int(math.floor(b.y1)),
            int(math.ceil(b.x2)),
            int(math.ceil(b.y2)),
        )

    def to_list(self) -> list[int]:
        return list(self.to_int_tuple())

    @classmethod
    def union(cls, boxes: Iterable["BBox"]) -> Optional["BBox"]:
        values = [b.ordered() for b in boxes if b is not None and b.area > 0]
        if not values:
            return None
        return cls(
            min(b.x1 for b in values),
            min(b.y1 for b in values),
            max(b.x2 for b in values),
            max(b.y2 for b in values),
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
            record["points"] = [list(p) for p in self.points]
        return record


@dataclass
class CandidateResult:
    marker: Primitive
    roi_bbox: BBox
    primitives: list[Primitive]
    mapping: dict[str, Any]
    initial_crop: BBox
    final_crop: BBox
    validation: dict[str, Any]
    image_path: str = ""


def log(message: str) -> None:
    print(message, flush=True)


def pil_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def call_vision_model(
    client: OpenAI,
    model: str,
    prompt: str,
    images: list[Image.Image],
    *,
    max_tokens: int,
    temperature: float = 0.1,
    retries: int = 2,
) -> str:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(
        {"type": "image_url", "image_url": {"url": pil_to_data_url(image)}}
        for image in images
    )
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:  # model servers can be temporarily busy
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"Model call failed for {model}: {last_error}") from last_error


def extract_json(text: str) -> Any:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON value found in model response")


def normalized_box_to_pixels(values: Iterable[float], width: int, height: int) -> BBox:
    coords = list(values)
    if len(coords) != 4:
        raise ValueError(f"Expected four box coordinates, received {coords!r}")
    x1, y1, x2, y2 = [float(value) for value in coords]
    return BBox(
        x1 / 1000.0 * width,
        y1 / 1000.0 * height,
        x2 / 1000.0 * width,
        y2 / 1000.0 * height,
    ).clamp(width, height)


def parse_locate_response(text: str, width: int, height: int) -> list[BBox]:
    boxes: list[BBox] = []
    pattern = re.compile(
        r"<box>\s*<(\d+(?:\.\d+)?)>\s*<(\d+(?:\.\d+)?)>"
        r"\s*<(\d+(?:\.\d+)?)>\s*<(\d+(?:\.\d+)?)>\s*</box>"
    )
    for match in pattern.finditer(text):
        boxes.append(normalized_box_to_pixels(match.groups(), width, height))

    if boxes:
        return [box for box in boxes if box.area >= 4]

    try:
        data = extract_json(text)
    except ValueError:
        return []

    records: list[Any]
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = (
            data.get("boxes")
            or data.get("detections")
            or data.get("objects")
            or []
        )
    else:
        records = []

    for record in records:
        values: Any = record
        if isinstance(record, dict):
            values = (
                record.get("bbox")
                or record.get("box")
                or record.get("bbox_2d")
            )
        if not isinstance(values, list) or len(values) != 4:
            continue
        numeric = [float(value) for value in values]
        if max(numeric) <= 1.0:
            numeric = [value * 1000.0 for value in numeric]
        elif max(numeric) > 1000.0:
            boxes.append(BBox(*numeric).clamp(width, height))
            continue
        boxes.append(normalized_box_to_pixels(numeric, width, height))
    return [box for box in boxes if box.area >= 4]


def locate_boxes(
    client: OpenAI,
    model: str,
    image: Image.Image,
    prompt: str,
    raw_output_path: Optional[Path] = None,
) -> list[BBox]:
    response = call_vision_model(
        client,
        model,
        prompt,
        [image],
        max_tokens=4096,
        temperature=0.1,
    )
    if raw_output_path:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_path.write_text(response, encoding="utf-8")
    return parse_locate_response(response, image.width, image.height)


def axis_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    step = max(1, int(tile_size * (1.0 - overlap)))
    starts = list(range(0, max(1, length - tile_size + 1), step))
    final_start = length - tile_size
    if not starts or starts[-1] != final_start:
        if starts and final_start - starts[-1] < step * 0.25:
            starts[-1] = final_start
        else:
            starts.append(final_start)
    return sorted(set(starts))


def iter_tiles(
    image: Image.Image,
    tile_size: int,
    overlap: float,
) -> Iterable[tuple[int, int, Image.Image]]:
    for y in axis_starts(image.height, tile_size, overlap):
        for x in axis_starts(image.width, tile_size, overlap):
            x2 = min(image.width, x + tile_size)
            y2 = min(image.height, y + tile_size)
            yield x, y, image.crop((x, y, x2, y2))


def deduplicate_boxes(boxes: list[BBox]) -> list[BBox]:
    ordered = sorted(boxes, key=lambda box: box.area, reverse=True)
    kept: list[BBox] = []
    for box in ordered:
        cx, cy = box.center
        duplicate = False
        for existing in kept:
            ex, ey = existing.center
            center_distance = math.hypot(cx - ex, cy - ey)
            size_scale = max(
                3.0,
                min(
                    max(box.width, box.height),
                    max(existing.width, existing.height),
                )
                * 0.55,
            )
            if box.iou(existing) >= 0.25 or center_distance <= size_scale:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return sorted(kept, key=lambda box: (box.y1, box.x1))


def detect_fai_candidates(
    client: OpenAI,
    model: str,
    image: Image.Image,
    tile_size: int,
    overlap: float,
    raw_dir: Path,
    tile_debug_dir: Optional[Path],
) -> list[BBox]:
    all_boxes: list[BBox] = []
    tile_index = 0
    for offset_x, offset_y, tile in iter_tiles(image, tile_size, overlap):
        log(
            f"[1/7] LocateAnything FAI scan tile {tile_index}: "
            f"origin=({offset_x},{offset_y}), size={tile.width}x{tile.height}"
        )
        if tile_debug_dir:
            tile_debug_dir.mkdir(parents=True, exist_ok=True)
            tile.save(tile_debug_dir / f"tile_{tile_index:03d}.png")
        local_boxes = locate_boxes(
            client,
            model,
            tile,
            FAI_PROMPT,
            raw_dir / f"fai_tile_{tile_index:03d}.txt",
        )
        all_boxes.extend(box.translate(offset_x, offset_y) for box in local_boxes)
        tile_index += 1
    return deduplicate_boxes(all_boxes)


def qwen_fai_fallback(
    client: OpenAI,
    model: str,
    image: Image.Image,
    raw_output_path: Path,
) -> list[BBox]:
    prompt = """Find all FAI inspection markers in this engineering drawing.

An FAI marker is a circle containing the literal text FAI and an identification
number. Do not return SPC circles, hole circles, datum circles, section labels,
or ordinary circled numbers.

Return JSON only. Coordinates are integers normalized to [0,1000]:
{"boxes": [{"bbox": [x1,y1,x2,y2], "number": null, "confidence": 0.0}]}
"""
    response = call_vision_model(
        client,
        model,
        prompt,
        [image],
        max_tokens=2048,
        temperature=0.1,
    )
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.write_text(response, encoding="utf-8")
    try:
        data = extract_json(response)
    except ValueError:
        return []
    boxes: list[BBox] = []
    for item in data.get("boxes", []) if isinstance(data, dict) else []:
        values = item.get("bbox") if isinstance(item, dict) else None
        if isinstance(values, list) and len(values) == 4:
            boxes.append(normalized_box_to_pixels(values, image.width, image.height))
    return deduplicate_boxes(boxes)


def horizontal_divider_score(
    gray: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
) -> float:
    height, width = gray.shape
    x1 = max(0, int(cx - radius * 0.85))
    x2 = min(width, int(cx + radius * 0.85))
    y1 = max(0, cy - max(2, int(radius * 0.12)))
    y2 = min(height, cy + max(3, int(radius * 0.12)) + 1)
    band = gray[y1:y2, x1:x2] < 180
    if band.size == 0:
        return 0.0
    return float(max((row.mean() for row in band), default=0.0))


def detect_circle_pair_candidates(image: Image.Image) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    minimum_dimension = min(image.width, image.height)
    min_radius = max(6, int(minimum_dimension * 0.012))
    max_radius = max(15, min(80, int(minimum_dimension * 0.05)))
    detected = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(12, int(minimum_dimension * 0.025)),
        param1=80,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if detected is None:
        return []

    circles = np.round(detected[0]).astype(int).tolist()
    pairs: list[dict[str, Any]] = []
    for first_index, first in enumerate(circles):
        for second in circles[first_index + 1 :]:
            average_radius = (first[2] + second[2]) / 2.0
            radius_ratio = first[2] / max(1.0, float(second[2]))
            dx = abs(first[0] - second[0])
            dy = abs(first[1] - second[1])
            if not (
                0.75 <= radius_ratio <= 1.33
                and 1.70 * average_radius <= dx <= 3.00 * average_radius
                and dy <= 0.35 * average_radius
            ):
                continue
            first_score = horizontal_divider_score(
                gray,
                first[0],
                first[1],
                first[2],
            )
            second_score = horizontal_divider_score(
                gray,
                second[0],
                second[1],
                second[2],
            )
            if min(first_score, second_score) < 0.72:
                continue

            left_circle, right_circle = sorted((first, second), key=lambda item: item[0])
            left_bbox = BBox(
                left_circle[0] - left_circle[2],
                left_circle[1] - left_circle[2],
                left_circle[0] + left_circle[2],
                left_circle[1] + left_circle[2],
            ).clamp(image.width, image.height)
            right_bbox = BBox(
                right_circle[0] - right_circle[2],
                right_circle[1] - right_circle[2],
                right_circle[0] + right_circle[2],
                right_circle[1] + right_circle[2],
            ).clamp(image.width, image.height)
            pair_union = BBox.union([left_bbox, right_bbox])
            if pair_union is None:
                continue
            padding = average_radius * 0.65
            context_bbox = pair_union.expand(
                padding,
                padding,
                padding,
                padding,
            ).clamp(image.width, image.height)
            pairs.append(
                {
                    "left_bbox": left_bbox,
                    "right_bbox": right_bbox,
                    "context_bbox": context_bbox,
                    "divider_score": min(first_score, second_score),
                }
            )

    pairs.sort(
        key=lambda item: (
            -float(item["divider_score"]),
            item["context_bbox"].y1,
            item["context_bbox"].x1,
        )
    )
    deduplicated: list[dict[str, Any]] = []
    for pair in pairs:
        if any(
            pair["context_bbox"].iou(existing["context_bbox"]) >= 0.55
            for existing in deduplicated
        ):
            continue
        pair["id"] = f"P{len(deduplicated)}"
        deduplicated.append(pair)
    return deduplicated[:32]


def build_circle_pair_contact_sheet(
    image: Image.Image,
    pairs: list[dict[str, Any]],
) -> Image.Image:
    cell_width = 320
    cell_height = 190
    label_height = 22
    columns = min(4, max(1, len(pairs)))
    rows = math.ceil(len(pairs) / columns)
    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, pair in enumerate(pairs):
        column = index % columns
        row = index // columns
        origin_x = column * cell_width
        origin_y = row * cell_height
        crop = image.crop(pair["context_bbox"].to_int_tuple()).convert("RGB")
        available_width = cell_width - 12
        available_height = cell_height - label_height - 10
        scale = min(
            available_width / max(1, crop.width),
            available_height / max(1, crop.height),
        )
        resized = crop.resize(
            (
                max(1, int(crop.width * scale)),
                max(1, int(crop.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
        paste_x = origin_x + (cell_width - resized.width) // 2
        paste_y = origin_y + label_height + (available_height - resized.height) // 2
        sheet.paste(resized, (paste_x, paste_y))
        draw.rectangle(
            (
                origin_x,
                origin_y,
                origin_x + cell_width - 1,
                origin_y + cell_height - 1,
            ),
            outline=(80, 80, 80),
            width=1,
        )
        draw.text(
            (origin_x + 5, origin_y + 3),
            str(pair["id"]),
            fill=(200, 0, 0),
        )
    return sheet


def qwen_validate_circle_pairs(
    client: OpenAI,
    model: str,
    image: Image.Image,
    pairs: list[dict[str, Any]],
    raw_output_path: Path,
    contact_sheet_path: Optional[Path],
) -> list[BBox]:
    if not pairs:
        return []
    contact_sheet = build_circle_pair_contact_sheet(image, pairs)
    if contact_sheet_path:
        contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
        contact_sheet.save(contact_sheet_path)

    prompt = """This contact sheet contains magnified candidate pairs of circular
engineering-drawing markers. Each cell has an ID P0, P1, and so on.

Select a cell only when one circle visibly contains the literal uppercase text
"FAI" above a horizontal divider and an inspection number below it. An adjacent
SPC circle is supporting evidence but is not itself the FAI marker.

Reject datum symbols, hole circles, geometric-tolerance symbols, leader joints,
and circular-looking fragments of part geometry. Do not guess unreadable cells.

Return JSON only:
{"matches":[{"id":"P0","side":"left","fai_number":"10","confidence":0.0}]}
"""
    response = call_vision_model(
        client,
        model,
        prompt,
        [contact_sheet],
        max_tokens=2048,
        temperature=0.1,
    )
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.write_text(response, encoding="utf-8")
    try:
        data = extract_json(response)
    except ValueError:
        return []
    pair_by_id = {str(pair["id"]): pair for pair in pairs}
    matches = data.get("matches", []) if isinstance(data, dict) else []
    boxes: list[BBox] = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        pair = pair_by_id.get(str(match.get("id")))
        if not pair:
            continue
        side = str(match.get("side", "left")).lower()
        boxes.append(
            pair["right_bbox"] if side == "right" else pair["left_bbox"]
        )
    return deduplicate_boxes(boxes)


def candidate_roi(marker_bbox: BBox, width: int, height: int) -> BBox:
    marker = marker_bbox.ordered()
    cx, cy = marker.center
    half_width = min(1200.0, max(360.0, marker.width * 10.0, width * 0.35))
    half_height = min(1200.0, max(280.0, marker.height * 7.0, height * 0.32))
    return BBox(
        cx - half_width,
        cy - half_height,
        cx + half_width,
        cy + half_height,
    ).clamp(width, height)


def marker_local_bbox(marker_bbox: BBox, roi_bbox: BBox) -> BBox:
    return marker_bbox.translate(-roi_bbox.x1, -roi_bbox.y1)


def make_selection_overlay(
    roi_image: Image.Image,
    selected_marker: BBox,
    arrow_boxes: Optional[list[BBox]] = None,
) -> Image.Image:
    marked = roi_image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    marker = selected_marker.to_int_tuple()
    stroke = max(2, round(min(marked.size) * 0.004))
    draw.rectangle(marker, outline=(255, 0, 0), width=stroke)
    label_y = max(0, marker[1] - 18)
    draw.rectangle(
        (marker[0], label_y, marker[0] + 100, label_y + 16),
        fill=(255, 255, 255),
    )
    draw.text((marker[0] + 2, label_y + 1), "SELECTED FAI", fill=(220, 0, 0))
    for arrow in arrow_boxes or []:
        draw.rectangle(arrow.to_int_tuple(), outline=(0, 210, 220), width=stroke)
    return marked


def run_tesseract_lines(image: Image.Image) -> list[tuple[BBox, str, float]]:
    executable = shutil.which("tesseract")
    if not executable:
        return []
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    command = [
        executable,
        "stdin",
        "stdout",
        "--psm",
        "11",
        "-l",
        "eng",
        "tsv",
    ]
    try:
        completed = subprocess.run(
            command,
            input=buffer.getvalue(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=90,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []

    decoded = completed.stdout.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(decoded), delimiter="\t")
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in reader:
        text = (row.get("text") or "").strip()
        try:
            confidence = float(row.get("conf") or -1)
        except ValueError:
            confidence = -1
        if not text or confidence < 5:
            continue
        key = (
            row.get("page_num", ""),
            row.get("block_num", ""),
            row.get("par_num", ""),
            row.get("line_num", ""),
        )
        grouped.setdefault(key, []).append(row)

    results: list[tuple[BBox, str, float]] = []
    for rows in grouped.values():
        x1 = min(int(row["left"]) for row in rows)
        y1 = min(int(row["top"]) for row in rows)
        x2 = max(int(row["left"]) + int(row["width"]) for row in rows)
        y2 = max(int(row["top"]) + int(row["height"]) for row in rows)
        text = " ".join((row.get("text") or "").strip() for row in rows).strip()
        confidences = [
            float(row.get("conf") or 0)
            for row in rows
            if float(row.get("conf") or -1) >= 0
        ]
        confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
        bbox = BBox(x1, y1, x2, y2).clamp(image.width, image.height)
        if bbox.area >= 9 and text:
            results.append((bbox, text, confidence))
    return sorted(results, key=lambda item: (item[0].y1, item[0].x1))[:120]


def detect_line_segments(image: Image.Image) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 140, apertureSize=3)
    diagonal = math.hypot(image.width, image.height)
    min_length = max(18, int(diagonal * 0.025))
    max_gap = max(3, int(diagonal * 0.006))
    threshold = max(15, int(min_length * 0.55))
    detected = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=threshold,
        minLineLength=min_length,
        maxLineGap=max_gap,
    )
    if detected is None:
        return []

    unique: list[tuple[tuple[int, int], tuple[int, int]]] = []
    signatures: set[tuple[int, int, int, int]] = set()
    for raw in detected[:, 0]:
        x1, y1, x2, y2 = map(int, raw)
        if (x2, y2) < (x1, y1):
            x1, y1, x2, y2 = x2, y2, x1, y1
        signature = (
            round(x1 / 4),
            round(y1 / 4),
            round(x2 / 4),
            round(y2 / 4),
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        unique.append(((x1, y1), (x2, y2)))
    return sorted(
        unique,
        key=lambda line: math.hypot(
            line[1][0] - line[0][0],
            line[1][1] - line[0][1],
        ),
        reverse=True,
    )


def point_to_bbox_distance(point: tuple[int, int], bbox: BBox) -> float:
    x, y = point
    b = bbox.ordered()
    dx = max(b.x1 - x, 0.0, x - b.x2)
    dy = max(b.y1 - y, 0.0, y - b.y2)
    return math.hypot(dx, dy)


def filter_relevant_lines(
    lines: list[tuple[tuple[int, int], tuple[int, int]]],
    anchors: list[BBox],
    image: Image.Image,
    limit: int = 70,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if not lines:
        return []
    diagonal = math.hypot(image.width, image.height)
    margin = max(10.0, diagonal * 0.018)
    anchor_union = BBox.union(anchors)
    corridor = (
        anchor_union.expand(diagonal * 0.06, diagonal * 0.06, diagonal * 0.06, diagonal * 0.06)
        if anchor_union
        else BBox(0, 0, image.width, image.height)
    )
    scored: list[
        tuple[float, tuple[tuple[int, int], tuple[int, int]]]
    ] = []
    for line in lines:
        p1, p2 = line
        midpoint = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        min_distance = min(
            (
                point_to_bbox_distance(p1, anchor),
                point_to_bbox_distance(p2, anchor),
                point_to_bbox_distance((int(midpoint[0]), int(midpoint[1])), anchor),
            )
            for anchor in anchors
        ) if anchors else (0.0, 0.0, 0.0)
        distance = min(min_distance)
        if distance <= margin or corridor.contains_point(*midpoint):
            score = length - distance * 0.5
            scored.append((score, line))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [line for _, line in scored[:limit]]


def detect_triangle_candidates(
    image: Image.Image,
    line_segments: list[tuple[tuple[int, int], tuple[int, int]]],
    arrow_anchors: list[BBox],
    limit: int = 30,
) -> list[BBox]:
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = image.width * image.height
    line_endpoints = [point for line in line_segments for point in line]
    candidates: list[tuple[float, BBox]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 6 or area > max(600.0, roi_area * 0.003):
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        polygon = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
        if not 3 <= len(polygon) <= 6:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < 3 or max(w, h) > max(80, min(image.size) * 0.12):
            continue
        bbox = BBox(x, y, x + w, y + h)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity < 0.55:
            continue
        cx, cy = bbox.center
        anchor_distance = min(
            (point_to_bbox_distance((int(cx), int(cy)), anchor) for anchor in arrow_anchors),
            default=float("inf"),
        )
        endpoint_distance = min(
            (math.hypot(cx - px, cy - py) for px, py in line_endpoints),
            default=float("inf"),
        )
        threshold = max(18.0, math.hypot(image.width, image.height) * 0.025)
        if arrow_anchors and anchor_distance > threshold:
            continue
        if not arrow_anchors and endpoint_distance > threshold:
            continue
        score = solidity * 100.0 - min(anchor_distance, endpoint_distance) * 0.2
        candidates.append((score, bbox))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return deduplicate_boxes([bbox for _, bbox in candidates[:limit]])


def build_evidence_overlay(
    raw_roi: Image.Image,
    primitives: list[Primitive],
) -> Image.Image:
    image = raw_roi.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    colors = {
        "fai_marker": (255, 0, 0),
        "annotation": (255, 140, 0),
        "arrowhead": (0, 200, 220),
        "target_part": (0, 180, 0),
        "ocr_text": (180, 0, 200),
        "leader_segment": (0, 80, 255),
        "triangle": (0, 160, 160),
    }
    stroke = max(1, round(min(image.size) * 0.003))
    for primitive in primitives:
        color = colors.get(primitive.kind, (80, 80, 80))
        if primitive.kind == "leader_segment" and len(primitive.points) >= 2:
            draw.line(primitive.points, fill=color, width=stroke)
            anchor = primitive.points[len(primitive.points) // 2]
        else:
            draw.rectangle(primitive.bbox.to_int_tuple(), outline=color, width=stroke)
            anchor = (int(primitive.bbox.x1), int(primitive.bbox.y1))
        label = primitive.id
        text_bbox = draw.textbbox(anchor, label)
        label_box = (
            text_bbox[0] - 1,
            text_bbox[1] - 1,
            text_bbox[2] + 1,
            text_bbox[3] + 1,
        )
        draw.rectangle(label_box, fill=(255, 255, 255))
        draw.text(anchor, label, fill=color)
    return image


def create_candidate_evidence(
    locate_client: OpenAI,
    locate_model: str,
    full_image: Image.Image,
    marker: Primitive,
    roi_bbox: BBox,
    raw_dir: Path,
    candidate_index: int,
    use_tesseract: bool,
) -> tuple[Image.Image, list[Primitive], Image.Image]:
    roi = full_image.crop(roi_bbox.to_int_tuple()).convert("RGB")
    marker_local = marker_local_bbox(marker.bbox, roi_bbox)
    selected_overlay = make_selection_overlay(roi, marker_local)

    log(f"[2/7] Candidate {candidate_index}: LocateAnything annotation proposal")
    annotation_boxes = locate_boxes(
        locate_client,
        locate_model,
        selected_overlay,
        ANNOTATION_PROMPT,
        raw_dir / f"candidate_{candidate_index:03d}_annotation.txt",
    )
    log(f"[2/7] Candidate {candidate_index}: LocateAnything arrowhead proposal")
    arrow_boxes = locate_boxes(
        locate_client,
        locate_model,
        selected_overlay,
        ARROW_PROMPT,
        raw_dir / f"candidate_{candidate_index:03d}_arrows.txt",
    )

    target_overlay = make_selection_overlay(roi, marker_local, arrow_boxes)
    log(f"[2/7] Candidate {candidate_index}: LocateAnything target-part proposal")
    target_boxes = locate_boxes(
        locate_client,
        locate_model,
        target_overlay,
        TARGET_PROMPT,
        raw_dir / f"candidate_{candidate_index:03d}_target.txt",
    )

    primitives: list[Primitive] = [
        Primitive("F0", "fai_marker", marker_local, "LocateAnything")
    ]
    primitives.extend(
        Primitive(f"A{index}", "annotation", box, "LocateAnything")
        for index, box in enumerate(deduplicate_boxes(annotation_boxes))
    )
    primitives.extend(
        Primitive(f"H{index}", "arrowhead", box, "LocateAnything")
        for index, box in enumerate(deduplicate_boxes(arrow_boxes))
    )
    primitives.extend(
        Primitive(f"R{index}", "target_part", box, "LocateAnything")
        for index, box in enumerate(deduplicate_boxes(target_boxes))
    )

    ocr_boxes: list[BBox] = []
    if use_tesseract:
        log(f"[3/7] Candidate {candidate_index}: local OCR")
        ocr_results = run_tesseract_lines(roi)
        ocr_boxes = [bbox for bbox, _, _ in ocr_results]
        primitives.extend(
            Primitive(
                f"T{index}",
                "ocr_text",
                bbox,
                "Tesseract",
                text=text,
                confidence=confidence,
            )
            for index, (bbox, text, confidence) in enumerate(ocr_results)
        )

    log(f"[3/7] Candidate {candidate_index}: OpenCV line and arrow geometry")
    all_lines = detect_line_segments(roi)
    anchors = (
        annotation_boxes
        + arrow_boxes
        + target_boxes
        + ocr_boxes
        + [marker_local]
    )
    lines = filter_relevant_lines(all_lines, anchors, roi)
    line_primitives: list[Primitive] = []
    for index, (start, end) in enumerate(lines):
        line_bbox = BBox(
            min(start[0], end[0]),
            min(start[1], end[1]),
            max(start[0], end[0]) + 1,
            max(start[1], end[1]) + 1,
        )
        line_primitives.append(
            Primitive(
                f"L{index}",
                "leader_segment",
                line_bbox,
                "OpenCV-HoughLinesP",
                points=[start, end],
            )
        )
    primitives.extend(line_primitives)

    triangle_boxes = detect_triangle_candidates(roi, lines, arrow_boxes)
    primitives.extend(
        Primitive(f"G{index}", "triangle", box, "OpenCV-contour")
        for index, box in enumerate(triangle_boxes)
    )

    evidence_overlay = build_evidence_overlay(roi, primitives)
    return roi, primitives, evidence_overlay


def semantic_mapping_prompt(primitives: list[Primitive]) -> str:
    records = [primitive.prompt_record() for primitive in primitives]
    return f"""You are the semantic association stage of an engineering-drawing pipeline.

IMAGE 1 is the clean candidate ROI.
IMAGE 2 is the same ROI with evidence IDs:
- F*: selected FAI marker (red)
- A*: measurement/tolerance annotation candidate (orange)
- H*: leader arrowhead candidate (cyan)
- R*: local touched-part candidate (green)
- T*: OCR text line (purple)
- L*: OpenCV line segment (blue)
- G*: OpenCV triangular arrowhead candidate (teal)

The selected marker is F0. Build exactly one complete FAI annotation group for F0.
Do not associate a nearby dimension, section arrow, grid label, datum, or SPC marker
unless it is connected to F0's measurement annotation.

A complete crop must contain:
1. F0 and its inspection number.
2. Associated SPC letter if present.
3. Parameter, tolerance, feature-control-frame, and description text.
4. All leader segments connected to that annotation.
5. All terminal arrowheads of those leaders.
6. The smallest useful local component/cross-section touched by each arrowhead.

Prefer selecting evidence IDs. Never alter the coordinates behind an ID.
Every selected ID must exist verbatim in the Evidence dictionary. If a required
evidence type is absent, leave its ID list empty and use the corresponding
fallback box; never invent IDs.
OpenCV line candidates include both leaders and ordinary part/dimension lines; select
only line IDs that are part of the FAI leader path.

measurement_description means an instruction sentence directly associated with
the FAI measurement. A section title such as "SECTION A-A" is not a measurement
description. is_range_measurement is true only when the annotation explicitly
specifies a min/max interval, range, or limit pair; a two-row feature-control
frame by itself is not a range measurement.

If evidence is missing, provide a fallback box in normalized ROI coordinates
[0,1000]. The complete_group fallback is a safety envelope that includes the full
annotation, every connected leader/arrowhead, and the touched local part. Do not
expand it to unrelated neighboring views.

Evidence dictionary:
{json.dumps(records, ensure_ascii=False, separators=(",", ":"))}

Return JSON only:
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
  "fallback_boxes_norm": {{
    "annotation": null,
    "target": null,
    "complete_group": null
  }},
  "complete": false,
  "missing": [],
  "confidence": 0.0
}}
"""


def run_semantic_mapping(
    client: OpenAI,
    model: str,
    raw_roi: Image.Image,
    overlay: Image.Image,
    primitives: list[Primitive],
    raw_output_path: Path,
) -> dict[str, Any]:
    response = call_vision_model(
        client,
        model,
        semantic_mapping_prompt(primitives),
        [raw_roi, overlay],
        max_tokens=8192,
        temperature=0.1,
    )
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.write_text(response, encoding="utf-8")
    try:
        data = extract_json(response)
    except ValueError as exc:
        return {
            "marker_id": "F0",
            "complete": False,
            "missing": ["semantic_mapping_json"],
            "confidence": 0.0,
            "parse_error": str(exc),
            "raw_response": response,
        }
    if not isinstance(data, dict):
        return {
            "marker_id": "F0",
            "complete": False,
            "missing": ["semantic_mapping_object"],
            "confidence": 0.0,
        }
    valid_kinds = {
        "annotation_ids": {"annotation"},
        "parameter_text_ids": {"ocr_text", "annotation"},
        "description_text_ids": {"ocr_text", "annotation"},
        "leader_ids": {"leader_segment"},
        "arrowhead_ids": {"arrowhead", "triangle"},
        "target_ids": {"target_part"},
    }
    kind_by_id = {primitive.id: primitive.kind for primitive in primitives}
    for key, allowed_kinds in valid_kinds.items():
        data[key] = [
            primitive_id
            for primitive_id in string_list(data.get(key))
            if kind_by_id.get(primitive_id) in allowed_kinds
        ]
    data["marker_id"] = "F0"
    return data


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


def fallback_box(
    mapping: dict[str, Any],
    name: str,
    roi_width: int,
    roi_height: int,
) -> Optional[BBox]:
    container = mapping.get("fallback_boxes_norm")
    if not isinstance(container, dict):
        return None
    values = container.get(name)
    if not isinstance(values, list) or len(values) != 4:
        return None
    try:
        return normalized_box_to_pixels(values, roi_width, roi_height)
    except (TypeError, ValueError):
        return None


def calculate_initial_crop(
    mapping: dict[str, Any],
    primitives: list[Primitive],
    roi_bbox: BBox,
    full_width: int,
    full_height: int,
) -> BBox:
    by_id = {primitive.id: primitive for primitive in primitives}
    selected_primitives = [
        by_id[primitive_id]
        for primitive_id in mapping_selected_ids(mapping)
        if primitive_id in by_id
    ]
    selected = [primitive.bbox for primitive in selected_primitives]

    selected_kinds = {
        by_id[primitive_id].kind
        for primitive_id in mapping_selected_ids(mapping)
        if primitive_id in by_id
    }
    if "annotation" not in selected_kinds:
        selected.extend(
            primitive.bbox
            for primitive in primitives
            if primitive.kind == "annotation"
        )
    if "arrowhead" not in selected_kinds and "triangle" not in selected_kinds:
        selected.extend(
            primitive.bbox
            for primitive in primitives
            if primitive.kind in {"arrowhead", "triangle"}
        )
    if "target_part" not in selected_kinds:
        selected.extend(
            primitive.bbox
            for primitive in primitives
            if primitive.kind == "target_part"
        )

    for name in ("annotation", "target", "complete_group"):
        candidate = fallback_box(mapping, name, int(roi_bbox.width), int(roi_bbox.height))
        if candidate is not None:
            selected.append(candidate)

    local_diagonal = math.hypot(roi_bbox.width, roi_bbox.height)
    arrow_context = max(24.0, local_diagonal * 0.045)
    selected.extend(
        primitive.bbox.expand(
            arrow_context,
            arrow_context,
            arrow_context,
            arrow_context,
        )
        for primitive in selected_primitives
        if primitive.kind in {"arrowhead", "triangle"}
    )

    local_union = BBox.union(selected)
    if local_union is None:
        local_union = BBox(0, 0, roi_bbox.width, roi_bbox.height)

    padding = max(12.0, local_diagonal * 0.018)
    local_crop = local_union.expand(padding, padding, padding, padding)
    return local_crop.translate(roi_bbox.x1, roi_bbox.y1).clamp(
        full_width,
        full_height,
    )


def validation_prompt() -> str:
    return """Validate a proposed crop from an engineering drawing.

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

Expansion values are normalized fractions [0,1000] of the current crop width or
height. For example, bottom=150 means add 15% of the crop height at the bottom.

Return JSON only:
{
  "valid": true,
  "missing": [],
  "expand_norm": {"left": 0, "top": 0, "right": 0, "bottom": 0},
  "confidence": 0.0
}
"""


def validate_crop(
    client: OpenAI,
    model: str,
    context: Image.Image,
    crop: Image.Image,
    raw_output_path: Path,
) -> dict[str, Any]:
    response = call_vision_model(
        client,
        model,
        validation_prompt(),
        [context, crop],
        max_tokens=8192,
        temperature=0.1,
    )
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.write_text(response, encoding="utf-8")
    try:
        data = extract_json(response)
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


def build_validation_context(
    full_image: Image.Image,
    roi_bbox: BBox,
    marker_bbox: BBox,
    crop_bbox: BBox,
) -> Image.Image:
    context = full_image.crop(roi_bbox.to_int_tuple()).convert("RGB")
    draw = ImageDraw.Draw(context)
    local_crop = crop_bbox.translate(-roi_bbox.x1, -roi_bbox.y1)
    local_marker = marker_bbox.translate(-roi_bbox.x1, -roi_bbox.y1)
    stroke = max(2, round(min(context.size) * 0.004))
    draw.rectangle(local_crop.to_int_tuple(), outline=(220, 0, 220), width=stroke)
    draw.rectangle(local_marker.to_int_tuple(), outline=(255, 0, 0), width=stroke)
    draw.text(
        (max(0, int(local_crop.x1) + 2), max(0, int(local_crop.y1) + 2)),
        "PROPOSED CROP",
        fill=(220, 0, 220),
    )
    return context


def expansion_from_validation(
    bbox: BBox,
    validation: dict[str, Any],
) -> tuple[float, float, float, float]:
    values = validation.get("expand_norm")
    if not isinstance(values, dict):
        values = {}
    try:
        left = max(0.0, min(500.0, float(values.get("left", 0))))
        top = max(0.0, min(500.0, float(values.get("top", 0))))
        right = max(0.0, min(500.0, float(values.get("right", 0))))
        bottom = max(0.0, min(500.0, float(values.get("bottom", 0))))
    except (TypeError, ValueError):
        left = top = right = bottom = 0.0

    if not validation.get("valid") and left + top + right + bottom == 0:
        left = top = right = bottom = 80.0
    return (
        bbox.width * left / 1000.0,
        bbox.height * top / 1000.0,
        bbox.width * right / 1000.0,
        bbox.height * bottom / 1000.0,
    )


def annotate_overview(
    image: Image.Image,
    results: list[CandidateResult],
    output_path: Path,
) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for index, result in enumerate(results):
        marker_box = result.marker.bbox.to_int_tuple()
        crop_box = result.final_crop.to_int_tuple()
        draw.rectangle(marker_box, outline=(255, 0, 0), width=3)
        draw.rectangle(crop_box, outline=(180, 0, 220), width=4)
        number = result.mapping.get("fai_number")
        label = f"C{index} FAI {number if number is not None else '?'}"
        draw.text((crop_box[0] + 2, max(0, crop_box[1] - 16)), label, fill=(180, 0, 220))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def result_record(result: CandidateResult) -> dict[str, Any]:
    valid_ids = {primitive.id for primitive in result.primitives}
    return {
        "fai_number": result.mapping.get("fai_number"),
        "spc_letter": result.mapping.get("spc_letter"),
        "parameter_values": result.mapping.get("parameter_values", []),
        "measurement_description": result.mapping.get("measurement_description"),
        "is_range_measurement": result.mapping.get("is_range_measurement", False),
        "marker_bbox": result.marker.bbox.to_list(),
        "roi_bbox": result.roi_bbox.to_list(),
        "initial_crop_bbox": result.initial_crop.to_list(),
        "final_crop_bbox": result.final_crop.to_list(),
        "selected_evidence_ids": [
            primitive_id
            for primitive_id in mapping_selected_ids(result.mapping)
            if primitive_id in valid_ids
        ],
        "mapping_complete": result.mapping.get("complete", False),
        "mapping_missing": result.mapping.get("missing", []),
        "mapping_confidence": result.mapping.get("confidence", 0.0),
        "validation": result.validation,
        "image_path": result.image_path,
    }


def safe_fai_name(value: Any) -> str:
    if value is None:
        return "unknown"
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return cleaned or "unknown"


def process_image(args: argparse.Namespace) -> list[CandidateResult]:
    input_path = Path(args.image).expanduser().resolve()
    if input_path.suffix.lower() != ".png":
        raise ValueError("Version 3 currently accepts PNG input only.")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_responses"
    debug_dir = output_dir / "debug"
    roi_dir = debug_dir / "candidate_rois"
    evidence_dir = debug_dir / "evidence"
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(input_path).convert("RGB")
    client = OpenAI(
        base_url=args.endpoint,
        api_key=args.api_key,
        timeout=args.timeout,
    )

    marker_boxes = detect_fai_candidates(
        client,
        args.locate_model,
        image,
        args.tile_size,
        args.tile_overlap,
        raw_dir,
        debug_dir / "tiles" if args.debug else None,
    )
    log("[1/7] OpenCV circle-pair proposals + Qwen magnified validation")
    circle_pairs = detect_circle_pair_candidates(image)
    circle_boxes = qwen_validate_circle_pairs(
        client,
        args.qwen_model,
        image,
        circle_pairs,
        raw_dir / "circle_pair_validation.txt",
        debug_dir / "circle_pair_contact_sheet.png" if args.debug else None,
    )
    marker_boxes = deduplicate_boxes(marker_boxes + circle_boxes)
    if not marker_boxes:
        log("[1/7] Hybrid proposals found no FAI marker; trying full-page Qwen fallback")
        marker_boxes = qwen_fai_fallback(
            client,
            args.qwen_model,
            image,
            raw_dir / "qwen_fai_fallback.txt",
        )
    if not marker_boxes:
        raise RuntimeError("No FAI marker candidates were found.")

    marker_boxes = marker_boxes[: args.max_candidates]
    log(f"[1/7] Found {len(marker_boxes)} deduplicated FAI candidate(s)")

    results: list[CandidateResult] = []
    for index, marker_box in enumerate(marker_boxes):
        marker = Primitive(
            id="F0",
            kind="fai_marker",
            bbox=marker_box,
            source="LocateAnything",
        )
        roi_bbox = candidate_roi(marker_box, image.width, image.height)
        raw_roi, primitives, overlay = create_candidate_evidence(
            client,
            args.locate_model,
            image,
            marker,
            roi_bbox,
            raw_dir,
            index,
            use_tesseract=not args.no_tesseract,
        )

        if args.debug:
            roi_dir.mkdir(parents=True, exist_ok=True)
            evidence_dir.mkdir(parents=True, exist_ok=True)
            raw_roi.save(roi_dir / f"candidate_{index:03d}.png")
            overlay.save(evidence_dir / f"candidate_{index:03d}_evidence.png")
            (evidence_dir / f"candidate_{index:03d}_primitives.json").write_text(
                json.dumps(
                    [primitive.prompt_record() for primitive in primitives],
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        log(f"[4/7] Candidate {index}: Qwen semantic association")
        mapping = run_semantic_mapping(
            client,
            args.qwen_model,
            raw_roi,
            overlay,
            primitives,
            raw_dir / f"candidate_{index:03d}_mapping.txt",
        )
        initial_crop = calculate_initial_crop(
            mapping,
            primitives,
            roi_bbox,
            image.width,
            image.height,
        )
        final_crop = initial_crop
        validation: dict[str, Any] = {
            "valid": None,
            "missing": [],
            "expand_norm": {"left": 0, "top": 0, "right": 0, "bottom": 0},
            "confidence": None,
            "skipped": args.no_verify,
        }

        if not args.no_verify:
            log(f"[5/7] Candidate {index}: Qwen crop completeness validation")
            candidate_crop = image.crop(initial_crop.to_int_tuple())
            validation_context = build_validation_context(
                image,
                roi_bbox,
                marker_box,
                initial_crop,
            )
            validation = validate_crop(
                client,
                args.qwen_model,
                validation_context,
                candidate_crop,
                raw_dir / f"candidate_{index:03d}_validation.txt",
            )
            if not validation.get("valid", False):
                first_validation = validation
                expansion = expansion_from_validation(initial_crop, validation)
                final_crop = initial_crop.expand(*expansion).clamp(
                    image.width,
                    image.height,
                )
                log(
                    f"[5/7] Candidate {index}: expanded crop "
                    f"{initial_crop.to_list()} -> {final_crop.to_list()}"
                )
                retry_context = build_validation_context(
                    image,
                    roi_bbox,
                    marker_box,
                    final_crop,
                )
                retry_crop = image.crop(final_crop.to_int_tuple())
                validation = validate_crop(
                    client,
                    args.qwen_model,
                    retry_context,
                    retry_crop,
                    raw_dir / f"candidate_{index:03d}_validation_retry.txt",
                )
                validation["first_pass"] = first_validation

        fai_name = safe_fai_name(mapping.get("fai_number"))
        crop_path = crop_dir / f"FAI{fai_name}_{index:03d}.png"
        image.crop(final_crop.to_int_tuple()).save(crop_path)
        results.append(
            CandidateResult(
                marker=marker,
                roi_bbox=roi_bbox,
                primitives=primitives,
                mapping=mapping,
                initial_crop=initial_crop,
                final_crop=final_crop,
                validation=validation,
                image_path=str(crop_path),
            )
        )
        log(f"[6/7] Candidate {index}: saved {crop_path}")

    # Remove near-identical duplicate crops while preserving marker-level records.
    deduplicated: list[CandidateResult] = []
    for result in results:
        if any(
            result.final_crop.iou(existing.final_crop) >= 0.90
            and result.mapping.get("fai_number") == existing.mapping.get("fai_number")
            for existing in deduplicated
        ):
            continue
        deduplicated.append(result)

    manifest = {
        "source": str(input_path),
        "source_size": [image.width, image.height],
        "coordinate_space": "absolute_pixels",
        "models": {
            "locate": args.locate_model,
            "semantic_and_validation": args.qwen_model,
            "endpoint": args.endpoint,
        },
        "pipeline": {
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "circle_pair_validation_enabled": True,
            "tesseract_enabled": not args.no_tesseract,
            "verification_enabled": not args.no_verify,
        },
        "results": [result_record(result) for result in deduplicated],
    }
    (output_dir / "results.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.debug:
        annotate_overview(image, deduplicated, debug_dir / "overview.png")
    log(f"[7/7] Complete. Results: {output_dir / 'results.json'}")
    return deduplicated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect FAI annotation groups in a PNG engineering drawing and "
            "crop each complete group."
        )
    )
    parser.add_argument("image", help="Path to the input PNG")
    parser.add_argument(
        "-o",
        "--output",
        default="output_v3",
        help="Output directory (default: output_v3)",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LOCAL_VLM_API_KEY", DEFAULT_API_KEY),
    )
    parser.add_argument("--locate-model", default=DEFAULT_LOCATE_MODEL)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1200,
        help="Maximum LocateAnything tile width/height",
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.20,
        help="Fractional overlap between neighboring tiles",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=50,
        help="Maximum number of FAI candidates to process",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request model timeout in seconds",
    )
    parser.add_argument(
        "--no-tesseract",
        action="store_true",
        help="Disable local Tesseract OCR evidence",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the final Qwen crop-completeness check",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save tiles, clean ROIs, evidence overlays, and overview",
    )
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
    try:
        process_image(args)
    except Exception as exc:
        log(f"ERROR: {exc}")
        if args.debug:
            raise
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
