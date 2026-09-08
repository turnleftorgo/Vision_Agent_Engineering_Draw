"""FAI marker proposal and validation for the standalone Super pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image

from .inference import call_vision_model, extract_json, locate_boxes, normalized_box_to_pixels
from .models import BBox
from .utils import log


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
    image: Image.Image, tile_size: int, overlap: float
) -> Iterable[tuple[int, int, Image.Image]]:
    for y in axis_starts(image.height, tile_size, overlap):
        for x in axis_starts(image.width, tile_size, overlap):
            yield x, y, image.crop(
                (x, y, min(image.width, x + tile_size), min(image.height, y + tile_size))
            )


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
                min(max(box.width, box.height), max(existing.width, existing.height))
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
    for tile_index, (offset_x, offset_y, tile) in enumerate(
        iter_tiles(image, tile_size, overlap)
    ):
        log(
            f"[1/8] LocateAnything + OpenCV FAI scan tile {tile_index}: "
            f"origin=({offset_x},{offset_y}), size={tile.width}x{tile.height}"
        )
        if tile_debug_dir:
            tile_debug_dir.mkdir(parents=True, exist_ok=True)
            tile.save(tile_debug_dir / f"tile_{tile_index:03d}.png")
        locate_local_boxes = locate_boxes(
            client,
            model,
            tile,
            FAI_PROMPT,
            raw_dir / f"fai_tile_{tile_index:03d}.txt",
        )
        circle_pairs = detect_circle_pair_candidates(tile)
        opencv_local_boxes = circle_pair_marker_boxes(circle_pairs)
        local_boxes = locate_local_boxes + opencv_local_boxes
        all_boxes.extend(box.translate(offset_x, offset_y) for box in local_boxes)
    return deduplicate_boxes(all_boxes)


def qwen_fai_fallback(
    client: OpenAI, model: str, image: Image.Image, raw_output_path: Path
) -> list[BBox]:
    prompt = """Find all FAI inspection markers in this engineering drawing.

An FAI marker is a circle containing the literal text FAI and an identification
number. Do not return SPC circles, hole circles, datum circles, section labels,
or ordinary circled numbers.

Return JSON only. Coordinates are integers normalized to [0,1000]:
{"boxes": [{"bbox": [x1,y1,x2,y2], "number": null, "confidence": 0.0}]}
"""
    response = call_vision_model(
        client, model, prompt, [image], max_tokens=2048, temperature=0.1
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
    gray: np.ndarray, cx: int, cy: int, radius: int
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
    detected = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(12, int(minimum_dimension * 0.025)),
        param1=80,
        param2=30,
        minRadius=max(6, int(minimum_dimension * 0.012)),
        maxRadius=max(15, min(80, int(minimum_dimension * 0.05))),
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
            first_score = horizontal_divider_score(gray, *first)
            second_score = horizontal_divider_score(gray, *second)
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
            pairs.append(
                {
                    "left_bbox": left_bbox,
                    "right_bbox": right_bbox,
                    "context_bbox": pair_union.expand(
                        padding, padding, padding, padding
                    ).clamp(image.width, image.height),
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


def circle_pair_marker_boxes(pairs: list[dict[str, Any]]) -> list[BBox]:
    """Promote the conventional left-hand FAI circle from each detected pair.

    These remain high-recall proposals.  The per-candidate semantic stage owns
    the later decision about whether each proposed marker is a valid FAI.
    """
    return deduplicate_boxes(
        [pair["left_bbox"] for pair in pairs if isinstance(pair.get("left_bbox"), BBox)]
    )


def candidate_roi(marker_bbox: BBox, width: int, height: int) -> BBox:
    marker = marker_bbox.ordered()
    cx, cy = marker.center
    half_width = min(1200.0, max(360.0, marker.width * 10.0, width * 0.35))
    half_height = min(1200.0, max(280.0, marker.height * 7.0, height * 0.32))
    return BBox(
        cx - half_width, cy - half_height, cx + half_width, cy + half_height
    ).clamp(width, height)
