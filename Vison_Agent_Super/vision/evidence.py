"""Build local OCR and geometry evidence for one FAI candidate ROI."""

from __future__ import annotations

import csv
import io
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image, ImageDraw

from .detection import deduplicate_boxes
from .inference import locate_boxes
from .models import BBox, Primitive
from .utils import log


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


def marker_local_bbox(marker_bbox: BBox, roi_bbox: BBox) -> BBox:
    return marker_bbox.translate(-roi_bbox.x1, -roi_bbox.y1)


def make_selection_overlay(
    roi_image: Image.Image,
    selected_marker: BBox,
    arrow_boxes: list[BBox] | None = None,
) -> Image.Image:
    marked = roi_image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    marker = selected_marker.to_int_tuple()
    stroke = max(2, round(min(marked.size) * 0.004))
    draw.rectangle(marker, outline=(255, 0, 0), width=stroke)
    label_y = max(0, marker[1] - 18)
    draw.rectangle(
        (marker[0], label_y, marker[0] + 100, label_y + 16), fill=(255, 255, 255)
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
    command = [executable, "stdin", "stdout", "--psm", "11", "-l", "eng", "tsv"]
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
    reader = csv.DictReader(
        io.StringIO(completed.stdout.decode("utf-8", errors="replace")), delimiter="\t"
    )
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in reader:
        text = (row.get("text") or "").strip()
        try:
            confidence = float(row.get("conf") or -1)
        except ValueError:
            confidence = -1
        if not text or confidence < 5:
            continue
        key = tuple(
            row.get(name, "")
            for name in ("page_num", "block_num", "par_num", "line_num")
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


def detect_line_segments(
    image: Image.Image,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 140, apertureSize=3)
    diagonal = math.hypot(image.width, image.height)
    min_length = max(18, int(diagonal * 0.025))
    detected = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(15, int(min_length * 0.55)),
        minLineLength=min_length,
        maxLineGap=max(3, int(diagonal * 0.006)),
    )
    if detected is None:
        return []
    unique: list[tuple[tuple[int, int], tuple[int, int]]] = []
    signatures: set[tuple[int, int, int, int]] = set()
    for raw in detected[:, 0]:
        x1, y1, x2, y2 = map(int, raw)
        if (x2, y2) < (x1, y1):
            x1, y1, x2, y2 = x2, y2, x1, y1
        signature = (round(x1 / 4), round(y1 / 4), round(x2 / 4), round(y2 / 4))
        if signature in signatures:
            continue
        signatures.add(signature)
        unique.append(((x1, y1), (x2, y2)))
    return sorted(
        unique,
        key=lambda line: math.hypot(
            line[1][0] - line[0][0], line[1][1] - line[0][1]
        ),
        reverse=True,
    )


def point_to_bbox_distance(point: tuple[int, int], bbox: BBox) -> float:
    x, y = point
    box = bbox.ordered()
    return math.hypot(max(box.x1 - x, 0.0, x - box.x2), max(box.y1 - y, 0.0, y - box.y2))


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
        anchor_union.expand(*(diagonal * 0.06 for _ in range(4)))
        if anchor_union
        else BBox(0, 0, image.width, image.height)
    )
    scored: list[tuple[float, tuple[tuple[int, int], tuple[int, int]]]] = []
    for line in lines:
        first, second = line
        midpoint = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
        length = math.dist(first, second)
        min_distance = (
            min(
                (
                    point_to_bbox_distance(first, anchor),
                    point_to_bbox_distance(second, anchor),
                    point_to_bbox_distance(
                        (int(midpoint[0]), int(midpoint[1])), anchor
                    ),
                )
                for anchor in anchors
            )
            if anchors
            else (0.0, 0.0, 0.0)
        )
        distance = min(min_distance)
        if distance <= margin or corridor.contains_point(*midpoint):
            scored.append((length - distance * 0.5, line))
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
        x, y, width, height = cv2.boundingRect(contour)
        if min(width, height) < 3 or max(width, height) > max(80, min(image.size) * 0.12):
            continue
        bbox = BBox(x, y, x + width, y + height)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
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
        candidates.append(
            (solidity * 100.0 - min(anchor_distance, endpoint_distance) * 0.2, bbox)
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return deduplicate_boxes([bbox for _, bbox in candidates[:limit]])


def build_evidence_overlay(
    raw_roi: Image.Image, primitives: list[Primitive]
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
        text_bbox = draw.textbbox(anchor, primitive.id)
        draw.rectangle(
            (text_bbox[0] - 1, text_bbox[1] - 1, text_bbox[2] + 1, text_bbox[3] + 1),
            fill=(255, 255, 255),
        )
        draw.text(anchor, primitive.id, fill=color)
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
    anchors = annotation_boxes + arrow_boxes + target_boxes + ocr_boxes + [marker_local]
    lines = filter_relevant_lines(all_lines, anchors, roi)
    for index, (start, end) in enumerate(lines):
        primitives.append(
            Primitive(
                f"L{index}",
                "leader_segment",
                BBox(
                    min(start[0], end[0]),
                    min(start[1], end[1]),
                    max(start[0], end[0]) + 1,
                    max(start[1], end[1]) + 1,
                ),
                "OpenCV-HoughLinesP",
                points=[start, end],
            )
        )
    primitives.extend(
        Primitive(f"G{index}", "triangle", box, "OpenCV-contour")
        for index, box in enumerate(detect_triangle_candidates(roi, lines, arrow_boxes))
    )
    return roi, primitives, build_evidence_overlay(roi, primitives)
