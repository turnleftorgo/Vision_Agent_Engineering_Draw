"""CPU-only structural X-ray evidence for Stage 2 FAI cluster detection.

This module intentionally performs no model/API calls. Tesseract and OpenCV
produce visible, auditable F/A/T/L/H/G/R hints for the Stage 2 Qwen request.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


Point = tuple[int, int]
Line = tuple[Point, Point]

CLUSTER_COLORS: tuple[tuple[int, int, int], ...] = (
    (220, 30, 30),
    (25, 105, 220),
    (0, 155, 90),
    (210, 105, 0),
    (145, 55, 200),
    (0, 155, 175),
    (205, 45, 125),
    (100, 120, 0),
)


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def expand(self, amount: float, width: int, height: int) -> "Box":
        return Box(
            max(0, round(self.x1 - amount)),
            max(0, round(self.y1 - amount)),
            min(width, round(self.x2 + amount)),
            min(height, round(self.y2 + amount)),
        )

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class OCRLine:
    box: Box
    text: str
    confidence: float


@dataclass
class XRayCluster:
    cluster_id: str
    fai_number: str | None
    color: tuple[int, int, int]
    marker: Box
    annotation: list[OCRLine]
    text: list[OCRLine]
    leaders: list[Line]
    arrowheads: list[Box]
    geometry_seeds: list[Box]
    target_regions: list[Box]
    confidence: float

    def record(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "fai_number_hint": self.fai_number,
            "color_rgb": list(self.color),
            "F": [self.marker.to_list()],
            "A": [item.box.to_list() for item in self.annotation],
            "T": [
                {
                    "bbox": item.box.to_list(),
                    "text": item.text,
                    "confidence": round(item.confidence, 3),
                }
                for item in self.text
            ],
            "L": [[list(first), list(second)] for first, second in self.leaders],
            "H": [item.to_list() for item in self.arrowheads],
            "G": [item.to_list() for item in self.geometry_seeds],
            "R": [item.to_list() for item in self.target_regions],
            "missing": [
                name
                for name, values in (
                    ("A", self.annotation),
                    ("T", self.text),
                    ("L", self.leaders),
                    ("H", self.arrowheads),
                    ("G", self.geometry_seeds),
                    ("R", self.target_regions),
                )
                if not values
            ],
            "confidence": round(self.confidence, 3),
        }


@dataclass(frozen=True)
class XRayResult:
    image: Image.Image
    clusters: list[XRayCluster]
    diagnostics: dict[str, object]
    debug_image: Image.Image | None = None
    use_for_qwen: bool = True

    @property
    def records(self) -> list[dict[str, object]]:
        return [cluster.record() for cluster in self.clusters]


def _run_tesseract(image: Image.Image) -> list[OCRLine]:
    executable = shutil.which("tesseract")
    if not executable:
        return []
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    completed = subprocess.run(
        [executable, "stdin", "stdout", "--psm", "11", "tsv"],
        input=buffer.getvalue(),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        return []
    rows = csv.DictReader(
        io.StringIO(completed.stdout.decode("utf-8", errors="replace")),
        delimiter="\t",
    )
    values: list[OCRLine] = []
    for row in rows:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            confidence = float(row.get("conf") or -1) / 100.0
            left = int(row.get("left") or 0)
            top = int(row.get("top") or 0)
            width = int(row.get("width") or 0)
            height = int(row.get("height") or 0)
        except ValueError:
            continue
        if confidence >= 0 and width > 1 and height > 1:
            values.append(OCRLine(Box(left, top, left + width, top + height), text, confidence))
    return values


def _normalise_text(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper().replace("1", "I"))


def _box_gap(first: Box, second: Box) -> float:
    dx = max(first.x1 - second.x2, second.x1 - first.x2, 0)
    dy = max(first.y1 - second.y2, second.y1 - first.y2, 0)
    return math.hypot(dx, dy)


def _point_box_gap(point: Point, box: Box) -> float:
    x, y = point
    dx = max(box.x1 - x, x - box.x2, 0)
    dy = max(box.y1 - y, y - box.y2, 0)
    return math.hypot(dx, dy)


def _marker_candidates(
    lines: list[OCRLine], image: Image.Image
) -> list[tuple[Box, str | None, float]]:
    diagonal = math.hypot(image.width, image.height)
    results: list[tuple[Box, str | None, float]] = []
    for line in lines:
        if "FAI" not in _normalise_text(line.text):
            continue
        nearby = sorted(
            (other for other in lines if other is not line),
            key=lambda other: _box_gap(line.box, other.box),
        )
        number: str | None = None
        boxes = [line.box]
        for other in nearby[:8]:
            if _box_gap(line.box, other.box) > max(20.0, diagonal * 0.012):
                break
            matches = re.findall(r"\d{2,5}", other.text)
            if matches:
                number = matches[0]
                boxes.append(other.box)
                break
        x1 = min(box.x1 for box in boxes)
        y1 = min(box.y1 for box in boxes)
        x2 = max(box.x2 for box in boxes)
        y2 = max(box.y2 for box in boxes)
        pad = max(4, round(max(x2 - x1, y2 - y1) * 0.25))
        results.append((Box(x1, y1, x2, y2).expand(pad, image.width, image.height), number, line.confidence))
    return results


def _detect_fai_bubbles(
    image: Image.Image, lines: list[OCRLine]
) -> list[tuple[Box, str | None, float]]:
    """Use circular contours only to refine OCR-backed FAI candidates."""
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    fai_words = [line for line in lines if "FAI" in _normalise_text(line.text)]
    results: list[tuple[Box, str | None, float]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < 14 or height < 14 or not 0.55 <= width / max(1, height) <= 1.8:
            continue
        box = Box(x, y, x + width, y + height)
        matching = [line for line in fai_words if _box_gap(box, line.box) <= max(width, height) * 0.4]
        if not matching:
            continue
        numbers = [
            match
            for line in lines
            if _box_gap(box, line.box) <= max(width, height) * 0.45
            for match in re.findall(r"\d{2,5}", line.text)
        ]
        results.append((box, numbers[0] if numbers else None, 0.72))
    return results


def _merge_marker_candidates(
    first: list[tuple[Box, str | None, float]],
    second: list[tuple[Box, str | None, float]],
) -> list[tuple[Box, str | None, float]]:
    kept: list[tuple[Box, str | None, float]] = []
    for candidate in sorted(first + second, key=lambda item: item[2], reverse=True):
        box = candidate[0]
        if any(
            math.dist(box.center, existing[0].center)
            <= max(8.0, min(max(box.width, box.height), max(existing[0].width, existing[0].height)))
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return sorted(kept, key=lambda item: (item[0].y1, item[0].x1))


def _detect_lines(image: Image.Image) -> list[Line]:
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 70, 170)
    minimum = max(12, round(min(image.size) * 0.012))
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=minimum, maxLineGap=10)
    if raw is None:
        return []
    values = [((int(x1), int(y1)), (int(x2), int(y2))) for x1, y1, x2, y2 in raw[:, 0]]
    return sorted(values, key=lambda line: -math.dist(line[0], line[1]))[:500]


def _detect_annotation_frames(image: Image.Image) -> list[Box]:
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    results: list[Box] = []
    area = image.width * image.height
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if 20 <= width and 8 <= height and width > height * 1.4 and width * height <= area * 0.08:
            results.append(Box(x, y, x + width, y + height))
    return results[:300]


def _detect_triangles(image: Image.Image, maximum: int = 160) -> list[Box]:
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 70, 170)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    results: list[Box] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.08 * perimeter, True)
        x, y, width, height = cv2.boundingRect(contour)
        if 3 <= len(polygon) <= 4 and 5 <= width <= 60 and 5 <= height <= 60:
            results.append(Box(x, y, x + width, y + height))
    return results[:maximum]


def _choose_text(lines: list[OCRLine], marker: Box, diagonal: float) -> list[OCRLine]:
    candidates = [
        line
        for line in lines
        if _box_gap(marker, line.box) <= diagonal * 0.14
        and "FAI" not in _normalise_text(line.text)
        and "SPC" not in _normalise_text(line.text)
    ]
    candidates.sort(key=lambda line: (_box_gap(marker, line.box), -line.confidence))
    return candidates[:3]


def _annotation_group(items: list[OCRLine]) -> list[OCRLine]:
    if not items:
        return []
    x1 = min(item.box.x1 for item in items)
    y1 = min(item.box.y1 for item in items)
    x2 = max(item.box.x2 for item in items)
    y2 = max(item.box.y2 for item in items)
    return [OCRLine(Box(x1, y1, x2, y2), "[OCR annotation group]", min(item.confidence for item in items))]


def _candidate_leaders(lines: list[Line], anchors: list[Box], diagonal: float) -> list[Line]:
    if not anchors:
        return []
    threshold = max(10.0, min(35.0, diagonal * 0.009))
    selected = [
        line
        for line in lines
        if min((_point_box_gap(point, box) for point in line for box in anchors), default=float("inf")) <= threshold
    ]
    # Add directly connected continuation segments, but cap propagation to avoid
    # coloring an entire mechanical drawing.
    for _ in range(2):
        endpoints = [point for line in selected for point in line]
        additions = [
            line
            for line in lines
            if line not in selected
            and min((math.dist(point, endpoint) for point in line for endpoint in endpoints), default=float("inf")) <= threshold
        ]
        selected.extend(additions[:40])
    return selected[:80]


def _target_region(image: Image.Image, seed: Box) -> Box:
    amount = max(18, round(max(seed.width, seed.height) * 2.5))
    return seed.expand(amount, image.width, image.height)


def _render(image: Image.Image, clusters: list[XRayCluster]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    stroke = max(2, round(min(image.size) * 0.0015))
    for cluster in clusters:
        color = cluster.color
        prefix = cluster.cluster_id
        draw.rectangle(cluster.marker.to_list(), outline=color, width=stroke)
        draw.text((cluster.marker.x1, max(0, cluster.marker.y1 - 12)), f"{prefix}-F", fill=color)
        for item in cluster.annotation:
            draw.rectangle(item.box.to_list(), outline=color, width=stroke)
            draw.text((item.box.x1, item.box.y1), f"{prefix}-A", fill=color)
        for item in cluster.text:
            draw.rectangle(item.box.to_list(), outline=color, width=max(1, stroke - 1))
            draw.text((item.box.x1, item.box.y1), f"{prefix}-T", fill=color)
        for first, second in cluster.leaders:
            draw.line((*first, *second), fill=color, width=stroke)
        for box in cluster.arrowheads:
            draw.rectangle(box.to_list(), outline=color, width=stroke)
            draw.text((box.x1, box.y1), f"{prefix}-H", fill=color)
        for box in cluster.geometry_seeds:
            cx, cy = map(round, box.center)
            draw.line((cx - 5, cy, cx + 5, cy), fill=color, width=stroke)
            draw.line((cx, cy - 5, cx, cy + 5), fill=color, width=stroke)
        for box in cluster.target_regions:
            draw.rectangle(box.to_list(), outline=color, width=stroke)
            draw.text((box.x1, box.y1), f"{prefix}-R", fill=color)
    return canvas


def build_fai_xray(image: Image.Image) -> XRayResult:
    """Build one visible CPU-only X-ray image for a Stage 2 module."""
    image = image.convert("RGB")
    ocr_lines = _run_tesseract(image)
    markers = _merge_marker_candidates(
        _marker_candidates(ocr_lines, image),
        _detect_fai_bubbles(image, ocr_lines),
    )
    detected_lines = _detect_lines(image)
    triangles = _detect_triangles(image)
    annotation_frames = _detect_annotation_frames(image)
    diagonal = math.hypot(image.width, image.height)

    provisional: list[XRayCluster] = []
    for index, (marker, number, marker_confidence) in enumerate(markers):
        text = _choose_text(ocr_lines, marker, diagonal)
        nearby_frames = sorted(
            (
                frame
                for frame in annotation_frames
                if 3 < _box_gap(marker, frame) <= diagonal * 0.13
                and frame.width >= marker.width * 1.4
            ),
            key=lambda frame: _box_gap(marker, frame),
        )
        annotation = (
            [OCRLine(nearby_frames[0], "[OpenCV annotation frame]", 0.7)]
            if nearby_frames
            else _annotation_group(text)
        )
        anchors = [item.box for item in annotation] or [marker]
        leaders = _candidate_leaders(detected_lines, anchors, diagonal)
        endpoints = [point for line in leaders for point in line]
        arrowheads = sorted(
            (
                box
                for box in triangles
                if min((math.dist(box.center, point) for point in endpoints), default=float("inf"))
                <= max(9.0, min(18.0, diagonal * 0.0065))
            ),
            key=lambda box: min((math.dist(box.center, point) for point in endpoints), default=float("inf")),
        )[:6]
        seeds = [box.expand(3, image.width, image.height) for box in arrowheads]
        targets = [_target_region(image, seed) for seed in seeds[:4]]
        evidence_ratio = sum(bool(value) for value in (annotation, text, leaders, arrowheads, seeds, targets)) / 6.0
        provisional.append(
            XRayCluster(
                cluster_id=f"C{index + 1:02d}",
                fai_number=number,
                color=CLUSTER_COLORS[index % len(CLUSTER_COLORS)],
                marker=marker,
                annotation=annotation,
                text=text,
                leaders=leaders,
                arrowheads=arrowheads,
                geometry_seeds=seeds,
                target_regions=targets,
                confidence=min(1.0, marker_confidence * 0.65 + evidence_ratio * 0.35),
            )
        )

    line_owner: dict[Line, tuple[float, int]] = {}
    for cluster_index, cluster in enumerate(provisional):
        anchors = [cluster.marker] + [item.box for item in cluster.annotation]
        for line in cluster.leaders:
            distance = min(
                (_point_box_gap(point, box) for point in line for box in anchors),
                default=float("inf"),
            )
            if line not in line_owner or distance < line_owner[line][0]:
                line_owner[line] = (distance, cluster_index)
    for cluster_index, cluster in enumerate(provisional):
        cluster.leaders = [
            line for line in cluster.leaders if line_owner.get(line, (0, -1))[1] == cluster_index
        ]

    diagnostics: dict[str, object] = {
        "mode": "cpu_ocr_opencv",
        "cpu_only": True,
        "input_size": list(image.size),
        "ocr_line_count": len(ocr_lines),
        "fai_marker_count": len(markers),
        "opencv_line_count": len(detected_lines),
        "opencv_triangle_count": len(triangles),
        "opencv_annotation_frame_count": len(annotation_frames),
        "cluster_count": len(provisional),
        "locate_call_count": 0,
        "quality_gate": "pass" if provisional else "original_without_overlay",
    }
    return XRayResult(
        _render(image, provisional),
        provisional,
        diagnostics,
        use_for_qwen=bool(provisional),
    )


def save_xray_result(
    result: XRayResult,
    image_path: Path,
    json_path: Path,
    debug_image_path: Path | None = None,
) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(image_path)
    if debug_image_path is not None:
        (result.debug_image or result.image).save(debug_image_path)
    json_path.write_text(
        json.dumps(
            {
                "diagnostics": result.diagnostics,
                "use_for_qwen": result.use_for_qwen,
                "clusters": result.records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


XRAY_PROMPT_APPENDIX = """

【CPU结构X光标记】
输入图已经由CPU侧OCR和OpenCV添加候选结构标记。相同颜色及相同Cxx前缀表示
Python认为它们可能属于同一个FAI集群：Cxx-F是FAI marker，A是annotation，
T是OCR文字，L是候选引线，H是箭头候选，G是箭头接触几何种子，R是候选目标
区域。这些是高召回辅助证据，不是最终答案，可能缺失或误检。优先利用连续关系，
但必须结合原始黑色工程图线条自行核验；不要因为某个彩色候选框存在就盲目接受。
彩色Cxx并不是完整FAI清单。必须再次扫描整张原始黑色图纸，任何清晰可见但没有
彩色Cxx标记的FAI圆圈仍必须建立独立输出对象。辅助证据只能帮助关联，不能作为
忽略未着色FAI、缩小bbox或跳过完整目标追踪的理由。
"""
