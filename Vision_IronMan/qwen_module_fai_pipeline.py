#!/usr/bin/env python3
"""Streaming dual-model pipeline with single-agent crop recovery."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable

from openai import OpenAI
from PIL import Image, ImageDraw

import qwen_fai_cluster_detector as fai
from fai_xray import XRAY_PROMPT_APPENDIX, build_fai_xray, save_xray_result


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DETECTOR_PATH = SCRIPT_DIR / "qwen_module_blind_test V2.py"
DEFAULT_ENDPOINT = "http://127.0.0.1:8001/v1"
DEFAULT_MODEL = "Qwen3.8-27B-MLX-8bit"
DEFAULT_RECOVERY_MODEL = "Qwen3.8-27B-MLX-8bit"
DEFAULT_API_KEY = "anything"

RECOVERY_ACTIONS = {"finish", "expand"}
RECOVERY_SIDES = {"left", "right", "up", "down"}
RECOVERY_REASONS = {
    "complete",
    "target_clipped",
    "uncertain",
}
RECOVERY_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["finish", "expand"]},
        "expand_sides": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["left", "right", "up", "down"],
            },
            "uniqueItems": True,
        },
        "reason": {
            "type": "string",
            "enum": ["complete", "target_clipped", "uncertain"],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["action", "expand_sides", "reason", "confidence"],
    "additionalProperties": False,
}
RECOVERY_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "crop_recovery_decision",
        "strict": True,
        "schema": RECOVERY_DECISION_SCHEMA,
    },
}
RECOVERY_SYSTEM_PROMPT = """You are a crop recovery decision agent.

You receive exactly one image: a wider view made from the original drawing. The
MAGENTA rectangle marks the exact current crop. The RED ellipse marks the one
selected FAI circle and its inspection number. The area outside the magenta
rectangle is additional context. Do not re-detect the candidate.

Your only task is to trace the annotation associated with the RED FAI ellipse to
its physical target geometry, then judge whether that physical geometry is
sufficiently complete inside the MAGENTA rectangle. Annotation geometry is not
physical target geometry.

First identify which general annotation topology is present:

1. Direct leader or callout:
   Follow the leader to its terminal arrowhead. The arrowhead directly contacts
   the physical target. Starting at that contact point, trace the connected
   physical contour or structure.

2. Linear, angular, or distance dimension:
   Dimension arrowheads normally terminate on extension lines, witness lines, or
   angular rays rather than on the physical part itself. Do not treat those lines
   as the target. Follow each corresponding extension line or ray toward the
   measured surfaces until the physical target geometry is reached. If the line
   leaves the visible context before reaching a physical surface, the target has
   not yet been found and finish is forbidden.

3. Diameter or radius dimension, identified by symbols such as Ø, ⌀, or R:
   Trace the dimension or leader to the circular, cylindrical, hole, or arc
   feature being measured. The text, dimension line, centerline, extension line,
   and arrowhead are not the target. A short arc fragment or only one side of a
   diameter is insufficient when the relevant circular or cylindrical feature
   continues beyond the crop.

Only after the physical target has been found may you assess completeness. A
recognizable target is not necessarily a complete target. The arrow contact
point, measured surface, hole edge, short arc, corner, or small hatched patch is
only a target seed. Starting from that seed, trace the connected physical contour
or structural unit to a closed contour, visible component boundary, structural
separation, or another natural endpoint.

In a sectional view, hatching represents physical material. When the target seed
belongs to a hatched region, follow the region and its enclosing outline to the
natural boundary of the same structural unit. A small visible hatch fragment is
never sufficient when the same structure continues beyond the current crop.

Rules:
- Before returning finish, inspect the LEFT, RIGHT, TOP, and BOTTOM edges of the
  MAGENTA rectangle. Finish is allowed only when every associated physical target
  reaches its visible natural boundaries inside the rectangle.
- If any target-related physical contour, surface, wall, hatch region, or
  connected structure touches or crosses a MAGENTA edge and continues in the
  wider context, finish is forbidden. Return every such edge in expand_sides.
- Follow only leaders and arrows belonging to the RED FAI marker. Ignore every
  other FAI/SPC circle, dimension, arrow, and target visible in the context.
- Never return finish when no physical target geometry associated with the RED
  FAI is visible inside the MAGENTA rectangle. FAI/SPC markers, text, feature
  control frames, dimension lines, extension or witness lines, centerlines,
  leaders, and arrowheads do not count as physical target geometry.
- When the annotation path exits the MAGENTA rectangle before reaching its
  physical target, expand the side crossed by that path. When it exits the wider
  observation as well, use its exit side as the required expansion direction.
- Use geometry outside the MAGENTA rectangle as direct evidence. If it shows
  continuation of the target inside the rectangle, expand the crossed side even
  when the arrow contact point and a recognizable target fragment are inside.
- expand_sides names every side of the MAGENTA rectangle where relevant target
  geometry is missing. It is not the visual pointing direction of an arrowhead.
- annotation elements need not remain inside the final crop after their
  association has been traced; final completeness is judged on physical targets.
- if target geometry is clipped on multiple sides, return all of those sides in
  the same response, for example ["left", "down"]. Do not choose only one side.
- Python expands each requested side by the percentage stated in the user
  message. Request all necessary sides in the same decision.
- uncertainty is not evidence of completeness. Do not hide uncertainty with finish.

Allowed action values: finish, expand.
Allowed expand_sides values: left, right, up, down. Values must be unique.
Allowed reason values: complete, target_clipped, uncertain.

Return exactly one JSON object and no Markdown or analysis.
Expansion example:
{"action":"expand","expand_sides":["left","down"],"reason":"target_clipped","confidence":0.93}
Finish example:
{"action":"finish","expand_sides":[],"reason":"complete","confidence":0.93}
"""


@dataclass(frozen=True)
class CropBox:
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

    def clamp(self, width: int, height: int) -> "CropBox":
        x1 = max(0, min(width, self.x1))
        y1 = max(0, min(height, self.y1))
        x2 = max(0, min(width, self.x2))
        y2 = max(0, min(height, self.y2))
        return CropBox(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class RecoveryCandidate:
    key: str
    module_index: int
    cluster_index: int
    fai_number: str | None
    source_crop_path: Path
    initial_box: CropBox
    marker_box: CropBox | None = None


@dataclass(frozen=True)
class RecoveryDecision:
    action: str | None
    expand_sides: list[str]
    reason: str | None
    confidence: float | None
    elapsed_seconds: float
    raw_response: str
    metadata: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def log(message: str) -> None:
    print(message, flush=True)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def local_box_to_global(
    module_box: Iterable[int | float], cluster_box: Iterable[int | float]
) -> CropBox:
    """Translate a Stage 2 module-local cluster box into full-image pixels."""
    mx1, my1, _, _ = [round(float(value)) for value in module_box]
    cx1, cy1, cx2, cy2 = [round(float(value)) for value in cluster_box]
    return CropBox(mx1 + cx1, my1 + cy1, mx1 + cx2, my1 + cy2)


def fit_observation_image(image: Image.Image, max_edge: int) -> Image.Image:
    longest = max(image.size)
    if longest <= max_edge:
        return image
    scale = max_edge / float(longest)
    size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    return image.resize(size, Image.Resampling.LANCZOS)


def build_recovery_observation(
    full_image: Image.Image,
    current: CropBox,
    context_fraction: float,
    max_image_edge: int,
    marker_box: CropBox | None = None,
) -> tuple[Image.Image, CropBox]:
    """Expand current crop on four sides and mark its box in one context image."""
    current = current.clamp(full_image.width, full_image.height)
    if current.width <= 0 or current.height <= 0:
        raise ValueError("Recovery crop must have positive width and height")
    fraction = max(0.05, min(1.0, context_fraction))
    pad_x = max(1, round(current.width * fraction))
    pad_y = max(1, round(current.height * fraction))
    context_box = CropBox(
        current.x1 - pad_x,
        current.y1 - pad_y,
        current.x2 + pad_x,
        current.y2 + pad_y,
    ).clamp(full_image.width, full_image.height)
    observation = full_image.crop(tuple(context_box.to_list())).convert("RGB")
    local_current = CropBox(
        current.x1 - context_box.x1,
        current.y1 - context_box.y1,
        current.x2 - context_box.x1,
        current.y2 - context_box.y1,
    )
    draw = ImageDraw.Draw(observation)
    stroke = max(2, round(min(observation.size) * 0.004))
    draw.rectangle(
        tuple(local_current.to_list()), outline=(220, 0, 220), width=stroke
    )
    if marker_box is not None:
        marker = marker_box.clamp(full_image.width, full_image.height)
        local_marker = CropBox(
            marker.x1 - context_box.x1,
            marker.y1 - context_box.y1,
            marker.x2 - context_box.x1,
            marker.y2 - context_box.y1,
        )
        draw.ellipse(
            tuple(local_marker.to_list()), outline=(220, 0, 0), width=stroke
        )
    return fit_observation_image(observation, max_image_edge), context_box


def build_refined_crop(
    full_image: Image.Image,
    crop_box: CropBox,
    marker_box: CropBox | None,
) -> Image.Image:
    """Crop from the full image and draw a red rectangle around the target FAI."""
    crop_box = crop_box.clamp(full_image.width, full_image.height)
    refined = full_image.crop(tuple(crop_box.to_list())).convert("RGB")
    if marker_box is None:
        return refined

    marker = marker_box.clamp(full_image.width, full_image.height)
    pad = max(3, round(max(marker.width, marker.height) * 0.08))
    left = max(0, marker.x1 - crop_box.x1 - pad)
    top = max(0, marker.y1 - crop_box.y1 - pad)
    right = min(refined.width - 1, marker.x2 - crop_box.x1 + pad)
    bottom = min(refined.height - 1, marker.y2 - crop_box.y1 + pad)
    if left < right and top < bottom:
        stroke = max(2, round(min(refined.size) * 0.004))
        ImageDraw.Draw(refined).rectangle(
            (left, top, right, bottom), outline=(220, 0, 0), width=stroke
        )
    return refined


def _strip_response_wrappers(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_recovery_decision(
    raw_response: str,
    elapsed_seconds: float,
    metadata: dict[str, Any],
) -> RecoveryDecision:
    """Strictly validate one recovery decision."""
    try:
        value = json.loads(_strip_response_wrappers(raw_response))
        if not isinstance(value, dict) or set(value) != {
            "action",
            "expand_sides",
            "reason",
            "confidence",
        }:
            raise ValueError(
                "response must contain exactly action, expand_sides, reason, confidence"
            )
        action = value["action"]
        expand_sides = value["expand_sides"]
        reason = value["reason"]
        confidence = value["confidence"]
        if not isinstance(action, str) or action not in RECOVERY_ACTIONS:
            raise ValueError("invalid action")
        if (
            not isinstance(expand_sides, list)
            or not all(isinstance(side, str) and side in RECOVERY_SIDES for side in expand_sides)
            or len(expand_sides) != len(set(expand_sides))
        ):
            raise ValueError("expand_sides must contain unique valid sides")
        if not isinstance(reason, str) or reason not in RECOVERY_REASONS:
            raise ValueError("invalid reason")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ValueError("confidence must be within [0, 1]")
        if action == "finish" and reason != "complete":
            raise ValueError("finish requires reason=complete")
        if action == "finish" and expand_sides:
            raise ValueError("finish requires empty expand_sides")
        if action == "expand" and not expand_sides:
            raise ValueError("expand requires at least one side")
        if action == "expand" and reason == "complete":
            raise ValueError("expansion cannot use reason=complete")
        return RecoveryDecision(
            action=action,
            expand_sides=expand_sides,
            reason=reason,
            confidence=float(confidence),
            elapsed_seconds=elapsed_seconds,
            raw_response=raw_response,
            metadata=metadata,
        )
    except Exception as exc:
        return RecoveryDecision(
            action=None,
            expand_sides=[],
            reason=None,
            confidence=None,
            elapsed_seconds=elapsed_seconds,
            raw_response=raw_response,
            metadata=metadata,
            error=f"{type(exc).__name__}: {exc}",
        )


def call_recovery_agent(
    client: OpenAI,
    model: str,
    user_prompt: str,
    image_urls: list[str],
    max_tokens: int,
    temperature: float,
) -> RecoveryDecision:
    started = time.perf_counter()
    try:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": url}}
            for url in image_urls
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": RECOVERY_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=RECOVERY_RESPONSE_FORMAT,
        )
        raw = (response.choices[0].message.content or "").strip()
        elapsed = time.perf_counter() - started
        return parse_recovery_decision(raw, elapsed, response_metadata(response))
    except Exception as exc:
        return RecoveryDecision(
            action=None,
            expand_sides=[],
            reason=None,
            confidence=None,
            elapsed_seconds=time.perf_counter() - started,
            raw_response="",
            metadata={},
            error=f"{type(exc).__name__}: {exc}",
        )


def request_recovery_decision(
    client: OpenAI,
    model: str,
    image: Image.Image,
    *,
    round_number: int,
    fai_number: str | None = None,
    max_tokens: int,
    temperature: float,
    step_norm: int,
) -> RecoveryDecision:
    """Ask one recovery agent which crop sides must expand together."""
    image_urls = [fai.image_to_data_url(image)]
    step_percent = step_norm / 10.0
    prompt = (
        f"This is recovery round {round_number}. Inspect the one wider-context "
        f"image. The selected FAI number is {fai_number or 'shown in the red ellipse'}. "
        "Trace its annotation to the physical target, then apply the four-edge "
        "physical-contour audit to the MAGENTA current-crop rectangle. A "
        "recognizable fragment or arrow contact point is not a complete target. "
        "If incomplete on multiple borders, include every required border in "
        f"expand_sides. Python will expand each requested side by {step_percent:g} "
        "percent of the current crop width or height. Return exactly one decision JSON."
    )
    return call_recovery_agent(
        client, model, prompt, image_urls, max_tokens, temperature
    )


def expand_crop_sides(
    current: CropBox,
    sides: Iterable[str],
    full_size: tuple[int, int],
    step_norm: int,
) -> CropBox:
    """Expand all requested sides at once; 1000 norm equals current side length."""
    side_set = set(sides)
    if not side_set or not side_set <= RECOVERY_SIDES:
        raise ValueError(f"Invalid expansion sides: {sorted(side_set)!r}")
    horizontal = max(1, round(current.width * step_norm / 1000.0))
    vertical = max(1, round(current.height * step_norm / 1000.0))
    expanded = CropBox(
        current.x1 - horizontal if "left" in side_set else current.x1,
        current.y1 - vertical if "up" in side_set else current.y1,
        current.x2 + horizontal if "right" in side_set else current.x2,
        current.y2 + vertical if "down" in side_set else current.y2,
    )
    return expanded.clamp(*full_size)


def recover_crop_candidate(
    *,
    client: OpenAI,
    model: str,
    full_image: Image.Image,
    candidate: RecoveryCandidate,
    output_dir: Path,
    max_rounds: int,
    max_tokens: int,
    temperature: float,
    context_fraction: float,
    max_image_edge: int,
    step_norm: int,
) -> dict[str, Any]:
    """Run at most max_rounds single-agent decisions for one crop2 candidate."""
    started = time.perf_counter()
    detail_dir = output_dir / "stage3_recovery" / candidate.key
    refined_dir = output_dir / "crop2_refined"
    detail_dir.mkdir(parents=True, exist_ok=True)
    refined_dir.mkdir(parents=True, exist_ok=True)
    current = candidate.initial_box.clamp(full_image.width, full_image.height)
    full_image.crop(tuple(current.to_list())).save(detail_dir / "initial_crop.png")
    rounds: list[dict[str, Any]] = []
    status = "max_rounds_exhausted"

    log(f"[Stage 3][{candidate.key}] recovery started with {model}")
    for round_number in range(1, max_rounds + 1):
        round_dir = detail_dir / f"round_{round_number:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        observation, context_box = build_recovery_observation(
            full_image,
            current,
            context_fraction,
            max_image_edge,
            candidate.marker_box,
        )
        observation.save(round_dir / "observation.png")
        decision = request_recovery_decision(
            client,
            model,
            observation,
            round_number=round_number,
            fai_number=candidate.fai_number,
            max_tokens=max_tokens,
            temperature=temperature,
            step_norm=step_norm,
        )
        (round_dir / "decision_raw.txt").write_text(
            decision.raw_response, encoding="utf-8"
        )
        write_json(round_dir / "decision.json", decision.to_dict())

        before = current
        if decision.action == "finish":
            status = "finished"
        elif decision.action == "expand":
            current = expand_crop_sides(
                current, decision.expand_sides, full_image.size, step_norm
            )
            if current == before:
                status = "boundary_exhausted"

        round_record = {
            "round": round_number,
            "crop_before": before.to_list(),
            "context_bbox": context_box.to_list(),
            "decision": decision.to_dict(),
            "crop_after": current.to_list(),
        }
        rounds.append(round_record)
        write_json(round_dir / "decision_result.json", round_record)
        side_text = "+".join(decision.expand_sides)
        action_text = (
            f"expand({side_text})" if decision.action == "expand" else decision.action
        )
        log(
            f"[Stage 3][{candidate.key}] round {round_number}/{max_rounds}: "
            f"{action_text or 'invalid; retry'}"
        )
        if status in {"finished", "boundary_exhausted"}:
            break

    refined_path = refined_dir / candidate.source_crop_path.name
    build_refined_crop(full_image, current, candidate.marker_box).save(refined_path)
    result = {
        "status": status,
        "valid": status == "finished",
        "candidate_key": candidate.key,
        "module_index": candidate.module_index,
        "cluster_index": candidate.cluster_index,
        "fai_number": candidate.fai_number,
        "model": model,
        "source_crop_path": str(candidate.source_crop_path.relative_to(output_dir)),
        "refined_crop_path": str(refined_path.relative_to(output_dir)),
        "initial_bbox_full_image": candidate.initial_box.to_list(),
        "marker_bbox_full_image": (
            candidate.marker_box.to_list() if candidate.marker_box else None
        ),
        "final_bbox_full_image": current.to_list(),
        "round_count": len(rounds),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "rounds": rounds,
    }
    write_json(detail_dir / "result.json", result)
    log(
        f"[Stage 3][{candidate.key}] {status} after {len(rounds)} round(s) -> "
        f"{refined_path.name}"
    )
    return result


def load_module_detector() -> ModuleType:
    if not MODULE_DETECTOR_PATH.is_file():
        raise FileNotFoundError(f"Module detector not found: {MODULE_DETECTOR_PATH}")
    module_name = "qwen_module_blind_test_v2"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_DETECTOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module detector: {MODULE_DETECTOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def response_metadata(response: Any) -> dict[str, Any]:
    choice = response.choices[0]
    return {
        "finish_reason": choice.finish_reason,
        "usage": response.usage.model_dump() if response.usage is not None else None,
    }


def call_vision_once(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image: Image.Image,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": fai.image_to_data_url(image)},
                    },
                ],
            },
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip(), response_metadata(response)


def save_jpeg(image: Image.Image, path: Path) -> None:
    image.convert("RGB").save(path, format="JPEG", quality=95, subsampling=0)


def stage1_detect_modules(
    *,
    client: OpenAI,
    module_detector: ModuleType,
    model: str,
    image: Image.Image,
    output_dir: Path,
    max_tokens: int,
    temperature: float,
) -> tuple[list[Any], list[Path], dict[str, Any]]:
    stage_dir = output_dir / "stage1_modules"
    crop1_dir = output_dir / "crop1"
    stage_dir.mkdir(parents=True, exist_ok=True)
    crop1_dir.mkdir(parents=True, exist_ok=True)

    (stage_dir / "system_prompt.txt").write_text(
        module_detector.SYSTEM_PROMPT, encoding="utf-8"
    )
    (stage_dir / "user_prompt.txt").write_text(
        module_detector.USER_PROMPT, encoding="utf-8"
    )

    log("[Stage 1/3] Sending the complete large image to 27B for module detection ...")
    started = time.perf_counter()
    raw, metadata = call_vision_once(
        client,
        model,
        module_detector.SYSTEM_PROMPT,
        module_detector.USER_PROMPT,
        image,
        max_tokens,
        temperature,
    )
    elapsed = time.perf_counter() - started
    (stage_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
    write_json(stage_dir / "response_metadata.json", metadata)
    log(
        f"[Stage 1/3] 27B response received in {elapsed:.1f}s "
        f"(finish_reason={metadata.get('finish_reason')})"
    )

    parsed = module_detector.parse_model_json(raw)
    write_json(stage_dir / "parsed_response.json", parsed)
    modules, invalid = module_detector.validate_detections(
        parsed, image.width, image.height
    )
    if invalid:
        log(f"[Stage 1/3] Warning: {len(invalid)} invalid module object(s) were skipped")
    if not modules:
        raise RuntimeError("Stage 1 returned no valid modules")

    crop_paths: list[Path] = []
    serialized: list[dict[str, Any]] = []
    for index, module in enumerate(modules, start=1):
        x1, y1, x2, y2 = module.bbox_pixels
        crop_path = crop1_dir / f"module_{index:03d}.png"
        image.crop((x1, y1, x2, y2)).save(crop_path)
        crop_paths.append(crop_path)
        item = asdict(module)
        item["crop_path"] = str(crop_path.relative_to(output_dir))
        serialized.append(item)

    overview = module_detector.draw_detections(image, modules)
    save_jpeg(overview, output_dir / "visualization.jpg")
    result = {
        "status": "ok" if not invalid else "partial_schema_error",
        "elapsed_seconds": round(elapsed, 3),
        "response_metadata": metadata,
        "valid_module_count": len(modules),
        "invalid_module_count": len(invalid),
        "valid_modules": serialized,
        "invalid_modules": invalid,
        "visualization_path": "visualization.jpg",
    }
    write_json(stage_dir / "results.json", result)
    log(f"[Stage 1/3] Saved {len(modules)} module crop(s) to {crop1_dir}")
    log(f"[Stage 1/3] Large-image visualization: {output_dir / 'visualization.jpg'}")
    return modules, crop_paths, result


def save_stage2_crops(
    image: Image.Image,
    clusters: list[fai.FAICluster],
    crop2_dir: Path,
    module_index: int,
    module_bbox: Iterable[int | float],
    on_crop: Callable[[RecoveryCandidate], None] | None = None,
) -> list[Path]:
    """Persist crop2 images and immediately publish each recovery candidate."""
    paths: list[Path] = []
    for cluster_index, cluster in enumerate(clusters, start=1):
        x1, y1, x2, y2 = cluster.bbox_pixels
        filename = (
            f"module_{module_index:03d}_FAI_"
            f"{fai.safe_name(cluster.fai_number)}_{cluster_index:03d}.png"
        )
        path = crop2_dir / filename
        image.crop((x1, y1, x2, y2)).save(path)
        paths.append(path)
        marker_pixels = getattr(cluster, "marker_bbox_pixels", None)
        if on_crop is not None:
            on_crop(
                RecoveryCandidate(
                    key=path.stem,
                    module_index=module_index,
                    cluster_index=cluster_index,
                    fai_number=cluster.fai_number,
                    source_crop_path=path,
                    initial_box=local_box_to_global(
                        module_bbox, cluster.bbox_pixels
                    ),
                    marker_box=(
                        local_box_to_global(module_bbox, marker_pixels)
                        if marker_pixels is not None
                        else None
                    ),
                )
            )
    return paths


def stage2_detect_fai_for_module(
    *,
    client: OpenAI,
    model: str,
    module_image: Image.Image,
    module_path: Path,
    module_bbox: Iterable[int | float],
    module_index: int,
    module_total: int,
    output_dir: Path,
    max_tokens: int,
    temperature: float,
    on_crop: Callable[[RecoveryCandidate], None] | None = None,
) -> dict[str, Any]:
    name = f"module_{module_index:03d}"
    detail_dir = output_dir / "stage2_fai" / name
    visualization_dir = output_dir / "fai_visualizations"
    crop2_dir = output_dir / "crop2"
    detail_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    crop2_dir.mkdir(parents=True, exist_ok=True)

    (detail_dir / "system_prompt.txt").write_text(fai.SYSTEM_PROMPT, encoding="utf-8")
    (detail_dir / "user_prompt.txt").write_text(fai.USER_PROMPT, encoding="utf-8")
    log(
        f"[Stage 2/3][{module_index}/{module_total}] 27B detecting FAI clusters in "
        f"{module_path.name} ({module_image.width}x{module_image.height}) ..."
    )
    started = time.perf_counter()

    try:
        xray_started = time.perf_counter()
        module_image.save(detail_dir / "module_original.png")
        xray = build_fai_xray(module_image)
        xray_path = detail_dir / "xray.png"
        xray_json_path = detail_dir / "xray.json"
        save_xray_result(xray, xray_path, xray_json_path)
        xray_elapsed = time.perf_counter() - xray_started
        xray_system_prompt = fai.SYSTEM_PROMPT + XRAY_PROMPT_APPENDIX
        (detail_dir / "system_prompt_xray.txt").write_text(
            xray_system_prompt, encoding="utf-8"
        )
        log(
            f"[Stage 2/3][{module_index}/{module_total}] CPU X-ray saved "
            f"({len(xray.clusters)} FAI candidate(s), {xray_elapsed:.2f}s)"
        )
        raw, metadata = call_vision_once(
            client,
            model,
            xray_system_prompt,
            fai.USER_PROMPT,
            xray.image,
            max_tokens,
            temperature,
        )
        elapsed = time.perf_counter() - started
        (detail_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
        write_json(detail_dir / "response_metadata.json", metadata)
        log(
            f"[Stage 2/3][{module_index}/{module_total}] 27B response received in "
            f"{elapsed:.1f}s (finish_reason={metadata.get('finish_reason')})"
        )

        parsed = fai.parse_response(raw)
        write_json(detail_dir / "parsed_response.json", parsed)
        clusters, invalid = fai.validate_clusters(
            parsed, module_image.width, module_image.height
        )
        crop_paths = save_stage2_crops(
            module_image,
            clusters,
            crop2_dir,
            module_index,
            module_bbox,
            on_crop,
        )
        visualization_path = visualization_dir / f"{name}_visualization.jpg"
        save_jpeg(fai.draw_visualization(module_image, clusters), visualization_path)

        serialized: list[dict[str, Any]] = []
        for cluster, crop_path in zip(clusters, crop_paths):
            item = asdict(cluster)
            item["crop_path"] = str(crop_path.relative_to(output_dir))
            item["bbox_full_image"] = local_box_to_global(
                module_bbox, cluster.bbox_pixels
            ).to_list()
            item["recovery_key"] = crop_path.stem
            serialized.append(item)

        result = {
            "status": "ok" if not invalid else "partial_schema_error",
            "module_index": module_index,
            "module_crop_path": str(module_path.relative_to(output_dir)),
            "elapsed_seconds": round(elapsed, 3),
            "response_metadata": metadata,
            "xray_elapsed_seconds": round(xray_elapsed, 3),
            "xray_path": str(xray_path.relative_to(output_dir)),
            "xray_json_path": str(xray_json_path.relative_to(output_dir)),
            "xray_debug_path": str(xray_debug_path.relative_to(output_dir)),
            "xray_used_for_qwen": xray.use_for_qwen,
            "xray_diagnostics": xray.diagnostics,
            "valid_cluster_count": len(clusters),
            "invalid_cluster_count": len(invalid),
            "valid_clusters": serialized,
            "invalid_clusters": invalid,
            "visualization_path": str(visualization_path.relative_to(output_dir)),
        }
        write_json(detail_dir / "results.json", result)
        log(
            f"[Stage 2/3][{module_index}/{module_total}] Found {len(clusters)} valid "
            f"FAI cluster(s); visualization saved"
        )
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - started
        result = {
            "status": "error",
            "module_index": module_index,
            "module_crop_path": str(module_path.relative_to(output_dir)),
            "elapsed_seconds": round(elapsed, 3),
            "error": str(exc),
            "valid_cluster_count": 0,
            "valid_clusters": [],
        }
        write_json(detail_dir / "results.json", result)
        log(
            f"[Stage 2/3][{module_index}/{module_total}] ERROR: {exc}; "
            "continuing with the next module"
        )
        return result


def run(args: argparse.Namespace) -> int:
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    module_detector = load_module_detector()
    with Image.open(image_path) as loaded:
        large_image = loaded.convert("RGB")

    write_json(
        output_dir / "run_config.json",
        {
            "input_image": str(image_path),
            "image_width": large_image.width,
            "image_height": large_image.height,
            "endpoint": args.endpoint,
            "front_model": args.model,
            "xray_mode": "cpu_ocr_opencv",
            "recovery_endpoint": args.recovery_endpoint or args.endpoint,
            "recovery_model": args.recovery_model,
            "module_max_tokens": args.module_max_tokens,
            "fai_max_tokens": args.fai_max_tokens,
            "temperature": args.temperature,
            "recovery_max_rounds": args.recovery_max_rounds,
            "recovery_max_tokens": args.recovery_max_tokens,
            "recovery_response_format": "json_schema",
            "recovery_temperature": args.recovery_temperature,
            "recovery_context_fraction": args.recovery_context_fraction,
            "recovery_max_image_edge": args.recovery_max_image_edge,
            "recovery_step_norm": args.recovery_step_norm,
            "recovery_workers": args.recovery_workers,
            "execution": "streaming dual-model pipeline with single-agent multi-side recovery",
        },
    )

    front_client = OpenAI(
        base_url=args.endpoint,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    recovery_client = OpenAI(
        base_url=args.recovery_endpoint or args.endpoint,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    pipeline_started = time.perf_counter()
    log(f"[Start] Input: {image_path} ({large_image.width}x{large_image.height})")
    modules, module_paths, stage1_result = stage1_detect_modules(
        client=front_client,
        module_detector=module_detector,
        model=args.model,
        image=large_image,
        output_dir=output_dir,
        max_tokens=args.module_max_tokens,
        temperature=args.temperature,
    )

    total = len(module_paths)
    log(f"[Stage 2/3] Starting 27B FAI detection for {total} module(s)")
    stage2_results: list[dict[str, Any]] = []
    recovery_futures: dict[Future[dict[str, Any]], RecoveryCandidate] = {}
    recovery_results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(
        max_workers=args.recovery_workers,
        thread_name_prefix="crop2-recovery",
    ) as recovery_executor:

        def submit_recovery(candidate: RecoveryCandidate) -> None:
            log(
                f"[Stage 3][{candidate.key}] crop2 arrived; submitting immediate "
                "single-agent multi-side recovery"
            )
            future = recovery_executor.submit(
                recover_crop_candidate,
                client=recovery_client,
                model=args.recovery_model,
                full_image=large_image,
                candidate=candidate,
                output_dir=output_dir,
                max_rounds=args.recovery_max_rounds,
                max_tokens=args.recovery_max_tokens,
                temperature=args.recovery_temperature,
                context_fraction=args.recovery_context_fraction,
                max_image_edge=args.recovery_max_image_edge,
                step_norm=args.recovery_step_norm,
            )
            recovery_futures[future] = candidate

        for index, (module, module_path) in enumerate(
            zip(modules, module_paths), start=1
        ):
            with Image.open(module_path) as loaded:
                module_image = loaded.convert("RGB")
            stage2_results.append(
                stage2_detect_fai_for_module(
                    client=front_client,
                    model=args.model,
                    module_image=module_image,
                    module_path=module_path,
                    module_bbox=module.bbox_pixels,
                    module_index=index,
                    module_total=total,
                    output_dir=output_dir,
                    max_tokens=args.fai_max_tokens,
                    temperature=args.temperature,
                    on_crop=submit_recovery,
                )
            )

        log(
            f"[Stage 3/3] Waiting for {len(recovery_futures)} submitted crop2 "
            "recovery job(s)"
        )
        for future in as_completed(recovery_futures):
            candidate = recovery_futures[future]
            try:
                recovery_results.append(future.result())
            except Exception as exc:
                error_result = {
                    "status": "error",
                    "valid": False,
                    "candidate_key": candidate.key,
                    "module_index": candidate.module_index,
                    "cluster_index": candidate.cluster_index,
                    "fai_number": candidate.fai_number,
                    "model": args.recovery_model,
                    "source_crop_path": str(
                        candidate.source_crop_path.relative_to(output_dir)
                    ),
                    "initial_bbox_full_image": candidate.initial_box.to_list(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                recovery_results.append(error_result)
                error_dir = output_dir / "stage3_recovery" / candidate.key
                error_dir.mkdir(parents=True, exist_ok=True)
                write_json(error_dir / "result.json", error_result)
                log(f"[Stage 3][{candidate.key}] ERROR: {exc}")

    recovery_results.sort(
        key=lambda item: (item.get("module_index", 0), item.get("cluster_index", 0))
    )

    elapsed = time.perf_counter() - pipeline_started
    failed = sum(result["status"] == "error" for result in stage2_results)
    total_clusters = sum(result.get("valid_cluster_count", 0) for result in stage2_results)
    recovery_finished = sum(
        result.get("status") == "finished" for result in recovery_results
    )
    recovery_unresolved = len(recovery_results) - recovery_finished
    pipeline_result = {
        "status": "ok" if failed == 0 and recovery_unresolved == 0 else "partial_error",
        "elapsed_seconds": round(elapsed, 3),
        "module_count": len(modules),
        "fai_cluster_count": total_clusters,
        "failed_module_count": failed,
        "recovery_finished_count": recovery_finished,
        "recovery_unresolved_count": recovery_unresolved,
        "stage1": stage1_result,
        "stage2": stage2_results,
        "stage3": {
            "model": args.recovery_model,
            "submitted_crop_count": len(recovery_results),
            "finished_count": recovery_finished,
            "unresolved_count": recovery_unresolved,
            "results": recovery_results,
        },
    }
    write_json(output_dir / "pipeline_results.json", pipeline_result)
    log(
        f"[Complete] {len(modules)} module(s), {total_clusters} FAI cluster(s), "
        f"{recovery_finished} recovered, {recovery_unresolved} unresolved, "
        f"{failed} failed module(s), total {elapsed:.1f}s"
    )
    log(f"[Complete] crop1: {output_dir / 'crop1'}")
    log(f"[Complete] crop2: {output_dir / 'crop2'}")
    log(f"[Complete] refined crop2: {output_dir / 'crop2_refined'}")
    log(f"[Complete] FAI visualizations: {output_dir / 'fai_visualizations'}")
    return 0 if failed == 0 and recovery_unresolved == 0 else 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect modules and FAI clusters with 27B, then stream every crop2 "
            "through single-agent multi-side crop recovery."
        )
    )
    parser.add_argument("image", help="Path to the complete large image")
    parser.add_argument(
        "-o",
        "--output",
        default="output_module_fai_pipeline",
        help="Output directory (default: output_module_fai_pipeline)",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LOCAL_VLM_API_KEY", DEFAULT_API_KEY),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--recovery-model",
        default=DEFAULT_RECOVERY_MODEL,
        help="Model used by the crop-recovery decision agent",
    )
    parser.add_argument(
        "--recovery-endpoint",
        default=None,
        help="Optional recovery endpoint; defaults to --endpoint",
    )
    parser.add_argument("--module-max-tokens", type=int, default=8192)
    parser.add_argument("--fai-max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--recovery-max-rounds", type=int, default=6)
    parser.add_argument("--recovery-max-tokens", type=int, default=2048)
    parser.add_argument("--recovery-temperature", type=float, default=0.1)
    parser.add_argument("--recovery-context-fraction", type=float, default=0.5)
    parser.add_argument("--recovery-max-image-edge", type=int, default=2400)
    parser.add_argument("--recovery-step-norm", type=int, default=250)
    parser.add_argument(
        "--recovery-workers",
        type=int,
        default=1,
        help=(
            "Concurrent crop2 recovery jobs; each round issues one model call "
            "per job (default: 1)"
        ),
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if (
        args.module_max_tokens < 1
        or args.fai_max_tokens < 1
        or args.recovery_max_tokens < 1
    ):
        parser.error("token limits must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.recovery_max_rounds < 1:
        parser.error("--recovery-max-rounds must be positive")
    if not 0.0 <= args.recovery_temperature <= 2.0:
        parser.error("--recovery-temperature must be within [0, 2]")
    if not 0.05 <= args.recovery_context_fraction <= 1.0:
        parser.error("--recovery-context-fraction must be within [0.05, 1]")
    if args.recovery_max_image_edge < 512:
        parser.error("--recovery-max-image-edge must be at least 512")
    if not 1 <= args.recovery_step_norm <= 500:
        parser.error("--recovery-step-norm must be within [1, 500]")
    if args.recovery_workers < 1:
        parser.error("--recovery-workers must be positive")
    try:
        return run(args)
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
