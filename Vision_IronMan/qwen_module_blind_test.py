#!/usr/bin/env python3
"""Blind, single-pass module detection test for an OpenAI-compatible Qwen VLM.

The script deliberately sends the original full image once and performs no
tiling, OCR, classical-vision proposals, NMS, box merging, or model retries
that alter the answer.  Its output therefore remains useful for measuring the
model's native module-discovery and localization ability.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image, ImageDraw


DEFAULT_ENDPOINT = "http://127.0.0.1:8001/v1"
DEFAULT_MODEL = "Qwen3.8-27B-MLX-8bit"
DEFAULT_API_KEY = "anything"

SYSTEM_PROMPT = """你是一个图像区域分析器。

你的任务是直接观察输入的完整图像，找出其中在视觉结构或语义上相对独立的内容模块，并返回每个模块的边界框。

模块是能够作为一个整体区域被理解的内容单元，而不是单个字符、单条线、单个符号或零散图形。

要求：
1. 必须根据图像内容自行判断模块的数量、位置、形状和含义。
2. 不得假设图像具有固定布局。
3. 不得把相邻但彼此独立的模块合并为一个大区域。
4. 不得对同一个模块输出多个重复边界框。
5. 边界框应尽量紧密包围模块主体。
6. 使用归一化坐标，左上角为 (0, 0)，右下角为 (1000, 1000)。
7. 边界框格式为 [x_min, y_min, x_max, y_max]。
8. 只输出合法 JSON，不要输出分析过程、Markdown 或其他文字。"""

USER_PROMPT = """找出这张图像中的所有独立内容模块，并返回它们的边界框。

输出格式：
{
  "objects": [
    {
      "id": 1,
      "bbox": [x_min, y_min, x_max, y_max],
      "description": "该区域内容的简短描述"
    }
  ]
}"""


@dataclass(frozen=True)
class Detection:
    """One strictly validated model detection in both coordinate systems."""

    id: int | str
    description: str
    bbox_normalized: list[float]
    bbox_pixels: list[int]


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def remove_thinking_and_fences(text: str) -> str:
    """Remove common wrappers without repairing or changing model JSON."""

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_model_json(raw_text: str) -> dict[str, Any]:
    """Parse one JSON object; do not infer fields or repair malformed output."""

    cleaned = remove_thinking_and_fences(raw_text)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("The top-level JSON value must be an object")
    return value


def validate_detections(
    response: dict[str, Any], image_width: int, image_height: int
) -> tuple[list[Detection], list[dict[str, Any]]]:
    """Validate schema and coordinates without filtering valid model boxes."""

    objects = response.get("objects")
    if not isinstance(objects, list):
        raise ValueError('The response must contain an "objects" array')

    valid: list[Detection] = []
    invalid: list[dict[str, Any]] = []

    for index, item in enumerate(objects):
        errors: list[str] = []
        if not isinstance(item, dict):
            invalid.append(
                {"index": index, "value": item, "errors": ["object must be a JSON object"]}
            )
            continue

        bbox = item.get("bbox")
        coordinates: list[float] | None = None
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append("bbox must be an array of four numbers")
        elif any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
            errors.append("bbox coordinates must be numbers")
        else:
            coordinates = [float(value) for value in bbox]
            x1, y1, x2, y2 = coordinates
            if not all(0.0 <= value <= 1000.0 for value in coordinates):
                errors.append("bbox coordinates must be within [0, 1000]")
            if x1 >= x2 or y1 >= y2:
                errors.append("bbox must satisfy x_min < x_max and y_min < y_max")

        description = item.get("description", "")
        if not isinstance(description, str):
            errors.append("description must be a string")

        detection_id = item.get("id", index + 1)
        if isinstance(detection_id, (dict, list)) or detection_id is None:
            errors.append("id must be a scalar value")

        if errors:
            invalid.append({"index": index, "value": item, "errors": errors})
            continue

        assert coordinates is not None
        x1, y1, x2, y2 = coordinates
        pixels = [
            round(x1 / 1000.0 * image_width),
            round(y1 / 1000.0 * image_height),
            round(x2 / 1000.0 * image_width),
            round(y2 / 1000.0 * image_height),
        ]
        valid.append(
            Detection(
                id=detection_id,
                description=description,
                bbox_normalized=coordinates,
                bbox_pixels=pixels,
            )
        )

    return valid, invalid


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    """Draw only validated boxes; never modify or merge their geometry."""

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, round(max(canvas.size) / 700))
    palette = [
        "#ff00d4",
        "#00a651",
        "#0066ff",
        "#ff7a00",
        "#7b2cff",
        "#00a6a6",
        "#e31a1c",
        "#5b5b5b",
    ]

    for index, detection in enumerate(detections):
        color = palette[index % len(palette)]
        x1, y1, x2, y2 = detection.bbox_pixels
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        label = str(detection.id)
        label_bbox = draw.textbbox((0, 0), label)
        label_width = label_bbox[2] - label_bbox[0] + 8
        label_height = label_bbox[3] - label_bbox[1] + 6
        label_y = y1 if y1 + label_height <= canvas.height else max(0, y1 - label_height)
        draw.rectangle(
            (x1, label_y, min(canvas.width, x1 + label_width), label_y + label_height),
            fill=color,
        )
        draw.text((x1 + 4, label_y + 2), label, fill="white")

    return canvas


def call_qwen(
    client: OpenAI,
    model: str,
    image: Image.Image,
    max_tokens: int,
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image)},
                    },
                ],
            },
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as loaded:
        image = loaded.convert("RGB")
    width, height = image.size

    (output_dir / "system_prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    (output_dir / "user_prompt.txt").write_text(USER_PROMPT, encoding="utf-8")
    write_json(
        output_dir / "run_config.json",
        {
            "input_image": str(image_path),
            "image_width": width,
            "image_height": height,
            "endpoint": args.endpoint,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "coordinate_system": "normalized 0..1000",
            "single_full_image_request": True,
            "postprocessing": "schema validation and coordinate conversion only",
        },
    )

    client = OpenAI(
        base_url=args.endpoint,
        api_key=args.api_key,
        timeout=args.timeout,
    )

    started = time.perf_counter()
    raw_response = call_qwen(
        client,
        args.model,
        image,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    elapsed = time.perf_counter() - started
    (output_dir / "raw_response.txt").write_text(raw_response, encoding="utf-8")

    try:
        parsed = parse_model_json(raw_response)
    except (json.JSONDecodeError, ValueError) as exc:
        write_json(
            output_dir / "results.json",
            {
                "status": "json_parse_error",
                "error": str(exc),
                "elapsed_seconds": round(elapsed, 3),
                "valid_detections": [],
                "invalid_objects": [],
            },
        )
        print(f"Model response was saved, but JSON parsing failed: {exc}", file=sys.stderr)
        return 2

    write_json(output_dir / "parsed_response.json", parsed)

    try:
        detections, invalid_objects = validate_detections(parsed, width, height)
    except ValueError as exc:
        write_json(
            output_dir / "results.json",
            {
                "status": "schema_error",
                "error": str(exc),
                "elapsed_seconds": round(elapsed, 3),
                "valid_detections": [],
                "invalid_objects": [],
            },
        )
        print(f"JSON was saved, but schema validation failed: {exc}", file=sys.stderr)
        return 3

    result = {
        "status": "ok" if not invalid_objects else "partial_schema_error",
        "elapsed_seconds": round(elapsed, 3),
        "image": {"path": str(image_path), "width": width, "height": height},
        "model": args.model,
        "coordinate_system": {
            "normalized_width": 1000,
            "normalized_height": 1000,
            "origin": "top-left",
        },
        "valid_detection_count": len(detections),
        "invalid_object_count": len(invalid_objects),
        "valid_detections": [asdict(item) for item in detections],
        "invalid_objects": invalid_objects,
    }
    write_json(output_dir / "results.json", result)
    draw_detections(image, detections).save(output_dir / "visualization.png")

    print(f"Detected {len(detections)} valid module(s) in {elapsed:.2f}s")
    if invalid_objects:
        print(f"Model also returned {len(invalid_objects)} invalid object(s)")
    print(f"Raw response: {output_dir / 'raw_response.txt'}")
    print(f"Results:      {output_dir / 'results.json'}")
    print(f"Visualization:{output_dir / 'visualization.png'}")
    return 0 if not invalid_objects else 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test Qwen's zero-shot ability to find independent modules in one full image."
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "-o",
        "--output",
        default="output_module_blind_test",
        help="Output directory (default: output_module_blind_test)",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LOCAL_VLM_API_KEY", DEFAULT_API_KEY),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        return run(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
