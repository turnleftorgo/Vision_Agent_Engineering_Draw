#!/usr/bin/env python3
"""Detect complete FAI clusters in one small engineering-drawing image.

The original full image is sent to Qwen exactly once. The script performs no
tiling, OCR, classical-vision proposals, box expansion, merging, or NMS.
Every valid model box is converted to pixels, cropped, and drawn unchanged.
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

SYSTEM_PROMPT = """你是机械工程图纸中的 FAI 集群检测器。

请观察输入的完整工程图，找出所有独立的 FAI 集群，并为每个集群输出一个完整的整体边界框。

【FAI 集群定义】

每个 FAI 集群以一个明确写有“FAI”和检验编号的圆形标记为中心，例如：

FAI
180

一个完整的 FAI 集群必须包含以下相关内容：

1. FAI 圆形标记及其检验编号。
2. 与该 FAI 紧邻并属于同一组的 SPC 圆形标记及 SPC 字母。
3. 与该 FAI 对应的完整尺寸、公差、形位公差框、参数或描述文字。
4. 从该组标注出发的所有相关 leader、extension line 或引出线。
5. 每条相关引出线终端的箭头或箭头头部。
6. 每个箭头实际指向并归属的完整部件或完整结构单元，包括该部件可见的连续外轮廓及其内部必要特征。

【目标区域规则】

- 每个 FAI 标记必须单独产生一个 FAI 集群。
- 除了整体 bbox，还必须输出该集群中 FAI 圆形标记本身的 marker_bbox；marker_bbox 只框住包含“FAI”和检验编号的圆圈。
- 不得把相邻但属于不同 FAI 编号的标注合并成一个大框。
- 同一 FAI 有多条引出线或多个箭头时，必须全部包含在同一个集群框中。
- 边界框必须覆盖从 FAI/SPC 标记、描述文字到箭头目标之间的完整关联路径。
- 对每个终端箭头，以箭头端点实际接触的位置为种子，沿与该位置相连的可见连续轮廓向外追踪，直到该部件或结构单元的自然边界、闭合外轮廓、明确分界或可见终点。
- 必须包含箭头所属的完整部件或完整结构单元，不能只包含箭头端点附近的一小段线、一个角、一个孔或一小块表面。
- 如果目标是具有独立闭合轮廓的部件，必须包含该闭合轮廓所表示的完整部件；如果轮廓因图片边缘而不完整，则包含全部可见部分并设置 complete=false。
- 如果目标没有独立闭合轮廓，应沿连续轮廓和结构关系包含其所属的完整结构单元；不能为了缩小边界框而截掉该结构单元的任何可见部分。
- 边界框不能截断相关文字、leader、arrowhead 或目标部件的连续外轮廓。
- 对 FAI/SPC、关联标注、完整引出路径、箭头和完整目标部件求一个能够完整覆盖全部内容的轴对齐外接矩形。
- 完整性优先于边界框大小。宁可边界框更大并包含周围其他内容，也不能遗漏或截断箭头所指部件或结构单元的任何可见轮廓。
- 当目标部件的边界存在不确定性时，应向外扩展到能够确认完整部件边界的位置，不得保守地只框箭头附近区域。
- 相邻的不同 FAI 标记仍必须分别建立输出对象；即使它们的边界框因覆盖完整部件而大幅重叠，也不能合并为同一个 FAI 集群。
- 不要把普通圆孔、基准符号、圆圈尺寸、气泡编号或单独的数字误认为 FAI。
- 只有明确包含“FAI”文字和检验编号的圆形标记才能建立 FAI 集群。
- 如果某个 FAI 集群在图片边缘被截断，仍然输出可见部分的边界框，但设置 complete=false，并在 missing 中说明被截断的内容。
- 如果两个 FAI 集群共享同一个零件区域，它们的边界框允许重叠，但仍必须分别输出。

【关联判断顺序】

对每个 FAI 标记严格按照以下顺序关联：

1. 读取 FAI 检验编号。
2. 找到同组 SPC 标记。
3. 找到与该 FAI 对应的参数、公差和描述框。
4. 从描述框开始逐段追踪所有相关引出线。
5. 找到每条引出线的终端箭头。
6. 以箭头端点为种子确认其接触和归属的部件或结构单元。
7. 沿连续轮廓追踪该部件或结构单元的完整可见边界，不得停留在箭头附近的局部区域。
8. 对 FAI/SPC、标注、完整引出路径、箭头和完整目标部件求能够覆盖全部内容的轴对齐外接矩形，完整性优先于框的紧密程度。

不要仅按照距离关联。必须根据文字排列、线段连续性、箭头方向和目标接触关系判断归属。

【坐标规范】

- 使用归一化坐标。
- 图像左上角为 (0, 0)。
- 图像右下角为 (1000, 1000)。
- 边界框格式为：

[x_min, y_min, x_max, y_max]

必须满足：

0 <= x_min < x_max <= 1000
0 <= y_min < y_max <= 1000

【输出要求】

只输出一个合法 JSON 对象，不要输出分析过程、Markdown、代码块、XML 标签或其他文字。

输出格式：

{
  "fai_clusters": [
    {
      "id": 1,
      "fai_number": "180",
      "spc_code": "FC",
      "bbox": [x_min, y_min, x_max, y_max],
      "marker_bbox": [x_min, y_min, x_max, y_max],
      "description": "与该FAI关联的参数、公差或特征的简短描述",
      "arrow_count": 1,
      "target_summary": "箭头指向并归属的完整部件或结构单元",
      "complete": true,
      "missing": [],
      "confidence": 0.95
    }
  ]
}

字段规则：

- id：从1开始连续编号。
- fai_number：FAI圆圈内的检验编号，无法读取时使用 null。
- spc_code：同组SPC代码，未发现时使用 null。
- bbox：整个FAI集群的归一化边界框。
- marker_bbox：该对象中 FAI 圆形标记（包含 FAI 和检验编号）的归一化边界框，坐标相对于输入 module 图像。
- description：简短概括相关标注内容，不要编造无法读取的文字。
- arrow_count：确认属于该FAI集群的终端箭头数量。
- target_summary：简短描述箭头指向并归属的完整部件或完整结构单元。
- complete：FAI、SPC、描述、相关引出线、箭头和目标部件的完整可见轮廓是否全部可见。
- missing：complete=true 时必须是空数组；否则列出缺失或被截断的内容。
- confidence：0到1之间的数字。

最后再次检查：

1. 图片中每个明确的 FAI 标记是否都恰好对应一个输出对象。
2. 是否遗漏任何相关 leader 或终端箭头。
3. 是否错误合并了两个相邻 FAI 集群。
4. 是否只框住了箭头附近，而遗漏箭头所属部件或结构单元的完整连续轮廓。
5. 是否因为追求较小边界框而遗漏了目标部件的任何可见轮廓；如有疑问，应扩大边界框以保证部件完整。
6. 是否有边界框截断相关文字、线段、箭头或目标部件轮廓。"""

USER_PROMPT = "请按照上述规则检测这张完整工程图中的所有 FAI 集群。"


@dataclass(frozen=True)
class FAICluster:
    id: int
    fai_number: str | None
    spc_code: str | None
    bbox_normalized: list[float]
    bbox_pixels: list[int]
    marker_bbox_normalized: list[float] | None
    marker_bbox_pixels: list[int] | None
    description: str
    arrow_count: int
    target_summary: str
    complete: bool
    missing: list[str]
    confidence: float


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def remove_wrappers(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_response(raw_text: str) -> dict[str, Any]:
    value = json.loads(remove_wrappers(raw_text))
    if not isinstance(value, dict):
        raise ValueError("The top-level JSON value must be an object")
    return value


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_clusters(
    response: dict[str, Any], image_width: int, image_height: int
) -> tuple[list[FAICluster], list[dict[str, Any]]]:
    items = response.get("fai_clusters")
    if not isinstance(items, list):
        raise ValueError('The response must contain a "fai_clusters" array')

    valid: list[FAICluster] = []
    invalid: list[dict[str, Any]] = []

    for index, item in enumerate(items):
        errors: list[str] = []
        if not isinstance(item, dict):
            invalid.append(
                {"index": index, "value": item, "errors": ["cluster must be an object"]}
            )
            continue

        cluster_id = item.get("id")
        if not isinstance(cluster_id, int) or isinstance(cluster_id, bool):
            errors.append("id must be an integer")

        bbox = item.get("bbox")
        coordinates: list[float] | None = None
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append("bbox must contain four numbers")
        elif not all(is_number(value) for value in bbox):
            errors.append("bbox coordinates must be numbers")
        else:
            coordinates = [float(value) for value in bbox]
            x1, y1, x2, y2 = coordinates
            if not all(0.0 <= value <= 1000.0 for value in coordinates):
                errors.append("bbox coordinates must be within [0, 1000]")
            if x1 >= x2 or y1 >= y2:
                errors.append("bbox must satisfy x_min < x_max and y_min < y_max")

        marker_bbox = item.get("marker_bbox")
        marker_coordinates: list[float] | None = None
        if marker_bbox is not None:
            if not isinstance(marker_bbox, list) or len(marker_bbox) != 4:
                errors.append("marker_bbox must contain four numbers")
            elif not all(is_number(value) for value in marker_bbox):
                errors.append("marker_bbox coordinates must be numbers")
            else:
                marker_coordinates = [float(value) for value in marker_bbox]
                mx1, my1, mx2, my2 = marker_coordinates
                if not all(0.0 <= value <= 1000.0 for value in marker_coordinates):
                    errors.append("marker_bbox coordinates must be within [0, 1000]")
                if mx1 >= mx2 or my1 >= my2:
                    errors.append(
                        "marker_bbox must satisfy x_min < x_max and y_min < y_max"
                    )

        fai_number = item.get("fai_number")
        if fai_number is not None and not isinstance(fai_number, str):
            errors.append("fai_number must be a string or null")

        spc_code = item.get("spc_code")
        if spc_code is not None and not isinstance(spc_code, str):
            errors.append("spc_code must be a string or null")

        description = item.get("description")
        if not isinstance(description, str):
            errors.append("description must be a string")

        arrow_count = item.get("arrow_count")
        if (
            not isinstance(arrow_count, int)
            or isinstance(arrow_count, bool)
            or arrow_count < 0
        ):
            errors.append("arrow_count must be a non-negative integer")

        target_summary = item.get("target_summary")
        if not isinstance(target_summary, str):
            errors.append("target_summary must be a string")

        complete = item.get("complete")
        if not isinstance(complete, bool):
            errors.append("complete must be a boolean")

        missing = item.get("missing")
        if not isinstance(missing, list) or not all(
            isinstance(value, str) for value in missing
        ):
            errors.append("missing must be an array of strings")
        elif complete is True and missing:
            errors.append("missing must be empty when complete is true")

        confidence = item.get("confidence")
        if not is_number(confidence) or not 0.0 <= float(confidence) <= 1.0:
            errors.append("confidence must be a number within [0, 1]")

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
        marker_pixels = None
        if marker_coordinates is not None:
            mx1, my1, mx2, my2 = marker_coordinates
            marker_pixels = [
                round(mx1 / 1000.0 * image_width),
                round(my1 / 1000.0 * image_height),
                round(mx2 / 1000.0 * image_width),
                round(my2 / 1000.0 * image_height),
            ]
        valid.append(
            FAICluster(
                id=cluster_id,
                fai_number=fai_number,
                spc_code=spc_code,
                bbox_normalized=coordinates,
                bbox_pixels=pixels,
                marker_bbox_normalized=marker_coordinates,
                marker_bbox_pixels=marker_pixels,
                description=description,
                arrow_count=arrow_count,
                target_summary=target_summary,
                complete=complete,
                missing=missing,
                confidence=float(confidence),
            )
        )

    return valid, invalid


def safe_name(value: str | None) -> str:
    if not value:
        return "unknown"
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return cleaned or "unknown"


def save_crops(
    image: Image.Image, clusters: list[FAICluster], output_dir: Path
) -> list[str]:
    crops_dir = output_dir / "crop"
    crops_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    for index, cluster in enumerate(clusters, start=1):
        x1, y1, x2, y2 = cluster.bbox_pixels
        filename = f"FAI_{safe_name(cluster.fai_number)}_{index:03d}.png"
        image.crop((x1, y1, x2, y2)).save(crops_dir / filename)
        paths.append(str(Path("crop") / filename))

    return paths


def draw_visualization(image: Image.Image, clusters: list[FAICluster]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, round(max(canvas.size) / 650))
    palette = [
        "#ff00d4",
        "#00a651",
        "#0066ff",
        "#ff7a00",
        "#7b2cff",
        "#00a6a6",
        "#e31a1c",
        "#70543e",
    ]

    for index, cluster in enumerate(clusters):
        color = palette[index % len(palette)]
        x1, y1, x2, y2 = cluster.bbox_pixels
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        number = cluster.fai_number if cluster.fai_number is not None else "?"
        label = f"FAI {number}"
        text_box = draw.textbbox((0, 0), label)
        label_width = text_box[2] - text_box[0] + 10
        label_height = text_box[3] - text_box[1] + 8
        label_y = y1 if y1 + label_height <= canvas.height else max(0, y1 - label_height)
        draw.rectangle(
            (x1, label_y, min(canvas.width, x1 + label_width), label_y + label_height),
            fill=color,
        )
        draw.text((x1 + 5, label_y + 3), label, fill="white")

    return canvas


def call_qwen(
    client: OpenAI,
    model: str,
    image: Image.Image,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
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
    choice = response.choices[0]
    metadata = {
        "finish_reason": choice.finish_reason,
        "usage": response.usage.model_dump() if response.usage is not None else None,
    }
    return (choice.message.content or "").strip(), metadata


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
            "postprocessing": "schema validation, coordinate conversion, crop, and drawing only",
        },
    )

    client = OpenAI(
        base_url=args.endpoint,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    print(f"Sending one full-image request to {args.model} ...", flush=True)
    started = time.perf_counter()
    raw_response, response_metadata = call_qwen(
        client,
        args.model,
        image,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    elapsed = time.perf_counter() - started
    (output_dir / "raw_response.txt").write_text(raw_response, encoding="utf-8")
    write_json(output_dir / "response_metadata.json", response_metadata)

    try:
        parsed = parse_response(raw_response)
    except (json.JSONDecodeError, ValueError) as exc:
        write_json(
            output_dir / "results.json",
            {
                "status": "json_parse_error",
                "error": str(exc),
                "elapsed_seconds": round(elapsed, 3),
                "response_metadata": response_metadata,
                "valid_clusters": [],
                "invalid_clusters": [],
            },
        )
        print(f"Model response was saved, but JSON parsing failed: {exc}", file=sys.stderr)
        return 2

    write_json(output_dir / "parsed_response.json", parsed)
    try:
        clusters, invalid_clusters = validate_clusters(parsed, width, height)
    except ValueError as exc:
        write_json(
            output_dir / "results.json",
            {
                "status": "schema_error",
                "error": str(exc),
                "elapsed_seconds": round(elapsed, 3),
                "response_metadata": response_metadata,
                "valid_clusters": [],
                "invalid_clusters": [],
            },
        )
        print(f"JSON was saved, but schema validation failed: {exc}", file=sys.stderr)
        return 3

    crop_paths = save_crops(image, clusters, output_dir)
    serialized_clusters: list[dict[str, Any]] = []
    for cluster, crop_path in zip(clusters, crop_paths):
        item = asdict(cluster)
        item["crop_path"] = crop_path
        serialized_clusters.append(item)

    results = {
        "status": "ok" if not invalid_clusters else "partial_schema_error",
        "elapsed_seconds": round(elapsed, 3),
        "image": {"path": str(image_path), "width": width, "height": height},
        "model": args.model,
        "response_metadata": response_metadata,
        "coordinate_system": {
            "normalized_width": 1000,
            "normalized_height": 1000,
            "origin": "top-left",
        },
        "valid_cluster_count": len(clusters),
        "invalid_cluster_count": len(invalid_clusters),
        "valid_clusters": serialized_clusters,
        "invalid_clusters": invalid_clusters,
    }
    write_json(output_dir / "results.json", results)
    draw_visualization(image, clusters).save(
        output_dir / "visualization.jpg", format="JPEG", quality=95, subsampling=0
    )

    print(f"Detected {len(clusters)} valid FAI cluster(s) in {elapsed:.2f}s")
    if invalid_clusters:
        print(f"Model also returned {len(invalid_clusters)} invalid cluster(s)")
    print(f"Raw response:  {output_dir / 'raw_response.txt'}")
    print(f"Results:       {output_dir / 'results.json'}")
    print(f"Visualization: {output_dir / 'visualization.jpg'}")
    print(f"Crops:         {output_dir / 'crop'}")
    return 0 if not invalid_clusters else 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect complete FAI/SPC/leader/arrow/target clusters in one small image."
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "-o",
        "--output",
        default="output_fai_clusters",
        help="Output directory (default: output_fai_clusters)",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LOCAL_VLM_API_KEY", DEFAULT_API_KEY),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
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
