"""OpenAI-compatible vision inference helpers."""

from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from openai import DefaultHttpxClient, OpenAI
from PIL import Image

from .models import BBox


def create_vision_client(endpoint: str, api_key: str, timeout: float) -> OpenAI:
    """Create a local-model client that bypasses macOS/system HTTP proxies."""
    return OpenAI(
        base_url=endpoint,
        api_key=api_key,
        http_client=DefaultHttpxClient(timeout=timeout, trust_env=False),
    )


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
    request_overrides: dict[str, Any] = {}
    if "locateanything" in model.lower():
        # LocateAnything is a grounding model, not a reasoning model.  Explicitly
        # override oMLX per-model defaults so an accidentally enabled thinking
        # profile cannot stall or fail the request.
        request_overrides["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False},
            "thinking_budget": 0,
        }
    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=temperature,
                **request_overrides,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
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


def normalized_box_to_pixels(
    values: Iterable[float], width: int, height: int
) -> BBox:
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
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = data.get("boxes") or data.get("detections") or data.get("objects") or []
    else:
        records = []
    for record in records:
        values: Any = record
        if isinstance(record, dict):
            values = record.get("bbox") or record.get("box") or record.get("bbox_2d")
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
        client, model, prompt, [image], max_tokens=4096, temperature=0.1
    )
    if raw_output_path:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_output_path.write_text(response, encoding="utf-8")
    return parse_locate_response(response, image.width, image.height)
