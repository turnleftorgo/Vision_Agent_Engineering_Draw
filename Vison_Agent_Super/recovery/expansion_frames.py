"""Human-review frames for every crop expansion that actually changed pixels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .models import CropBox, ExpansionResult, RecoveryAction
from .observation import build_crop_image, fit_image

_EXPAND_ORDER = ("left", "top", "right", "bottom")


def expansion_frame_name(expansion: ExpansionResult) -> str:
    """Name a frame by the directions that actually changed, e.g. expand_left_172px."""
    parts = [
        f"{name}_{round(expansion.applied_pixels[name])}px"
        for name in _EXPAND_ORDER
        if round(expansion.applied_pixels[name]) > 0
    ]
    return "expand_" + "_".join(parts) if parts else "expand"


def dominant_expansion_direction(expansion: ExpansionResult) -> str:
    name, value = max(
        ((name, expansion.applied_pixels[name]) for name in _EXPAND_ORDER),
        key=lambda item: item[1],
    )
    return name if value > 0 else ""


def build_transition_image(
    full_image: Image.Image,
    marker_box: CropBox,
    before: CropBox,
    after: CropBox,
    *,
    context_fraction: float,
    max_image_edge: int,
) -> Image.Image:
    """Wider context with marker (red), pre-expansion (purple) and post (green) boxes."""
    width, height = full_image.size
    fraction = max(0.05, min(1.0, context_fraction))
    context = after.expand(
        after.width * fraction,
        after.height * fraction,
        after.width * fraction,
        after.height * fraction,
    ).clamp(width, height)
    canvas = full_image.crop(context.to_int_tuple()).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    stroke = max(2, round(min(canvas.size) * 0.004))

    def local(box: CropBox) -> tuple[int, int, int, int]:
        return box.translate(-context.x1, -context.y1).to_int_tuple()

    draw.rectangle(local(before), outline=(220, 0, 220), width=stroke)
    draw.rectangle(local(after), outline=(0, 160, 0), width=stroke)
    draw.rectangle(local(marker_box), outline=(255, 0, 0), width=stroke + 1)

    legend = [
        ("FAI MARKER", (255, 0, 0)),
        ("BEFORE", (220, 0, 220)),
        ("AFTER", (0, 160, 0)),
    ]
    label_x, label_y = 4, 4
    draw.rectangle(
        (label_x, label_y, label_x + 118, label_y + 3 * 15 + 6), fill="white"
    )
    for offset, (text, color) in enumerate(legend):
        draw.text((label_x + 3, label_y + 3 + offset * 15), text, fill=color)

    canvas, _ = fit_image(canvas, max_image_edge)
    return canvas


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def save_expansion_frame(
    directory: Path,
    *,
    full_image: Image.Image,
    marker_box: CropBox,
    initial_crop: CropBox,
    action: RecoveryAction,
    expansion: ExpansionResult,
    turn_number: int,
    expansion_number: int,
    context_fraction: float,
    max_image_edge: int,
) -> dict[str, Any]:
    """Persist one real expansion; numbering counts expansions, not observations.

    The saved after-frame is rendered through build_crop_image, the same path the
    next observation uses, so it is byte-identical to the next turn's crop image.
    """
    directory.mkdir(parents=True, exist_ok=True)
    if expansion_number == 1:
        initial_image, _ = build_crop_image(full_image, initial_crop, max_image_edge)
        initial_image.save(directory / "00_initial.png")

    frame_stem = (
        f"{expansion_number:02d}_turn_{turn_number:02d}_"
        f"{expansion_frame_name(expansion)}"
    )
    after_image, _ = build_crop_image(full_image, expansion.after, max_image_edge)
    after_image.save(directory / f"{frame_stem}.png")
    build_transition_image(
        full_image,
        marker_box,
        expansion.before,
        expansion.after,
        context_fraction=context_fraction,
        max_image_edge=max_image_edge,
    ).save(directory / f"{frame_stem}_transition.png")

    entry: dict[str, Any] = {
        "expansion": expansion_number,
        "turn": turn_number,
        "reason": list(action.missing),
        "direction": dominant_expansion_direction(expansion),
        "requested_norm": expansion.requested_norm.to_dict(),
        "applied_pixels": {
            name: round(expansion.applied_pixels[name], 3) for name in _EXPAND_ORDER
        },
        "before": expansion.before.to_list(),
        "after": expansion.after.to_list(),
        "limited_by": list(expansion.limited_by),
        "frame": f"{frame_stem}.png",
        "transition": f"{frame_stem}_transition.png",
    }
    timeline_path = directory / "timeline.json"
    timeline: list[dict[str, Any]] = []
    if timeline_path.exists():
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    timeline.append(entry)
    _save_json(timeline_path, timeline)
    return entry
