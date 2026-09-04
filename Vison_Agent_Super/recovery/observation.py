"""Generate a fresh crop and dynamic wider context from the full drawing."""

from __future__ import annotations

from typing import Optional

from PIL import Image, ImageDraw

from .models import CropBox, ExpansionResult, RecoveryObservation


def fit_image(image: Image.Image, max_edge: int) -> tuple[Image.Image, float]:
    longest = max(image.size)
    if longest <= max_edge:
        return image, 1.0
    scale = max_edge / float(longest)
    size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    return image.resize(size, Image.Resampling.LANCZOS), scale


def build_crop_image(
    full_image: Image.Image,
    crop: CropBox,
    max_image_edge: int,
) -> tuple[Image.Image, float]:
    """Render the exact crop frame the model observes for this crop box."""
    box = CropBox.from_values(crop.clamp(*full_image.size).to_int_tuple())
    clean_crop = full_image.crop(box.to_int_tuple()).convert("RGB")
    return fit_image(clean_crop, max_image_edge)


def build_dynamic_observation(
    full_image: Image.Image,
    current_crop: CropBox,
    marker_box: CropBox,
    *,
    turn: int,
    context_fraction: float = 0.50,
    max_image_edge: int = 2400,
    previous_expansion: Optional[ExpansionResult] = None,
) -> RecoveryObservation:
    """Build an immutable observation directly from the original pixels."""
    width, height = full_image.size
    current = current_crop.clamp(width, height)
    if current.area <= 0:
        raise ValueError("Cannot observe an empty crop")
    fraction = max(0.05, min(1.0, context_fraction))
    context = current.expand(
        current.width * fraction,
        current.height * fraction,
        current.width * fraction,
        current.height * fraction,
    ).clamp(width, height)
    context_tuple = context.to_int_tuple()
    context = CropBox.from_values(context_tuple)
    current_tuple = current.to_int_tuple()
    current = CropBox.from_values(current_tuple)

    clean_crop, crop_scale = build_crop_image(full_image, current, max_image_edge)
    context_image = full_image.crop(context_tuple).convert("RGB")
    draw = ImageDraw.Draw(context_image)
    local_crop = current.translate(-context.x1, -context.y1)
    local_marker = marker_box.translate(-context.x1, -context.y1)
    stroke = max(2, round(min(context_image.size) * 0.004))
    draw.rectangle(local_crop.to_int_tuple(), outline=(220, 0, 220), width=stroke)
    draw.rectangle(local_marker.to_int_tuple(), outline=(255, 0, 0), width=stroke)
    label_x = max(0, int(local_crop.x1) + 3)
    label_y = max(0, int(local_crop.y1) + 3)
    draw.rectangle((label_x, label_y, label_x + 122, label_y + 17), fill="white")
    draw.text((label_x + 2, label_y + 2), "CURRENT CROP", fill=(220, 0, 220))

    locked = {
        "left": current.x1 <= 0,
        "top": current.y1 <= 0,
        "right": current.x2 >= width,
        "bottom": current.y2 >= height,
    }
    grey = (120, 120, 120)
    if locked["left"]:
        draw.line((0, 0, 0, context_image.height), fill=grey, width=stroke * 2)
    if locked["top"]:
        draw.line((0, 0, context_image.width, 0), fill=grey, width=stroke * 2)
    if locked["right"]:
        draw.line(
            (context_image.width - 1, 0, context_image.width - 1, context_image.height),
            fill=grey,
            width=stroke * 2,
        )
    if locked["bottom"]:
        draw.line(
            (
                0,
                context_image.height - 1,
                context_image.width,
                context_image.height - 1,
            ),
            fill=grey,
            width=stroke * 2,
        )

    if previous_expansion is not None:
        applied = previous_expansion.applied_pixels
        text = "LAST EXPAND " + ", ".join(
            f"{name}={round(value)}" for name, value in applied.items() if value > 0
        )
        if text != "LAST EXPAND ":
            draw.rectangle(
                (
                    4,
                    context_image.height - 22,
                    min(context_image.width - 1, 420),
                    context_image.height - 3,
                ),
                fill="white",
            )
            draw.text((7, context_image.height - 20), text, fill=(180, 120, 0))

    context_image, context_scale = fit_image(context_image, max_image_edge)
    return RecoveryObservation(
        turn=turn,
        crop_bbox=current,
        context_bbox=context,
        crop_image=clean_crop,
        context_image=context_image,
        boundary_locked=locked,
        crop_scale=crop_scale,
        context_scale=context_scale,
    )
