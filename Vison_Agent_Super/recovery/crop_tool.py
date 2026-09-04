"""The only trusted operation allowed to change a recovery crop."""

from __future__ import annotations

from .models import CropBox, ExpansionResult, RecoveryAction, RecoveryConfig


def _expanded(
    current: CropBox,
    deltas: dict[str, float],
    width: int,
    height: int,
    scale: float = 1.0,
) -> CropBox:
    return current.expand(
        deltas["left"] * scale,
        deltas["top"] * scale,
        deltas["right"] * scale,
        deltas["bottom"] * scale,
    ).clamp(width, height)


def _same_pixels(first: CropBox, second: CropBox) -> bool:
    return first.to_int_tuple() == second.to_int_tuple()


def expand_crop(
    current: CropBox,
    action: RecoveryAction,
    full_size: tuple[int, int],
    initial_area: float,
    config: RecoveryConfig,
) -> ExpansionResult:
    """Apply one bounded expansion without side effects or global state."""
    if action.action != "expand_crop":
        raise ValueError("expand_crop requires an expand_crop action")
    width, height = full_size
    if width <= 0 or height <= 0:
        raise ValueError("full_size must be positive")
    before = current.clamp(width, height)
    if before.area <= 0:
        raise ValueError("current crop must have positive area")

    requested = action.arguments
    maximum = config.max_direction_norm
    clamped = {
        "left": min(maximum, max(0, requested.left)),
        "top": min(maximum, max(0, requested.top)),
        "right": min(maximum, max(0, requested.right)),
        "bottom": min(maximum, max(0, requested.bottom)),
    }
    locked_before = {
        "left": before.x1 <= 0,
        "top": before.y1 <= 0,
        "right": before.x2 >= width,
        "bottom": before.y2 >= height,
    }
    limited: list[str] = []
    deltas: dict[str, float] = {}
    for name in ("left", "top", "right", "bottom"):
        norm = clamped[name]
        if locked_before[name] and norm:
            limited.append(f"{name}_boundary")
            norm = 0
        base = before.width if name in {"left", "right"} else before.height
        pixels = base * norm / 1000.0
        if norm > 0:
            pixels = max(1.0, pixels)
        deltas[name] = pixels

    full_area = float(width * height)
    configured_cap = min(
        full_area * config.max_crop_area_ratio,
        max(float(initial_area), 1.0) * config.max_crop_growth,
    )
    area_cap = max(before.area, min(full_area, configured_cap))
    after = _expanded(before, deltas, width, height)
    if after.area > area_cap + 1e-6:
        low = 0.0
        high = 1.0
        for _ in range(48):
            middle = (low + high) / 2.0
            candidate = _expanded(before, deltas, width, height, middle)
            if candidate.area <= area_cap:
                low = middle
            else:
                high = middle
        after = _expanded(before, deltas, width, height, low)
        limited.append("area_cap")

    applied = {
        "left": max(0.0, before.x1 - after.x1),
        "top": max(0.0, before.y1 - after.y1),
        "right": max(0.0, after.x2 - before.x2),
        "bottom": max(0.0, after.y2 - before.y2),
    }
    locked_after = {
        "left": after.x1 <= 0,
        "top": after.y1 <= 0,
        "right": after.x2 >= width,
        "bottom": after.y2 >= height,
    }
    changed = not _same_pixels(before, after)
    if not changed and requested.total() > 0 and not limited:
        limited.append("no_pixel_change")
    return ExpansionResult(
        before=before,
        after=after,
        requested_norm=requested,
        applied_pixels=applied,
        boundary_locked=locked_after,
        changed=changed,
        area_cap=area_cap,
        limited_by=tuple(dict.fromkeys(limited)),
    )
