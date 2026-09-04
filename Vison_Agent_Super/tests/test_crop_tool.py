from __future__ import annotations

import sys
import unittest
from pathlib import Path


SUPER_DIR = Path(__file__).resolve().parents[1]
if str(SUPER_DIR) not in sys.path:
    sys.path.insert(0, str(SUPER_DIR))

from recovery.crop_tool import expand_crop  # noqa: E402
from recovery.models import (  # noqa: E402
    CropBox,
    ExpansionNorm,
    RecoveryAction,
    RecoveryConfig,
)


def action(**directions: int) -> RecoveryAction:
    return RecoveryAction(
        "expand_crop",
        False,
        True,
        ("missing",),
        ExpansionNorm(**directions),
        0.8,
    )


class CropToolTests(unittest.TestCase):
    def test_single_direction_uses_current_width(self) -> None:
        before = CropBox(100, 100, 300, 300)
        result = expand_crop(
            before,
            action(right=500),
            (1000, 1000),
            before.area,
            RecoveryConfig(),
        )
        self.assertEqual(result.after.to_int_tuple(), (100, 100, 400, 300))
        self.assertEqual(before.to_int_tuple(), (100, 100, 300, 300))

    def test_locked_boundary_prevents_change(self) -> None:
        before = CropBox(0, 100, 200, 300)
        result = expand_crop(
            before,
            action(left=200),
            (1000, 1000),
            before.area,
            RecoveryConfig(),
        )
        self.assertFalse(result.changed)
        self.assertIn("left_boundary", result.limited_by)

    def test_area_cap_scales_expansion(self) -> None:
        before = CropBox(400, 400, 600, 600)
        config = RecoveryConfig(max_crop_area_ratio=1.0, max_crop_growth=1.10)
        result = expand_crop(
            before,
            action(left=500, top=500, right=500, bottom=500),
            (1000, 1000),
            before.area,
            config,
        )
        self.assertLessEqual(result.after.area, before.area * 1.10 + 1e-4)
        self.assertIn("area_cap", result.limited_by)


if __name__ == "__main__":
    unittest.main()
