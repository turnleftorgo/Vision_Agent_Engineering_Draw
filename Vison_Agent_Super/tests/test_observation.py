from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image


SUPER_DIR = Path(__file__).resolve().parents[1]
if str(SUPER_DIR) not in sys.path:
    sys.path.insert(0, str(SUPER_DIR))

from recovery.models import CropBox  # noqa: E402
from recovery.observation import build_dynamic_observation  # noqa: E402


class ObservationTests(unittest.TestCase):
    def test_context_is_rebuilt_around_current_crop(self) -> None:
        image = Image.new("RGB", (1000, 800), "white")
        observation = build_dynamic_observation(
            image,
            CropBox(500, 300, 700, 500),
            CropBox(520, 330, 550, 360),
            turn=2,
            context_fraction=0.5,
        )
        self.assertEqual(observation.context_bbox.to_int_tuple(), (400, 200, 800, 600))
        self.assertEqual(observation.crop_image.size, (200, 200))

    def test_full_image_boundary_is_locked(self) -> None:
        image = Image.new("RGB", (500, 400), "white")
        observation = build_dynamic_observation(
            image,
            CropBox(0, 0, 200, 150),
            CropBox(20, 20, 50, 50),
            turn=1,
        )
        self.assertTrue(observation.boundary_locked["left"])
        self.assertTrue(observation.boundary_locked["top"])


if __name__ == "__main__":
    unittest.main()
