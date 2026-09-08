from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


SUPER_DIR = Path(__file__).resolve().parents[1]
if str(SUPER_DIR) not in sys.path:
    sys.path.insert(0, str(SUPER_DIR))

from vision.detection import (  # noqa: E402
    axis_starts,
    circle_pair_marker_boxes,
    deduplicate_boxes,
    detect_fai_candidates,
)
from vision.inference import parse_locate_response  # noqa: E402
from vision.models import BBox, Primitive  # noqa: E402
from vision.utils import mapping_selected_ids, safe_fai_name  # noqa: E402


class StandaloneVisionTests(unittest.TestCase):
    def test_super_sources_do_not_import_v3(self) -> None:
        forbidden_module = "FAI_DET" + "_CROP_3"
        offenders = []
        for path in SUPER_DIR.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports = [
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in (
                    node.names
                    if isinstance(node, ast.Import)
                    else [ast.alias(name=node.module or "")]
                )
            ]
            if any(name == forbidden_module for name in imports):
                offenders.append(str(path.relative_to(SUPER_DIR)))
        self.assertEqual(offenders, [])

    def test_locate_response_uses_normalized_coordinates(self) -> None:
        boxes = parse_locate_response("<box><100><200><500><800></box>", 200, 100)
        self.assertEqual([box.to_int_tuple() for box in boxes], [(20, 20, 100, 80)])

    def test_deduplication_and_tile_edges_are_stable(self) -> None:
        boxes = deduplicate_boxes(
            [BBox(0, 0, 20, 20), BBox(1, 1, 19, 19), BBox(80, 80, 90, 90)]
        )
        self.assertEqual(len(boxes), 2)
        self.assertEqual(axis_starts(2500, 1200, 0.2)[-1], 1300)

    def test_circle_pairs_are_promoted_without_model_validation(self) -> None:
        pairs = [
            {"left_bbox": BBox(10, 10, 30, 30), "right_bbox": BBox(35, 10, 55, 30)},
            {"left_bbox": BBox(11, 11, 29, 29), "right_bbox": BBox(36, 11, 54, 29)},
            {"left_bbox": BBox(80, 80, 100, 100), "right_bbox": BBox(105, 80, 125, 100)},
        ]
        boxes = circle_pair_marker_boxes(pairs)
        self.assertEqual(
            [box.to_int_tuple() for box in boxes],
            [(10, 10, 30, 30), (80, 80, 100, 100)],
        )

    def test_each_tile_combines_locateanything_and_opencv_candidates(self) -> None:
        image = Image.new("RGB", (100, 100), "white")
        pair = {
            "left_bbox": BBox(60, 60, 80, 80),
            "right_bbox": BBox(82, 60, 98, 80),
        }
        with (
            tempfile.TemporaryDirectory() as temporary_dir,
            patch("vision.detection.locate_boxes", return_value=[BBox(10, 10, 30, 30)]),
            patch("vision.detection.detect_circle_pair_candidates", return_value=[pair]),
        ):
            boxes = detect_fai_candidates(
                object(),
                "LocateAnything-3B-8bit",
                image,
                100,
                0.2,
                Path(temporary_dir),
                None,
            )
        self.assertEqual(
            [box.to_int_tuple() for box in boxes],
            [(10, 10, 30, 30), (60, 60, 80, 80)],
        )

    def test_primitive_contract_and_mapping_helpers(self) -> None:
        primitive = Primitive("A0", "annotation", BBox(1, 2, 3, 4), "test")
        self.assertEqual(primitive.prompt_record()["bbox"], [1, 2, 3, 4])
        self.assertEqual(
            mapping_selected_ids(
                {"annotation_ids": ["A0"], "leader_ids": ["L0", "L0"]}
            ),
            ["F0", "A0", "L0"],
        )
        self.assertEqual(safe_fai_name("10 / A"), "10_A")


if __name__ == "__main__":
    unittest.main()
