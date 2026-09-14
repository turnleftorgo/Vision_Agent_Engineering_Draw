from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw


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
from vision.evidence import extend_crop_along_selected_leaders  # noqa: E402
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
        draw = ImageDraw.Draw(image)
        draw.ellipse((60, 60, 80, 80), outline="black", width=2)
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

    def test_per_tile_opencv_rejects_non_circular_left_candidate(self) -> None:
        image = Image.new("RGB", (140, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.line((20, 20, 60, 60), fill="black", width=2)
        draw.line((20, 60, 60, 20), fill="black", width=2)
        pair = {
            "left_bbox": BBox(20, 20, 60, 60),
            "right_bbox": BBox(65, 20, 105, 60),
        }
        self.assertEqual(circle_pair_marker_boxes([pair], image), [])

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

    def test_selected_leader_is_traced_to_local_target_geometry(self) -> None:
        source = Image.new("L", (600, 300), 255)
        draw = ImageDraw.Draw(source)
        draw.ellipse((80, 130, 110, 160), outline=0, width=2)
        # A leader exits the semantic ROI and terminates on a vertical part edge.
        draw.line((120, 150, 450, 165), fill=0, width=3)
        draw.line((450, 90, 450, 230), fill=0, width=4)
        image = np.asarray(source)
        primitives = [
            Primitive("F0", "fai_marker", BBox(80, 130, 110, 160), "test"),
            Primitive(
                "L0",
                "leader_segment",
                BBox(120, 150, 261, 158),
                "test",
                points=[(120, 150), (260, 156)],
            ),
        ]
        crop, traces = extend_crop_along_selected_leaders(
            image,
            BBox(0, 0, 300, 300),
            BBox(50, 80, 280, 220),
            primitives,
            ["F0", "L0"],
            (600, 300),
        )
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["side"], "right")
        self.assertGreater(traces[0]["terminal"][0], 430)
        self.assertGreater(crop.x2, 500)

    def test_non_circular_marker_cannot_launch_long_target_trace(self) -> None:
        source = Image.new("L", (600, 300), 255)
        draw = ImageDraw.Draw(source)
        draw.line((80, 130, 110, 160), fill=0, width=2)
        draw.line((80, 160, 110, 130), fill=0, width=2)
        draw.line((120, 150, 450, 165), fill=0, width=3)
        primitives = [
            Primitive("F0", "fai_marker", BBox(80, 130, 110, 160), "test"),
            Primitive(
                "L0",
                "leader_segment",
                BBox(120, 150, 261, 158),
                "test",
                points=[(120, 150), (260, 156)],
            ),
        ]
        initial = BBox(50, 80, 280, 220)
        crop, traces = extend_crop_along_selected_leaders(
            np.asarray(source),
            BBox(0, 0, 300, 300),
            initial,
            primitives,
            ["F0", "L0"],
            (600, 300),
        )
        self.assertEqual(traces, [])
        self.assertEqual(crop.to_int_tuple(), initial.to_int_tuple())


if __name__ == "__main__":
    unittest.main()
