from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import fai_xray


class XRayTests(unittest.TestCase):
    def test_cpu_xray_groups_evidence_by_fai_marker(self) -> None:
        image = Image.new("RGB", (500, 260), "white")
        ocr = [
            fai_xray.OCRLine(fai_xray.Box(40, 40, 70, 55), "FAI", 0.95),
            fai_xray.OCRLine(fai_xray.Box(43, 58, 68, 72), "101", 0.95),
            fai_xray.OCRLine(fai_xray.Box(85, 48, 165, 68), "0.20 +0.05", 0.9),
            fai_xray.OCRLine(fai_xray.Box(320, 40, 350, 55), "FAI", 0.95),
            fai_xray.OCRLine(fai_xray.Box(323, 58, 348, 72), "102", 0.95),
            fai_xray.OCRLine(fai_xray.Box(365, 48, 455, 68), "R 2.950", 0.9),
        ]
        lines = [((165, 58), (230, 120)), ((455, 58), (420, 150))]
        triangles = [fai_xray.Box(224, 114, 236, 126), fai_xray.Box(414, 144, 426, 156)]
        with (
            patch.object(fai_xray, "_run_tesseract", return_value=ocr),
            patch.object(fai_xray, "_detect_fai_bubbles", return_value=[]),
            patch.object(fai_xray, "_detect_lines", return_value=lines),
            patch.object(fai_xray, "_detect_triangles", return_value=triangles),
            patch.object(fai_xray, "_detect_annotation_frames", return_value=[]),
        ):
            result = fai_xray.build_fai_xray(image)

        self.assertEqual([item.cluster_id for item in result.clusters], ["C01", "C02"])
        self.assertEqual([item.fai_number for item in result.clusters], ["101", "102"])
        owned = [line for cluster in result.clusters for line in cluster.leaders]
        self.assertEqual(len(owned), len(set(owned)))
        self.assertEqual(result.diagnostics["cpu_only"], True)
        self.assertEqual(result.diagnostics["locate_call_count"], 0)

    def test_xray_is_saved_for_human_inspection(self) -> None:
        result = fai_xray.XRayResult(
            Image.new("RGB", (20, 20), "white"),
            [],
            {"cpu_only": True},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fai_xray.save_xray_result(result, root / "xray.png", root / "xray.json")
            self.assertTrue((root / "xray.png").is_file())
            self.assertIn('"cpu_only": true', (root / "xray.json").read_text())

if __name__ == "__main__":
    unittest.main()
