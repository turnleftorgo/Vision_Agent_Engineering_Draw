from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


SUPER_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = SUPER_DIR.parent
for path in (str(SUPER_DIR), str(PROJECT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import FAI_DET_CROP_3 as core  # noqa: E402
from semantic import (  # noqa: E402
    build_compact_semantic_evidence,
    compact_semantic_prompt,
    run_compact_semantic_mapping,
)


def primitive(
    item_id: str,
    kind: str,
    bbox: tuple[int, int, int, int],
    *,
    text: str = "",
    points: list[tuple[int, int]] | None = None,
) -> core.Primitive:
    return core.Primitive(
        item_id,
        kind,
        core.BBox(*bbox),
        "test",
        text=text,
        points=points or [],
    )


class CaptureClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=self.content,
                        reasoning_content="private reasoning",
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
            ),
        )


class CompactSemanticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.roi = Image.new("RGB", (600, 400), "white")
        self.primitives = [
            primitive("F0", "fai_marker", (70, 90, 120, 140)),
            primitive("A0", "annotation", (135, 90, 240, 140)),
            primitive("A1", "annotation", (390, 35, 550, 75)),
            primitive("T0", "ocr_text", (140, 95, 235, 135), text="0.25 ±0.05"),
            primitive("T1", "ocr_text", (390, 35, 550, 75), text="DIMENSIONS"),
            # Text/frame stroke: it must not become a leader path.
            primitive(
                "L0",
                "leader_segment",
                (392, 49, 548, 51),
                points=[(392, 50), (548, 50)],
            ),
            # Duplicate Hough hits plus a connected bend form one compact path.
            primitive(
                "L1",
                "leader_segment",
                (239, 114, 341, 116),
                points=[(240, 115), (340, 115)],
            ),
            primitive(
                "L2",
                "leader_segment",
                (240, 116, 340, 118),
                points=[(240, 117), (340, 117)],
            ),
            primitive(
                "L3",
                "leader_segment",
                (339, 114, 401, 181),
                points=[(340, 115), (400, 180)],
            ),
            primitive(
                "L4",
                "leader_segment",
                (500, 330, 580, 332),
                points=[(500, 331), (580, 331)],
            ),
            primitive("G0", "triangle", (392, 172, 408, 188)),
            primitive("R0", "target_part", (380, 165, 440, 225)),
        ]

    def test_builds_clean_compact_records_and_merged_paths(self) -> None:
        evidence = build_compact_semantic_evidence(self.roi, self.primitives)
        record_types = [record["type"] for record in evidence.records]

        self.assertNotIn(
            "L0", {item for path in evidence.paths for item in path.segment_ids}
        )
        self.assertLess(record_types.count("leader_path"), 4)
        self.assertTrue(any(len(path.segments) >= 2 for path in evidence.paths))
        self.assertNotIn("T1", {record["id"] for record in evidence.records})
        for record in evidence.records:
            self.assertTrue({"bbox", "points", "source"}.isdisjoint(record))
        self.assertEqual(evidence.image.size, self.roi.size)

    def test_prompt_is_one_image_id_only_protocol(self) -> None:
        evidence = build_compact_semantic_evidence(self.roi, self.primitives)
        prompt = compact_semantic_prompt(evidence.records)

        self.assertIn("exactly ONE", prompt)
        self.assertIn("LP*: merged leader-path", prompt)
        self.assertIn("Never output or invent coordinates", prompt)
        self.assertNotIn('"bbox"', prompt)
        self.assertNotIn('"points"', prompt)

    def test_model_receives_one_image_and_lp_maps_back_to_source_lines(self) -> None:
        evidence = build_compact_semantic_evidence(self.roi, self.primitives)
        path = next(path for path in evidence.paths if len(path.segments) >= 2)
        content = json.dumps(
            {
                "marker_id": "F0",
                "candidate_valid": True,
                "fai_number": "12",
                "spc_letter": None,
                "annotation_ids": ["A0"],
                "text_ids": ["T0"],
                "leader_path_ids": [path.id],
                "arrowhead_ids": ["G0"],
                "target_ids": ["R0"],
                "complete": True,
                "missing": [],
                "confidence": 0.91,
            }
        )
        client = CaptureClient(content)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = run_compact_semantic_mapping(
                client,
                "fake-model",
                evidence,
                root / "mapping.txt",
                root / "mapping_meta.json",
            )
            metadata = json.loads((root / "mapping_meta.json").read_text())

        call = client.calls[0]
        user_content = call["messages"][1]["content"]  # type: ignore[index]
        image_parts = [part for part in user_content if part["type"] == "image_url"]
        self.assertEqual(len(image_parts), 1)
        self.assertTrue(call["response_format"]["json_schema"]["strict"])  # type: ignore[index]
        self.assertEqual(mapping["leader_path_ids"], [path.id])
        self.assertEqual(mapping["leader_ids"], list(path.segment_ids))
        self.assertEqual(metadata["image_count"], 1)


if __name__ == "__main__":
    unittest.main()
