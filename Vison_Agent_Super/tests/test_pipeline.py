from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image


SUPER_DIR = Path(__file__).resolve().parents[1]
if str(SUPER_DIR) not in sys.path:
    sys.path.insert(0, str(SUPER_DIR))

from fai_DET_crop_5 import build_parser, run_streaming_candidates  # noqa: E402
from vision.models import BBox, Primitive  # noqa: E402


class StreamingPipelineTests(unittest.TestCase):
    def test_default_model_limits_are_locate_16_and_qwen_4(self) -> None:
        args = build_parser().parse_args(["input.png"])
        self.assertEqual(args.concurrent, 16)
        self.assertEqual(args.qwen_concurrent, 4)

    def test_stages_obey_dependencies_and_use_isolated_pools(self) -> None:
        events: list[tuple[str, int, str]] = []
        lock = threading.Lock()

        def mark(stage: str, index: int) -> None:
            with lock:
                events.append((stage, index, threading.current_thread().name))

        def annotation(*args: object) -> list[BBox]:
            index = int(args[-1])
            mark("annotation", index)
            return [BBox(1, 1, 5, 5)]

        def arrow(*args: object) -> list[BBox]:
            index = int(args[-1])
            mark("arrowhead", index)
            return [BBox(6, 6, 10, 10)]

        def target(*args: object) -> list[BBox]:
            index = int(args[-1])
            mark("target", index)
            return [BBox(11, 11, 15, 15)]

        def evidence(*args: object) -> tuple[Image.Image, list[Primitive], Image.Image]:
            index = int(args[-2])
            mark("evidence", index)
            roi = args[0]
            assert isinstance(roi, Image.Image)
            return roi, [Primitive("F0", "fai_marker", BBox(1, 1, 4, 4), "test")], roi

        def semantic(*args: object) -> object:
            index = int(args[5])
            mark("semantic", index)
            return object()

        def recovery(*args: object) -> dict[str, object]:
            prepared = args[3]
            index = next(i for i, state in states_seen.items() if state is prepared)
            mark("recovery", index)
            return {
                "candidate_index": index,
                "status": "verification_skipped",
                "final_crop_path": f"candidate_{index}.png",
            }

        # Capture the semantic objects so the recovery mock can identify each candidate.
        states_seen: dict[int, object] = {}

        def semantic_with_capture(*args: object) -> object:
            index = int(args[5])
            result = semantic(*args)
            states_seen[index] = result
            return result

        args = SimpleNamespace(
            concurrent=2,
            qwen_concurrent=1,
            locate_model="locate-test",
            no_tesseract=True,
        )
        image = Image.new("RGB", (100, 100), "white")
        gray = np.asarray(image.convert("L"))
        markers = [BBox(10, 10, 20, 20), BBox(60, 60, 70, 70)]

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            with (
                patch("fai_DET_crop_5.run_annotation_proposal", side_effect=annotation),
                patch("fai_DET_crop_5.run_arrowhead_proposal", side_effect=arrow),
                patch("fai_DET_crop_5.run_target_proposal", side_effect=target),
                patch("fai_DET_crop_5.assemble_candidate_evidence", side_effect=evidence),
                patch(
                    "fai_DET_crop_5.run_candidate_semantic_stage",
                    side_effect=semantic_with_capture,
                ),
                patch(
                    "fai_DET_crop_5.run_candidate_recovery_stage",
                    side_effect=recovery,
                ),
            ):
                records = run_streaming_candidates(
                    args,
                    object(),
                    image,
                    gray,
                    markers,
                    root / "raw",
                    root / "crop",
                    root / "related",
                    root / "expand",
                    None,
                )

        positions = {(stage, index): pos for pos, (stage, index, _) in enumerate(events)}
        for index in range(2):
            self.assertLess(positions[("arrowhead", index)], positions[("target", index)])
            self.assertLess(positions[("annotation", index)], positions[("evidence", index)])
            self.assertLess(positions[("target", index)], positions[("evidence", index)])
            self.assertLess(positions[("evidence", index)], positions[("semantic", index)])
            self.assertLess(positions[("semantic", index)], positions[("recovery", index)])

        threads = {stage: name for stage, _, name in events}
        self.assertIn("fai-locate", threads["annotation"])
        self.assertIn("fai-evidence", threads["evidence"])
        self.assertIn("fai-qwen", threads["semantic"])
        self.assertEqual([record["candidate_index"] for record in records], [0, 1])


if __name__ == "__main__":
    unittest.main()
