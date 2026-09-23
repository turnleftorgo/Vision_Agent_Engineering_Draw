from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import qwen_module_fai_pipeline as pipeline


def decision(
    action: str | None, sides: list[str] | None = None
) -> pipeline.RecoveryDecision:
    reason = None
    if action == "finish":
        reason = "complete"
    elif action is not None:
        reason = "target_clipped"
    return pipeline.RecoveryDecision(
        action=action,
        expand_sides=sides or [],
        reason=reason,
        confidence=0.9 if action is not None else None,
        elapsed_seconds=0.01,
        raw_response="{}",
        metadata={},
        error=None if action is not None else "invalid",
    )


class RecoveryUnitTests(unittest.TestCase):
    def test_prompt_uses_general_annotation_topologies(self) -> None:
        prompt = pipeline.RECOVERY_SYSTEM_PROMPT
        self.assertIn("Direct leader or callout", prompt)
        self.assertIn("Linear, angular, or distance dimension", prompt)
        self.assertIn("Diameter or radius dimension", prompt)
        self.assertIn("no physical target geometry", prompt)
        self.assertNotIn("437", prompt)

    def test_decision_parser_accepts_multiple_expansion_sides(self) -> None:
        parsed = pipeline.parse_recovery_decision(
            json.dumps(
                {
                    "action": "expand",
                    "expand_sides": ["left", "down"],
                    "reason": "target_clipped",
                    "confidence": 0.9,
                }
            ),
            0.1,
            {},
        )
        self.assertEqual(parsed.action, "expand")
        self.assertEqual(parsed.expand_sides, ["left", "down"])

    def test_decision_parser_rejects_finish_with_expansion_sides(self) -> None:
        parsed = pipeline.parse_recovery_decision(
            json.dumps(
                {
                    "action": "finish",
                    "expand_sides": ["left"],
                    "reason": "complete",
                    "confidence": 0.9,
                }
            ),
            0.1,
            {},
        )
        self.assertIsNone(parsed.action)

    def test_local_box_to_global(self) -> None:
        self.assertEqual(
            pipeline.local_box_to_global(
                [1000, 500, 3000, 2500], [200, 300, 800, 900]
            ),
            pipeline.CropBox(1200, 800, 1800, 1400),
        )

    def test_multi_side_expansion_and_clamp(self) -> None:
        box = pipeline.CropBox(10, 10, 110, 210)
        self.assertEqual(
            pipeline.expand_crop_sides(
                box, ["left", "right", "up", "down"], (500, 500), 180
            ),
            pipeline.CropBox(0, 0, 128, 246),
        )

    def test_observation_expands_fifty_percent_and_frames_current_crop(self) -> None:
        observation, context_box = pipeline.build_recovery_observation(
            Image.new("RGB", (400, 300), "white"),
            pipeline.CropBox(100, 80, 300, 220),
            0.5,
            1000,
            pipeline.CropBox(120, 100, 150, 130),
        )
        self.assertEqual(context_box, pipeline.CropBox(0, 10, 400, 290))
        self.assertEqual(observation.size, (400, 280))
        self.assertEqual(observation.getpixel((100, 70)), (220, 0, 220))
        self.assertEqual(observation.getpixel((120, 105)), (220, 0, 0))

    def test_crop2_is_published_immediately_after_save(self) -> None:
        published: list[pipeline.RecoveryCandidate] = []
        cluster = SimpleNamespace(fai_number="205", bbox_pixels=[20, 30, 80, 90])
        with tempfile.TemporaryDirectory() as temp:
            crop2_dir = Path(temp) / "crop2"
            crop2_dir.mkdir()

            def on_crop(candidate: pipeline.RecoveryCandidate) -> None:
                self.assertTrue(candidate.source_crop_path.is_file())
                published.append(candidate)

            paths = pipeline.save_stage2_crops(
                Image.new("RGB", (200, 200), "white"),
                [cluster],
                crop2_dir,
                3,
                [1000, 500, 1200, 700],
                on_crop,
            )
        self.assertEqual(len(paths), 1)
        self.assertEqual(len(published), 1)
        self.assertEqual(
            published[0].initial_box, pipeline.CropBox(1020, 530, 1080, 590)
        )

    def test_multi_side_expand_then_finish(self) -> None:
        rounds = [
            decision("expand", ["left", "down"]),
            decision("finish"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            source_dir = output / "crop2"
            source_dir.mkdir()
            source = source_dir / "candidate.png"
            Image.new("RGB", (100, 100), "white").save(source)
            candidate = pipeline.RecoveryCandidate(
                key="candidate",
                module_index=1,
                cluster_index=1,
                fai_number="205",
                source_crop_path=source,
                initial_box=pipeline.CropBox(100, 100, 200, 200),
            )
            with patch.object(
                pipeline, "request_recovery_decision", side_effect=rounds
            ) as mocked:
                result = pipeline.recover_crop_candidate(
                    client=SimpleNamespace(),
                    model="recovery-model",
                    full_image=Image.new("RGB", (500, 500), "white"),
                    candidate=candidate,
                    output_dir=output,
                    max_rounds=6,
                    max_tokens=128,
                    temperature=0.1,
                    context_fraction=0.5,
                    max_image_edge=2400,
                    step_norm=180,
                )
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["round_count"], 2)
        self.assertEqual(result["final_bbox_full_image"], [82, 100, 200, 218])
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(result["rounds"][0]["crop_after"], [82, 100, 200, 218])

    def test_six_invalid_decisions_exhaust_without_mutation(self) -> None:
        invalid = decision(None)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            source_dir = output / "crop2"
            source_dir.mkdir()
            source = source_dir / "candidate.png"
            Image.new("RGB", (100, 100), "white").save(source)
            candidate = pipeline.RecoveryCandidate(
                key="candidate",
                module_index=1,
                cluster_index=1,
                fai_number="205",
                source_crop_path=source,
                initial_box=pipeline.CropBox(100, 100, 200, 200),
            )
            with patch.object(
                pipeline,
                "request_recovery_decision",
                side_effect=[invalid] * 6,
            ) as mocked:
                result = pipeline.recover_crop_candidate(
                    client=SimpleNamespace(),
                    model="recovery-model",
                    full_image=Image.new("RGB", (500, 500), "white"),
                    candidate=candidate,
                    output_dir=output,
                    max_rounds=6,
                    max_tokens=128,
                    temperature=0.1,
                    context_fraction=0.5,
                    max_image_edge=2400,
                    step_norm=180,
                )
        self.assertEqual(result["status"], "max_rounds_exhausted")
        self.assertEqual(result["round_count"], 6)
        self.assertEqual(result["final_bbox_full_image"], [100, 100, 200, 200])
        self.assertEqual(mocked.call_count, 6)

    def test_request_recovery_decision_calls_model_once(self) -> None:
        class FakeCompletions:
            calls = 0
            last_kwargs: dict[str, object] | None = None

            def create(self, **kwargs: object) -> object:
                self.calls += 1
                self.last_kwargs = kwargs
                message = SimpleNamespace(
                    content=json.dumps(
                        {
                            "action": "finish",
                            "expand_sides": [],
                            "reason": "complete",
                            "confidence": 0.9,
                        }
                    )
                )
                choice = SimpleNamespace(message=message, finish_reason="stop")
                return SimpleNamespace(choices=[choice], usage=None)

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        image = Image.new("RGB", (32, 32), "white")
        result = pipeline.request_recovery_decision(
            client,
            "recovery-model",
            image,
            round_number=1,
            max_tokens=128,
            temperature=0.1,
        )
        self.assertEqual(result.action, "finish")
        self.assertEqual(client.chat.completions.calls, 1)
        self.assertEqual(
            client.chat.completions.last_kwargs["response_format"],
            pipeline.RECOVERY_RESPONSE_FORMAT,
        )

    def test_recovery_default_max_tokens_is_2048(self) -> None:
        args = pipeline.build_parser().parse_args(["drawing.png"])
        self.assertEqual(args.recovery_max_tokens, 2048)


if __name__ == "__main__":
    unittest.main()
