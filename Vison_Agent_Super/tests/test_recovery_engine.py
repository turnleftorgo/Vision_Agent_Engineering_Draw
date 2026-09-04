from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


SUPER_DIR = Path(__file__).resolve().parents[1]
if str(SUPER_DIR) not in sys.path:
    sys.path.insert(0, str(SUPER_DIR))

from recovery.engine import run_crop_recovery  # noqa: E402
from recovery.models import (  # noqa: E402
    CropBox,
    EvidenceBox,
    RecoveryConfig,
    RecoverySkill,
)


def expand_action(right: int = 250) -> str:
    return json.dumps(
        {
            "action": "expand_crop",
            "valid": False,
            "candidate_valid": True,
            "missing": ["target_feature"],
            "arguments": {
                "left_norm": 0,
                "top_norm": 0,
                "right_norm": right,
                "bottom_norm": 0,
            },
            "confidence": 0.85,
        }
    )


def finish_action() -> str:
    return json.dumps(
        {
            "action": "finish",
            "valid": True,
            "candidate_valid": True,
            "missing": [],
            "arguments": {
                "left_norm": 0,
                "top_norm": 0,
                "right_norm": 0,
                "bottom_norm": 0,
            },
            "confidence": 0.9,
        }
    )


def recommendation() -> str:
    return json.dumps(
        {
            "candidate_valid": True,
            "crop_complete": False,
            "missing": ["target_feature"],
            "expand": {"left": 0, "top": 0, "right": 250, "bottom": 0},
            "confidence": 0.8,
        }
    )


def response(content: str, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, reasoning_content="hidden"),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )


class SequenceClient:
    def __init__(self, main_responses: list[str]) -> None:
        self.main_responses = list(main_responses)
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: object) -> SimpleNamespace:
        if not self.main_responses:
            return response(finish_action())
        return response(self.main_responses.pop(0))


class SubagentClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=self)
        self.coordinator_calls = 0
        self.lock = threading.Lock()

    def create(self, **kwargs: object) -> SimpleNamespace:
        response_format = kwargs["response_format"]
        name = response_format["json_schema"]["name"]
        if "recommendation" in name:
            return response(recommendation())
        if "coordinator" in name:
            with self.lock:
                self.coordinator_calls += 1
                number = self.coordinator_calls
            return response(expand_action() if number == 1 else finish_action())
        return response("not json")


class RecoveryEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (800, 600), "white")
        self.marker = CropBox(120, 140, 160, 180)
        self.initial = CropBox(100, 100, 300, 300)
        self.evidence = [
            EvidenceBox("F0", "fai_marker", self.marker, True),
            EvidenceBox("R0", "target_part", CropBox(320, 150, 360, 210), True),
        ]
        self.skill = RecoverySkill(
            Path("SKILL.md"), "fai-crop-recovery", "test", "Follow SOP.", "abc"
        )

    def run_engine(self, client: object, config: RecoveryConfig, directory: Path):
        return run_crop_recovery(
            client,
            "fake-model",
            self.image,
            self.marker,
            self.initial,
            {"complete": False, "missing": ["target_feature"]},
            self.evidence,
            self.skill,
            config,
            directory,
            logger=lambda _message: None,
        )

    def test_expand_then_finish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_engine(
                SequenceClient([expand_action(), finish_action()]),
                RecoveryConfig(
                    max_turns=3,
                    max_subagents=0,
                    max_crop_area_ratio=1.0,
                    max_crop_growth=100.0,
                ),
                Path(directory),
            )
        self.assertTrue(result.valid)
        self.assertEqual(result.status, "validated_from_incomplete_mapping")
        self.assertEqual(len(result.turns), 2)
        self.assertGreater(result.final_crop.x2, self.initial.x2)

    def test_format_retry_does_not_add_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_engine(
                SequenceClient(["not json", expand_action(), finish_action()]),
                RecoveryConfig(
                    max_turns=3,
                    max_format_retries=2,
                    max_subagents=0,
                    max_crop_area_ratio=1.0,
                    max_crop_growth=100.0,
                ),
                Path(directory),
            )
        self.assertEqual(len(result.turns), 2)
        self.assertEqual(len(result.turns[0].format_attempts), 2)

    def test_protocol_failure_never_blindly_expands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_engine(
                SequenceClient(["bad", "bad", "bad"]),
                RecoveryConfig(max_turns=3, max_format_retries=2, max_subagents=0),
                Path(directory),
            )
        self.assertEqual(result.status, "protocol_failed")
        self.assertEqual(result.final_crop.to_int_tuple(), self.initial.to_int_tuple())

    def test_read_only_subagents_are_coordinated_once_per_turn(self) -> None:
        client = SubagentClient()
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_engine(
                client,
                RecoveryConfig(
                    max_turns=2,
                    max_format_retries=0,
                    max_subagents=3,
                    max_crop_area_ratio=1.0,
                    max_crop_growth=100.0,
                ),
                Path(directory),
            )
        self.assertTrue(result.valid)
        self.assertTrue(all(turn.used_subagents for turn in result.turns))
        self.assertEqual(client.coordinator_calls, 2)

    def test_six_observations_allow_at_most_five_expansions(self) -> None:
        client = SequenceClient([expand_action(100) for _ in range(6)])
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_engine(
                client,
                RecoveryConfig(
                    max_turns=6,
                    max_format_retries=0,
                    max_subagents=0,
                    max_crop_area_ratio=1.0,
                    max_crop_growth=100.0,
                ),
                Path(directory),
            )
        expansions = [turn for turn in result.turns if turn.expansion is not None]
        self.assertEqual(len(result.turns), 6)
        self.assertEqual(len(expansions), 5)
        self.assertEqual(result.status, "max_turns_exhausted")


if __name__ == "__main__":
    unittest.main()
