from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


SUPER_DIR = Path(__file__).resolve().parents[1]
if str(SUPER_DIR) not in sys.path:
    sys.path.insert(0, str(SUPER_DIR))

from recovery.models import AgentCall  # noqa: E402
from recovery.protocol import (  # noqa: E402
    ProtocolError,
    parse_recommendation,
    parse_recovery_action,
)


def call(value: object, finish_reason: str = "stop") -> AgentCall:
    content = value if isinstance(value, str) else json.dumps(value)
    return AgentCall(content, "", finish_reason, 0.01)


class ProtocolTests(unittest.TestCase):
    def test_valid_expand(self) -> None:
        action = parse_recovery_action(
            call(
                {
                    "action": "expand_crop",
                    "valid": False,
                    "candidate_valid": True,
                    "missing": ["target_feature"],
                    "arguments": {
                        "left_norm": 0,
                        "top_norm": 0,
                        "right_norm": 250,
                        "bottom_norm": 0,
                    },
                    "confidence": 0.82,
                }
            ),
            max_content_bytes=4096,
            max_direction_norm=500,
            final_turn=False,
        )
        self.assertEqual(action.action, "expand_crop")
        self.assertEqual(action.arguments.right, 250)

    def test_finish_requires_empty_missing_and_zero_expansion(self) -> None:
        value = {
            "action": "finish",
            "valid": True,
            "candidate_valid": True,
            "missing": ["arrow"],
            "arguments": {
                "left_norm": 0,
                "top_norm": 0,
                "right_norm": 0,
                "bottom_norm": 0,
            },
            "confidence": 0.9,
        }
        with self.assertRaisesRegex(ProtocolError, "finish_cross_field"):
            parse_recovery_action(
                call(value),
                max_content_bytes=4096,
                max_direction_norm=500,
                final_turn=False,
            )

    def test_final_turn_rejects_expansion(self) -> None:
        value = {
            "action": "expand_crop",
            "valid": False,
            "candidate_valid": True,
            "missing": ["target"],
            "arguments": {
                "left_norm": 100,
                "top_norm": 0,
                "right_norm": 0,
                "bottom_norm": 0,
            },
            "confidence": 0.8,
        }
        with self.assertRaisesRegex(ProtocolError, "final_turn"):
            parse_recovery_action(
                call(value),
                max_content_bytes=4096,
                max_direction_norm=500,
                final_turn=True,
            )

    def test_extra_prose_is_not_salvaged(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "one_json_object"):
            parse_recovery_action(
                call('analysis first {"action":"finish"}'),
                max_content_bytes=4096,
                max_direction_norm=500,
                final_turn=False,
            )

    def test_length_finish_reason_is_protocol_failure(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "finish_reason:length"):
            parse_recovery_action(
                call("{}", "length"),
                max_content_bytes=4096,
                max_direction_norm=500,
                final_turn=False,
            )

    def test_valid_subagent_recommendation(self) -> None:
        recommendation = parse_recommendation(
            call(
                {
                    "candidate_valid": True,
                    "crop_complete": False,
                    "missing": ["leader"],
                    "expand": {"left": 0, "top": 0, "right": 300, "bottom": 0},
                    "confidence": 0.75,
                }
            ),
            role="geometry_tracer",
            max_content_bytes=4096,
            max_direction_norm=500,
        )
        self.assertEqual(recommendation.role, "geometry_tracer")
        self.assertEqual(recommendation.expand.right, 300)


if __name__ == "__main__":
    unittest.main()
