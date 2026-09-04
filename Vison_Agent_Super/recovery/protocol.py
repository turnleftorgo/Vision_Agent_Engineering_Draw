"""Strict JSON schemas and semantic validation for recovery actions."""

from __future__ import annotations

import json
import re
from typing import Any

from .models import (
    AgentCall,
    ExpansionNorm,
    RecoveryAction,
    RecoveryRecommendation,
)


ACTION_FIELDS = {
    "action",
    "valid",
    "candidate_valid",
    "missing",
    "arguments",
    "confidence",
}
ARGUMENT_FIELDS = {"left_norm", "top_norm", "right_norm", "bottom_norm"}
RECOMMENDATION_FIELDS = {
    "candidate_valid",
    "crop_complete",
    "missing",
    "expand",
    "confidence",
}
RECOMMENDATION_EXPAND_FIELDS = {"left", "top", "right", "bottom"}


RECOVERY_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(ACTION_FIELDS),
    "properties": {
        "action": {
            "type": "string",
            "enum": ["expand_crop", "finish", "reject_candidate"],
        },
        "valid": {"type": "boolean"},
        "candidate_valid": {"type": "boolean"},
        "missing": {
            "type": "array",
            "maxItems": 12,
            "items": {"type": "string", "maxLength": 96},
        },
        "arguments": {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(ARGUMENT_FIELDS),
            "properties": {
                name: {"type": "integer", "minimum": 0, "maximum": 500}
                for name in ARGUMENT_FIELDS
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


RECOVERY_RECOMMENDATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": sorted(RECOMMENDATION_FIELDS),
    "properties": {
        "candidate_valid": {"type": "boolean"},
        "crop_complete": {"type": "boolean"},
        "missing": {
            "type": "array",
            "maxItems": 12,
            "items": {"type": "string", "maxLength": 96},
        },
        "expand": {
            "type": "object",
            "additionalProperties": False,
            "required": sorted(RECOMMENDATION_EXPAND_FIELDS),
            "properties": {
                name: {"type": "integer", "minimum": 0, "maximum": 500}
                for name in RECOMMENDATION_EXPAND_FIELDS
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


class ProtocolError(ValueError):
    """Raised when a model response cannot be trusted as an action."""


def _extract_json(text: str) -> Any:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ProtocolError("content_is_not_one_json_object") from exc


def _validate_call(call: AgentCall, max_content_bytes: int) -> dict[str, Any]:
    if call.error:
        raise ProtocolError(f"request_error:{call.error}")
    if call.finish_reason == "length":
        raise ProtocolError("finish_reason:length")
    if not call.content.strip():
        raise ProtocolError("empty_content")
    if len(call.content.encode("utf-8")) > max_content_bytes:
        raise ProtocolError("content_too_long")
    value = _extract_json(call.content)
    if not isinstance(value, dict):
        raise ProtocolError("json_not_object")
    return value


def _exact_fields(value: dict[str, Any], expected: set[str], location: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ProtocolError(f"{location}_fields:missing={missing},extra={extra}")


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ProtocolError(f"{name}_not_boolean")
    return value


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError("confidence_not_number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ProtocolError("confidence_out_of_range")
    return result


def _missing(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > 12:
        raise ProtocolError("missing_not_bounded_array")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or len(item) > 96:
            raise ProtocolError("missing_item_invalid")
        result.append(item.strip())
    return tuple(result)


def _integer(value: Any, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{name}_not_integer")
    if not 0 <= value <= maximum:
        raise ProtocolError(f"{name}_out_of_range")
    return value


def parse_recovery_action(
    call: AgentCall,
    *,
    max_content_bytes: int,
    max_direction_norm: int,
    final_turn: bool,
) -> RecoveryAction:
    value = _validate_call(call, max_content_bytes)
    _exact_fields(value, ACTION_FIELDS, "action")
    action_name = value["action"]
    if action_name not in {"expand_crop", "finish", "reject_candidate"}:
        raise ProtocolError("unknown_action")
    arguments = value["arguments"]
    if not isinstance(arguments, dict):
        raise ProtocolError("arguments_not_object")
    _exact_fields(arguments, ARGUMENT_FIELDS, "arguments")
    expansion = ExpansionNorm(
        left=_integer(arguments["left_norm"], "left_norm", max_direction_norm),
        top=_integer(arguments["top_norm"], "top_norm", max_direction_norm),
        right=_integer(arguments["right_norm"], "right_norm", max_direction_norm),
        bottom=_integer(arguments["bottom_norm"], "bottom_norm", max_direction_norm),
    )
    action = RecoveryAction(
        action=action_name,
        valid=_bool(value["valid"], "valid"),
        candidate_valid=_bool(value["candidate_valid"], "candidate_valid"),
        missing=_missing(value["missing"]),
        arguments=expansion,
        confidence=_confidence(value["confidence"]),
    )

    if action.action == "expand_crop":
        if final_turn:
            raise ProtocolError("expand_crop_not_allowed_on_final_turn")
        if action.valid or not action.candidate_valid or expansion.total() <= 0:
            raise ProtocolError("expand_crop_cross_field_violation")
    elif action.action == "finish":
        if (
            not action.valid
            or not action.candidate_valid
            or action.missing
            or expansion.total() != 0
        ):
            raise ProtocolError("finish_cross_field_violation")
    else:
        if action.valid or action.candidate_valid or expansion.total() != 0:
            raise ProtocolError("reject_cross_field_violation")
    return action


def parse_recommendation(
    call: AgentCall,
    *,
    role: str,
    max_content_bytes: int,
    max_direction_norm: int,
) -> RecoveryRecommendation:
    value = _validate_call(call, max_content_bytes)
    _exact_fields(value, RECOMMENDATION_FIELDS, "recommendation")
    expansion_value = value["expand"]
    if not isinstance(expansion_value, dict):
        raise ProtocolError("expand_not_object")
    _exact_fields(expansion_value, RECOMMENDATION_EXPAND_FIELDS, "expand")
    expansion = ExpansionNorm(
        left=_integer(expansion_value["left"], "left", max_direction_norm),
        top=_integer(expansion_value["top"], "top", max_direction_norm),
        right=_integer(expansion_value["right"], "right", max_direction_norm),
        bottom=_integer(expansion_value["bottom"], "bottom", max_direction_norm),
    )
    recommendation = RecoveryRecommendation(
        candidate_valid=_bool(value["candidate_valid"], "candidate_valid"),
        crop_complete=_bool(value["crop_complete"], "crop_complete"),
        missing=_missing(value["missing"]),
        expand=expansion,
        confidence=_confidence(value["confidence"]),
        role=role,
    )
    if recommendation.crop_complete and (
        not recommendation.candidate_valid
        or recommendation.missing
        or expansion.total() != 0
    ):
        raise ProtocolError("complete_recommendation_violation")
    if not recommendation.candidate_valid and expansion.total() != 0:
        raise ProtocolError("rejected_recommendation_expands")
    if (
        recommendation.candidate_valid
        and not recommendation.crop_complete
        and expansion.total() <= 0
    ):
        raise ProtocolError("incomplete_recommendation_without_expansion")
    return recommendation


def response_format(schema_name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": schema_name, "strict": True, "schema": schema},
    }
