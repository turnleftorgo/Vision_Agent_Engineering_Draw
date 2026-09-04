"""Model-facing recovery agents. They can propose actions but never mutate crops."""

from __future__ import annotations

import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable

from PIL import Image

from .models import (
    AgentCall,
    RecoveryConfig,
    RecoveryObservation,
    RecoveryRecommendation,
    RecoverySkill,
)
from .protocol import (
    RECOVERY_ACTION_SCHEMA,
    RECOVERY_RECOMMENDATION_SCHEMA,
    response_format,
)


SUBAGENT_ROLES: tuple[tuple[str, str], ...] = (
    (
        "completeness_auditor",
        "Audit whether FAI/SPC, parameter/tolerance text, every leader, every "
        "terminal arrowhead, and each touched target feature are present.",
    ),
    (
        "geometry_tracer",
        "Trace leader geometry segment by segment and determine the minimum "
        "required expansion direction and amount.",
    ),
    (
        "adversarial_verifier",
        "Try to disprove the candidate: detect false FAI markers, neighboring "
        "group contamination, or an already excessive crop.",
    ),
)


def _data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _reasoning_content(message: Any) -> str:
    value = getattr(message, "reasoning_content", None)
    if value is None:
        extra = getattr(message, "model_extra", None)
        if isinstance(extra, dict):
            value = extra.get("reasoning_content")
    return value if isinstance(value, str) else ""


def _usage_metadata(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    result: dict[str, Any] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, name, None) if usage is not None else None
        result[name] = int(value) if isinstance(value, (int, float)) else None
    return result


def call_structured_agent(
    client: Any,
    model: str,
    *,
    system_prompt: str,
    user_prompt: str,
    images: Iterable[Image.Image],
    schema_name: str,
    schema: dict[str, Any],
    max_tokens: int,
) -> AgentCall:
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    content.extend(
        {"type": "image_url", "image_url": {"url": _data_url(image)}}
        for image in images
    )
    started = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            max_tokens=max_tokens,
            temperature=0.1,
            response_format=response_format(schema_name, schema),
        )
        choice = response.choices[0]
        message = choice.message
        raw = message.content or ""
        if not isinstance(raw, str):
            raw = json.dumps(raw, ensure_ascii=False)
        return AgentCall(
            content=raw.strip(),
            reasoning=_reasoning_content(message),
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            elapsed_seconds=time.monotonic() - started,
            metadata=_usage_metadata(response),
        )
    except Exception as exc:
        return AgentCall(
            content="",
            reasoning="",
            finish_reason="error",
            elapsed_seconds=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def _mapping_context(mapping: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "fai_number",
        "spc_letter",
        "parameter_values",
        "measurement_description",
        "complete",
        "missing",
    )
    return {key: mapping.get(key) for key in keys}


def _observation_prompt(
    observation: RecoveryObservation,
    mapping: dict[str, Any],
    history: list[dict[str, Any]],
    *,
    max_turns: int,
    final_turn: bool,
    repair_error: str = "",
) -> str:
    state = {
        "turn": observation.turn,
        "max_turns": max_turns,
        "final_turn": final_turn,
        "crop_bbox_full_image": observation.crop_bbox.to_list(),
        "context_bbox_full_image": observation.context_bbox.to_list(),
        "boundary_locked": observation.boundary_locked,
        "semantic_hypothesis": _mapping_context(mapping),
        "prior_turns": history[-5:],
    }
    repair = ""
    if repair_error:
        repair = (
            "\nYour previous response failed protocol validation: "
            f"{repair_error}. Reinspect the same unchanged observation and return "
            "only one schema-conforming JSON object.\n"
        )
    final_rule = (
        "This is the final observation. expand_crop is forbidden; return finish "
        "only if complete or reject_candidate only if the red marker is false."
        if final_turn
        else "Request at most one bounded expansion if related evidence is clipped."
    )
    return f"""IMAGE 1 is a fresh wider context from the full drawing. The MAGENTA
rectangle is the current crop and the RED rectangle is the selected marker.
IMAGE 2 is the clean current crop.

Treat semantic information as a hypothesis, never as ground truth. Inspect the
pixels using the injected recovery SOP. {final_rule}
{repair}
STATE:
{json.dumps(state, ensure_ascii=False, separators=(",", ":"))}

Return exactly one action JSON matching the supplied schema. Do not put analysis,
Markdown, XML tags, or tool narration in final content."""


def request_main_action(
    client: Any,
    model: str,
    skill: RecoverySkill,
    observation: RecoveryObservation,
    mapping: dict[str, Any],
    history: list[dict[str, Any]],
    config: RecoveryConfig,
    *,
    final_turn: bool,
    repair_error: str = "",
) -> AgentCall:
    system = (
        "You are the primary FAI crop recovery agent. You can recommend exactly "
        "one action; Python owns all coordinates and state changes.\n\n" + skill.body
    )
    return call_structured_agent(
        client,
        model,
        system_prompt=system,
        user_prompt=_observation_prompt(
            observation,
            mapping,
            history,
            max_turns=config.max_turns,
            final_turn=final_turn,
            repair_error=repair_error,
        ),
        images=[observation.context_image, observation.crop_image],
        schema_name="fai_crop_recovery_action_v1",
        schema=RECOVERY_ACTION_SCHEMA,
        max_tokens=config.recovery_max_tokens,
    )


def _subagent_call(
    client: Any,
    model: str,
    skill: RecoverySkill,
    observation: RecoveryObservation,
    mapping: dict[str, Any],
    role: str,
    role_instruction: str,
    config: RecoveryConfig,
) -> AgentCall:
    state = {
        "turn": observation.turn,
        "crop_bbox_full_image": observation.crop_bbox.to_list(),
        "context_bbox_full_image": observation.context_bbox.to_list(),
        "boundary_locked": observation.boundary_locked,
        "semantic_hypothesis": _mapping_context(mapping),
    }
    system = (
        f"You are the read-only {role} for FAI crop recovery. {role_instruction} "
        "You cannot call tools or change coordinates; submit one recommendation.\n\n"
        + skill.body
    )
    prompt = f"""Inspect the same immutable observation used by the coordinator.
IMAGE 1 is dynamic wider context; IMAGE 2 is the clean crop.
STATE: {json.dumps(state, ensure_ascii=False, separators=(",", ":"))}
Return exactly one recommendation JSON matching the supplied schema."""
    return call_structured_agent(
        client,
        model,
        system_prompt=system,
        user_prompt=prompt,
        images=[observation.context_image, observation.crop_image],
        schema_name="fai_crop_recovery_recommendation_v1",
        schema=RECOVERY_RECOMMENDATION_SCHEMA,
        max_tokens=config.recovery_max_tokens,
    )


def request_subagent_opinions(
    client: Any,
    model: str,
    skill: RecoverySkill,
    observation: RecoveryObservation,
    mapping: dict[str, Any],
    config: RecoveryConfig,
) -> dict[str, AgentCall]:
    roles = SUBAGENT_ROLES[: config.max_subagents]
    if not roles:
        return {}
    results: dict[str, AgentCall] = {}
    with ThreadPoolExecutor(max_workers=len(roles)) as executor:
        futures = {
            executor.submit(
                _subagent_call,
                client,
                model,
                skill,
                observation,
                mapping,
                role,
                instruction,
                config,
            ): role
            for role, instruction in roles
        }
        for future in as_completed(futures):
            role = futures[future]
            try:
                results[role] = future.result()
            except Exception as exc:
                results[role] = AgentCall(
                    content="",
                    reasoning="",
                    finish_reason="error",
                    elapsed_seconds=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
    return results


def request_coordinator_action(
    client: Any,
    model: str,
    skill: RecoverySkill,
    observation: RecoveryObservation,
    recommendations: list[RecoveryRecommendation],
    config: RecoveryConfig,
    *,
    final_turn: bool,
    repair_error: str = "",
) -> AgentCall:
    system = (
        "You are the FAI crop recovery coordinator. Reconcile read-only expert "
        "recommendations into exactly one action. Python alone executes it.\n\n"
        + skill.body
    )
    state = {
        "turn": observation.turn,
        "final_turn": final_turn,
        "crop_bbox_full_image": observation.crop_bbox.to_list(),
        "boundary_locked": observation.boundary_locked,
        "recommendations": [item.to_dict() for item in recommendations],
    }
    repair = f" Previous output error: {repair_error}." if repair_error else ""
    final_rule = "Expansion is forbidden on this final turn." if final_turn else ""
    prompt = f"""IMAGE 1 is wider context and IMAGE 2 is the current crop.
Reconcile the structured opinions below. Do not average away a credible geometry
warning. {final_rule}{repair}
STATE: {json.dumps(state, ensure_ascii=False, separators=(",", ":"))}
Return only one action JSON matching the supplied schema."""
    return call_structured_agent(
        client,
        model,
        system_prompt=system,
        user_prompt=prompt,
        images=[observation.context_image, observation.crop_image],
        schema_name="fai_crop_recovery_coordinator_action_v1",
        schema=RECOVERY_ACTION_SCHEMA,
        max_tokens=config.recovery_max_tokens,
    )
