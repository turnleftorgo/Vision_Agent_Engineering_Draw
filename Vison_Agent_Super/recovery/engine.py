"""Single-owner observe/decide/act loop for FAI crop recovery."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image

from .agents import (
    request_coordinator_action,
    request_main_action,
    request_subagent_opinions,
)
from .crop_tool import expand_crop
from .expansion_frames import save_expansion_frame
from .models import (
    AgentCall,
    CropBox,
    EvidenceBox,
    ExpansionNorm,
    ExpansionResult,
    RecoveryAction,
    RecoveryConfig,
    RecoveryRecommendation,
    RecoveryResult,
    RecoverySkill,
    RecoveryTurn,
)
from .observation import build_dynamic_observation
from .protocol import ProtocolError, parse_recommendation, parse_recovery_action


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_call(
    directory: Path,
    stem: str,
    call: AgentCall,
    *,
    debug: bool,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.txt").write_text(call.content, encoding="utf-8")
    _save_json(directory / f"{stem}_meta.json", call.to_metadata())
    if debug and call.reasoning:
        (directory / f"{stem}_reasoning.txt").write_text(
            call.reasoning, encoding="utf-8"
        )


def _relevant_evidence(evidence: list[EvidenceBox]) -> list[EvidenceBox]:
    selected = [item for item in evidence if item.selected or item.kind == "fai_marker"]
    return selected if len(selected) > 1 else evidence


def score_crop_state(
    crop: CropBox,
    evidence: list[EvidenceBox],
    full_size: tuple[int, int],
) -> float:
    """Rank history deterministically; model confidence is deliberately absent."""
    weights = {
        "fai_marker": 6.0,
        "annotation": 2.5,
        "ocr_text": 1.5,
        "leader_segment": 1.0,
        "arrowhead": 3.0,
        "triangle": 2.5,
        "target_part": 3.0,
    }
    relevant = _relevant_evidence(evidence)
    total_weight = 0.0
    covered_weight = 0.0
    border_penalty = 0.0
    margin = max(2.0, min(crop.width, crop.height) * 0.012)
    for item in relevant:
        weight = weights.get(item.kind, 1.0)
        total_weight += weight
        if item.bbox.area <= 0:
            continue
        fraction = min(1.0, crop.intersection_area(item.bbox) / item.bbox.area)
        covered_weight += weight * fraction
        if fraction > 0 and not crop.contains_box(item.bbox, margin=margin):
            border_penalty += weight
    coverage = covered_weight / total_weight if total_weight else 0.0
    full_area = max(1.0, float(full_size[0] * full_size[1]))
    area_penalty = crop.area / full_area * 18.0
    return coverage * 100.0 - border_penalty * 2.5 - area_penalty


def _evidence_touches_border(crop: CropBox, evidence: list[EvidenceBox]) -> bool:
    margin = max(3.0, min(crop.width, crop.height) * 0.015)
    for item in _relevant_evidence(evidence):
        if item.kind == "fai_marker":
            continue
        intersection = crop.intersection_area(item.bbox)
        if intersection > 0 and not crop.contains_box(item.bbox, margin=margin):
            return True
    return False


def _dominant_direction(expansion: ExpansionNorm) -> str:
    values = {
        "left": expansion.left,
        "top": expansion.top,
        "right": expansion.right,
        "bottom": expansion.bottom,
    }
    name, value = max(values.items(), key=lambda item: item[1])
    return name if value > 0 else ""


def _direction_conflict(current: ExpansionNorm, prior_direction: str) -> bool:
    opposite = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
    direction = _dominant_direction(current)
    return bool(
        direction and prior_direction and opposite.get(direction) == prior_direction
    )


def should_escalate_to_subagents(
    action: Optional[RecoveryAction],
    *,
    evidence_border_warning: bool,
    prior_direction: str,
    stuck_count: int,
    confidence_threshold: float,
) -> bool:
    if action is None:
        return True
    if stuck_count >= 1:
        return True
    if action.confidence < confidence_threshold and action.missing:
        return True
    if action.action == "finish" and evidence_border_warning:
        return True
    if action.action == "expand_crop" and _direction_conflict(
        action.arguments, prior_direction
    ):
        return True
    return False


def _weighted_median(values: list[tuple[int, float]]) -> int:
    if not values:
        return 0
    ordered = sorted(values, key=lambda item: item[0])
    total = sum(max(0.01, weight) for _, weight in ordered)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += max(0.01, weight)
        if cumulative >= total / 2.0:
            return value
    return ordered[-1][0]


def deterministic_emergency_arbitration(
    recommendations: list[RecoveryRecommendation],
    *,
    final_turn: bool,
) -> Optional[RecoveryAction]:
    """Last-resort aggregation; never invent evidence or blind expansion."""
    strong_rejects = [
        item
        for item in recommendations
        if not item.candidate_valid and item.confidence >= 0.75
    ]
    if len(strong_rejects) >= 2:
        return RecoveryAction(
            action="reject_candidate",
            valid=False,
            candidate_valid=False,
            missing=tuple(
                dict.fromkeys(
                    reason for item in strong_rejects for reason in item.missing
                )
            ),
            arguments=ExpansionNorm(),
            confidence=statistics.fmean(item.confidence for item in strong_rejects),
        )

    strong_complete = [
        item
        for item in recommendations
        if item.candidate_valid and item.crop_complete and item.confidence >= 0.70
    ]
    adversarial_reject = any(
        item.role == "adversarial_verifier"
        and not item.candidate_valid
        and item.confidence >= 0.70
        for item in recommendations
    )
    if len(strong_complete) >= 2 and not adversarial_reject:
        return RecoveryAction(
            action="finish",
            valid=True,
            candidate_valid=True,
            missing=(),
            arguments=ExpansionNorm(),
            confidence=statistics.fmean(item.confidence for item in strong_complete),
        )

    expandable = [
        item
        for item in recommendations
        if item.candidate_valid and not item.crop_complete and item.expand.total() > 0
    ]
    if final_turn or not expandable:
        return None
    expansion = ExpansionNorm(
        left=_weighted_median(
            [(item.expand.left, item.confidence) for item in expandable]
        ),
        top=_weighted_median(
            [(item.expand.top, item.confidence) for item in expandable]
        ),
        right=_weighted_median(
            [(item.expand.right, item.confidence) for item in expandable]
        ),
        bottom=_weighted_median(
            [(item.expand.bottom, item.confidence) for item in expandable]
        ),
    )
    if expansion.total() <= 0:
        return None
    return RecoveryAction(
        action="expand_crop",
        valid=False,
        candidate_valid=True,
        missing=tuple(
            dict.fromkeys(reason for item in expandable for reason in item.missing)
        ),
        arguments=expansion,
        confidence=statistics.fmean(item.confidence for item in expandable),
    )


def _parse_action_attempts(
    request: Callable[[str], AgentCall],
    directory: Path,
    stem: str,
    config: RecoveryConfig,
    *,
    final_turn: bool,
    debug: bool,
) -> tuple[Optional[RecoveryAction], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    repair_error = ""
    for attempt in range(config.max_format_retries + 1):
        call = request(repair_error)
        attempt_stem = f"{stem}_attempt_{attempt:02d}"
        _save_call(directory, attempt_stem, call, debug=debug)
        record = {"attempt": attempt, **call.to_metadata(), "protocol_error": None}
        try:
            action = parse_recovery_action(
                call,
                max_content_bytes=config.max_content_bytes,
                max_direction_norm=config.max_direction_norm,
                final_turn=final_turn,
            )
            attempts.append(record)
            return action, attempts
        except ProtocolError as exc:
            repair_error = str(exc)
            record["protocol_error"] = repair_error
            attempts.append(record)
    return None, attempts


def _subagent_decision(
    client: Any,
    model: str,
    skill: RecoverySkill,
    observation: Any,
    mapping: dict[str, Any],
    config: RecoveryConfig,
    directory: Path,
    *,
    final_turn: bool,
    debug: bool,
) -> tuple[Optional[RecoveryAction], dict[str, Any], str]:
    calls = request_subagent_opinions(
        client, model, skill, observation, mapping, config
    )
    recommendations: list[RecoveryRecommendation] = []
    record: dict[str, Any] = {}
    for role, call in sorted(calls.items()):
        _save_call(directory, f"subagent_{role}", call, debug=debug)
        role_record = {
            "call": call.to_metadata(),
            "protocol_error": None,
            "result": None,
        }
        try:
            recommendation = parse_recommendation(
                call,
                role=role,
                max_content_bytes=config.max_content_bytes,
                max_direction_norm=config.max_direction_norm,
            )
            recommendations.append(recommendation)
            role_record["result"] = recommendation.to_dict()
        except ProtocolError as exc:
            role_record["protocol_error"] = str(exc)
        record[role] = role_record

    if recommendations:
        coordinator, attempts = _parse_action_attempts(
            lambda repair: request_coordinator_action(
                client,
                model,
                skill,
                observation,
                recommendations,
                config,
                final_turn=final_turn,
                repair_error=repair,
            ),
            directory,
            "coordinator",
            config,
            final_turn=final_turn,
            debug=debug,
        )
        record["coordinator_attempts"] = attempts
        if coordinator is not None:
            return coordinator, record, "coordinator"

    emergency = deterministic_emergency_arbitration(
        recommendations, final_turn=final_turn
    )
    if emergency is not None:
        record["emergency_action"] = emergency.to_dict()
        return emergency, record, "deterministic_arbitration"
    return None, record, "subagents_failed"


def _history_summary(turns: list[RecoveryTurn]) -> list[dict[str, Any]]:
    return [
        {
            "turn": turn.turn,
            "action": turn.action.action if turn.action else None,
            "missing": list(turn.action.missing) if turn.action else [],
            "crop_after": turn.crop_after.to_list(),
            "changed": turn.expansion.changed if turn.expansion else False,
        }
        for turn in turns[-5:]
    ]


def _save_history(
    path: Path,
    *,
    skill: RecoverySkill,
    status: str,
    initial_crop: CropBox,
    current_crop: CropBox,
    best_crop: CropBox,
    best_score: float,
    turns: list[RecoveryTurn],
) -> None:
    _save_json(
        path,
        {
            "skill": {
                "path": str(skill.path),
                "name": skill.name,
                "sha256": skill.sha256,
            },
            "status": status,
            "initial_crop": initial_crop.to_list(),
            "current_crop": current_crop.to_list(),
            "best_crop": best_crop.to_list(),
            "best_score": round(best_score, 4),
            "turns": [turn.to_dict() for turn in turns],
        },
    )


def run_crop_recovery(
    client: Any,
    model: str,
    full_image: Image.Image,
    marker_box: CropBox,
    initial_crop: CropBox,
    mapping: dict[str, Any],
    evidence: list[EvidenceBox],
    skill: RecoverySkill,
    config: RecoveryConfig,
    output_dir: Path,
    *,
    debug: bool = False,
    logger: Callable[[str], None] = print,
    expansion_output_dir: Optional[Path] = None,
) -> RecoveryResult:
    """Run at most six observations and five crop mutations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "skill_snapshot.md").write_text(
        f"---\nname: {skill.name}\nsha256: {skill.sha256}\n---\n\n{skill.body}\n",
        encoding="utf-8",
    )
    full_size = full_image.size
    initial = initial_crop.clamp(*full_size)
    current = initial
    initial_area = initial.area
    best = current
    best_score = score_crop_state(current, evidence, full_size)
    turns: list[RecoveryTurn] = []
    previous_expansion: Optional[ExpansionResult] = None
    prior_direction = ""
    stuck_count = 0
    expansion_count = 0
    status = "max_turns_exhausted"
    valid = False
    rejected = False
    history_path = output_dir / "history.json"

    for turn_number in range(1, config.max_turns + 1):
        final_turn = turn_number == config.max_turns
        logger(
            f"[recovery] observation {turn_number}/{config.max_turns} "
            f"crop={current.to_list()}"
        )
        observation = build_dynamic_observation(
            full_image,
            current,
            marker_box,
            turn=turn_number,
            context_fraction=config.context_fraction,
            max_image_edge=config.max_image_edge,
            previous_expansion=previous_expansion,
        )
        observation.context_image.save(
            output_dir / f"turn_{turn_number:02d}_context.png"
        )
        observation.crop_image.save(output_dir / f"turn_{turn_number:02d}_crop.png")
        _save_json(
            output_dir / f"turn_{turn_number:02d}_observation.json",
            {
                "turn": turn_number,
                "crop_bbox": observation.crop_bbox.to_list(),
                "context_bbox": observation.context_bbox.to_list(),
                "boundary_locked": observation.boundary_locked,
                "crop_scale": observation.crop_scale,
                "context_scale": observation.context_scale,
                "final_turn": final_turn,
            },
        )

        action, attempts = _parse_action_attempts(
            lambda repair: request_main_action(
                client,
                model,
                skill,
                observation,
                mapping,
                _history_summary(turns),
                config,
                final_turn=final_turn,
                repair_error=repair,
            ),
            output_dir,
            f"turn_{turn_number:02d}_main",
            config,
            final_turn=final_turn,
            debug=debug,
        )
        use_subagents = should_escalate_to_subagents(
            action,
            evidence_border_warning=_evidence_touches_border(current, evidence),
            prior_direction=prior_direction,
            stuck_count=stuck_count,
            confidence_threshold=config.subagent_confidence_threshold,
        )
        subagent_record: dict[str, Any] = {}
        decision_source = "main"
        if use_subagents and config.max_subagents > 0:
            logger(f"[recovery] turn {turn_number}: starting read-only subagents")
            action, subagent_record, decision_source = _subagent_decision(
                client,
                model,
                skill,
                observation,
                mapping,
                config,
                output_dir / f"turn_{turn_number:02d}_subagents",
                final_turn=final_turn,
                debug=debug,
            )

        score_before = score_crop_state(current, evidence, full_size)
        turn = RecoveryTurn(
            turn=turn_number,
            crop_before=current,
            crop_after=current,
            action=action,
            format_attempts=attempts,
            score_before=score_before,
            score_after=score_before,
            used_subagents=use_subagents and config.max_subagents > 0,
            subagents=subagent_record,
            decision_source=decision_source,
        )

        if action is None:
            turns.append(turn)
            status = "max_turns_exhausted" if final_turn else "protocol_failed"
            _save_history(
                history_path,
                skill=skill,
                status=status,
                initial_crop=initial,
                current_crop=current,
                best_crop=best,
                best_score=best_score,
                turns=turns,
            )
            break

        _save_json(
            output_dir / f"turn_{turn_number:02d}_action.json",
            {"source": decision_source, "action": action.to_dict()},
        )
        if action.action == "reject_candidate":
            rejected = True
            status = "rejected_not_fai"
            turns.append(turn)
            _save_history(
                history_path,
                skill=skill,
                status=status,
                initial_crop=initial,
                current_crop=current,
                best_crop=best,
                best_score=best_score,
                turns=turns,
            )
            break
        if action.action == "finish":
            valid = True
            status = (
                "validated"
                if bool(mapping.get("complete", False))
                else "validated_from_incomplete_mapping"
            )
            best = current
            best_score = max(best_score, score_before)
            turn.score_after = score_before
            turns.append(turn)
            _save_history(
                history_path,
                skill=skill,
                status=status,
                initial_crop=initial,
                current_crop=current,
                best_crop=best,
                best_score=best_score,
                turns=turns,
            )
            break

        if final_turn:
            # Strict parsing normally prevents this; retain a state-machine guard.
            status = "max_turns_exhausted"
            turns.append(turn)
            _save_history(
                history_path,
                skill=skill,
                status=status,
                initial_crop=initial,
                current_crop=current,
                best_crop=best,
                best_score=best_score,
                turns=turns,
            )
            break

        expansion = expand_crop(current, action, full_size, initial_area, config)
        previous_expansion = expansion
        turn.expansion = expansion
        turn.crop_after = expansion.after
        turn.score_after = score_crop_state(expansion.after, evidence, full_size)
        turns.append(turn)
        if turn.score_after > best_score:
            best = expansion.after
            best_score = turn.score_after
        if expansion.changed:
            expansion_count += 1
            if expansion_output_dir is not None:
                save_expansion_frame(
                    expansion_output_dir,
                    full_image=full_image,
                    marker_box=marker_box,
                    initial_crop=initial,
                    action=action,
                    expansion=expansion,
                    turn_number=turn_number,
                    expansion_number=expansion_count,
                    context_fraction=config.context_fraction,
                    max_image_edge=config.max_image_edge,
                )
        else:
            stuck_count += 1
            status = "boundary_exhausted"
            _save_history(
                history_path,
                skill=skill,
                status=status,
                initial_crop=initial,
                current_crop=current,
                best_crop=best,
                best_score=best_score,
                turns=turns,
            )
            break
        stuck_count = 0
        prior_direction = _dominant_direction(action.arguments)
        current = expansion.after
        _save_history(
            history_path,
            skill=skill,
            status="running",
            initial_crop=initial,
            current_crop=current,
            best_crop=best,
            best_score=best_score,
            turns=turns,
        )
    else:
        status = "max_turns_exhausted"
        _save_history(
            history_path,
            skill=skill,
            status=status,
            initial_crop=initial,
            current_crop=current,
            best_crop=best,
            best_score=best_score,
            turns=turns,
        )

    final_crop = current if valid or rejected else best
    result = RecoveryResult(
        status=status,
        final_crop=final_crop,
        best_crop=best,
        turns=turns,
        rejected=rejected,
        valid=valid,
        best_score=best_score,
        history_path=str(history_path),
    )
    _save_json(output_dir / "result.json", result.to_dict())
    return result
