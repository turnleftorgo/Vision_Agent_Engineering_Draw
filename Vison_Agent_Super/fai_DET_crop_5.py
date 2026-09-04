#!/usr/bin/env python3
"""Super FAI V5: evidence-first detection plus agentic crop recovery."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import FAI_DET_CROP_3 as core  # noqa: E402

from recovery import (  # noqa: E402
    CropBox,
    EvidenceBox,
    RecoveryConfig,
    load_recovery_skill,
    run_crop_recovery,
)
from semantic import (  # noqa: E402
    build_compact_selection_image,
    build_compact_semantic_evidence,
    run_compact_semantic_mapping,
)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def bbox_gap(first: core.BBox, second: core.BBox) -> float:
    a = first.ordered()
    b = second.ordered()
    dx = max(a.x1 - b.x2, b.x1 - a.x2, 0.0)
    dy = max(a.y1 - b.y2, b.y1 - a.y2, 0.0)
    return math.hypot(dx, dy)


def selected_component_ids(mapping: dict[str, Any]) -> list[str]:
    keys = (
        "annotation_ids",
        "text_ids",
        "leader_path_ids",
        "arrowhead_ids",
        "target_ids",
    )
    selected: list[str] = []
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, list):
            selected.extend(item for item in value if isinstance(item, str))
    return list(dict.fromkeys(selected))


def exact_or_bootstrap_crop(
    mapping: dict[str, Any],
    primitives: list[core.Primitive],
    roi_bbox: core.BBox,
    image_size: tuple[int, int],
) -> tuple[core.BBox, str, list[str]]:
    """Build an exact semantic union or deterministic V5 evidence bootstrap."""
    by_id = {item.id: item for item in primitives}
    chosen_ids = [item for item in core.mapping_selected_ids(mapping) if item in by_id]
    component_ids = [item for item in chosen_ids if item != "F0"]
    diagonal = math.hypot(roi_bbox.width, roi_bbox.height)

    if component_ids:
        selected = [by_id[item] for item in chosen_ids]
        boxes = [item.bbox for item in selected]
        arrow_margin = max(24.0, diagonal * 0.04)
        boxes.extend(
            item.bbox.expand(arrow_margin, arrow_margin, arrow_margin, arrow_margin)
            for item in selected
            if item.kind in {"arrowhead", "triangle"}
        )
        source = "semantic_exact_union"
    else:
        marker = by_id.get("F0")
        if marker is None:
            raise ValueError("Candidate evidence has no F0 marker")
        boxes = [marker.bbox]
        selected_ids = ["F0"]

        def take(
            kinds: set[str], limit: int, reference: core.BBox
        ) -> list[core.Primitive]:
            candidates = [item for item in primitives if item.kind in kinds]
            candidates.sort(
                key=lambda item: (bbox_gap(item.bbox, reference), -item.bbox.area)
            )
            return candidates[:limit]

        annotation = take({"annotation"}, 5, marker.bbox)
        boxes.extend(item.bbox for item in annotation)
        selected_ids.extend(item.id for item in annotation)
        anchor = core.BBox.union(boxes) or marker.bbox
        text = take({"ocr_text"}, 8, anchor)
        boxes.extend(item.bbox for item in text)
        selected_ids.extend(item.id for item in text)
        anchor = core.BBox.union(boxes) or marker.bbox
        leaders = take({"leader_segment"}, 24, anchor)
        boxes.extend(item.bbox for item in leaders)
        selected_ids.extend(item.id for item in leaders)
        anchor = core.BBox.union(boxes) or marker.bbox
        terminals = take({"arrowhead", "triangle", "target_part"}, 12, anchor)
        boxes.extend(item.bbox for item in terminals)
        selected_ids.extend(item.id for item in terminals)
        chosen_ids = list(dict.fromkeys(selected_ids))
        source = "evidence_bootstrap"

    union = core.BBox.union(boxes)
    if union is None:
        union = core.BBox(0, 0, roi_bbox.width, roi_bbox.height)
    marker_box = by_id["F0"].bbox
    cx, cy = marker_box.center
    minimum = core.BBox(
        cx - max(180.0, marker_box.width * 5.0),
        cy - max(140.0, marker_box.height * 4.0),
        cx + max(180.0, marker_box.width * 5.0),
        cy + max(140.0, marker_box.height * 4.0),
    ).clamp(round(roi_bbox.width), round(roi_bbox.height))
    union = core.BBox.union([union, minimum]) or union
    padding = max(12.0, diagonal * 0.018)
    local = union.expand(padding, padding, padding, padding).clamp(
        round(roi_bbox.width), round(roi_bbox.height)
    )
    global_box = local.translate(roi_bbox.x1, roi_bbox.y1).clamp(*image_size)
    return global_box, source, chosen_ids


def global_evidence(
    primitives: list[core.Primitive],
    roi_bbox: core.BBox,
    selected_ids: Iterable[str],
) -> list[EvidenceBox]:
    selected = set(selected_ids)
    return [
        EvidenceBox(
            id=item.id,
            kind=item.kind,
            bbox=CropBox.from_values(item.bbox.translate(roi_bbox.x1, roi_bbox.y1)),
            selected=item.id in selected,
        )
        for item in primitives
    ]


def write_overview(
    image: Image.Image,
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(records):
        marker = record.get("marker_bbox")
        crop = record.get("final_crop_bbox")
        if isinstance(marker, list) and len(marker) == 4:
            draw.rectangle(tuple(marker), outline=(255, 0, 0), width=3)
        if isinstance(crop, list) and len(crop) == 4:
            color = (
                (0, 160, 0)
                if record.get("status", "").startswith("validated")
                else (220, 0, 220)
            )
            draw.rectangle(tuple(crop), outline=color, width=4)
            draw.text((crop[0] + 3, crop[1] + 3), str(index), fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def manifest_base(
    input_path: Path,
    image: Image.Image,
    args: argparse.Namespace,
    skill: Any,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": "5-super-agentic-recovery",
        "source": str(input_path),
        "source_size": list(image.size),
        "coordinate_space": "absolute_pixels",
        "models": {
            "locate": args.locate_model,
            "semantic_and_recovery": args.qwen_model,
            "endpoint": args.endpoint,
        },
        "recovery": {
            "enabled": not args.no_verify,
            "max_turns": args.max_turns,
            "max_format_retries": args.max_format_retries,
            "max_subagents": args.max_subagents,
            "skill_path": str(skill.path) if skill else None,
            "skill_sha256": skill.sha256 if skill else None,
        },
        "semantic": {
            "image_count": 1,
            "format": "compact_clean_roi_with_merged_leader_paths",
            "max_tokens": args.semantic_max_tokens,
        },
        "results": records,
    }


def process_image_v5(args: argparse.Namespace) -> list[dict[str, Any]]:
    input_path = Path(args.image).expanduser().resolve()
    if input_path.suffix.lower() != ".png":
        raise ValueError("Super V5 currently accepts PNG input only")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    output_dir = Path(args.output).expanduser().resolve()
    raw_dir = output_dir / "raw_responses"
    debug_dir = output_dir / "debug"
    crop_dir = output_dir / "crop"
    related_dir = output_dir / "crop_related"
    expand_dir = output_dir / "expand"
    for directory in (crop_dir, related_dir, expand_dir):
        directory.mkdir(parents=True, exist_ok=True)

    image = Image.open(input_path).convert("RGB")
    client = OpenAI(base_url=args.endpoint, api_key=args.api_key, timeout=args.timeout)
    skill = None
    if not args.no_verify:
        skill = load_recovery_skill(Path(args.recovery_skill))

    marker_boxes = core.detect_fai_candidates(
        client,
        args.locate_model,
        image,
        args.tile_size,
        args.tile_overlap,
        raw_dir,
        debug_dir / "tiles" if args.debug else None,
    )
    core.log("[1/8] OpenCV circle-pair proposals + Qwen magnified validation")
    circle_pairs = core.detect_circle_pair_candidates(image)
    circle_boxes = core.qwen_validate_circle_pairs(
        client,
        args.qwen_model,
        image,
        circle_pairs,
        raw_dir / "circle_pair_validation.txt",
        debug_dir / "circle_pair_contact_sheet.png" if args.debug else None,
    )
    marker_boxes = core.deduplicate_boxes(marker_boxes + circle_boxes)
    if not marker_boxes:
        core.log("[1/8] Hybrid proposals empty; trying full-page Qwen fallback")
        marker_boxes = core.qwen_fai_fallback(
            client, args.qwen_model, image, raw_dir / "qwen_fai_fallback.txt"
        )
    if not marker_boxes:
        raise RuntimeError("No FAI marker candidates were found")
    marker_boxes = marker_boxes[: args.max_candidates]
    core.log(f"[1/8] Found {len(marker_boxes)} deduplicated FAI candidate(s)")

    records: list[dict[str, Any]] = []
    manifest_path = output_dir / "results.json"
    for index, marker_box in enumerate(marker_boxes):
        marker = core.Primitive("F0", "fai_marker", marker_box, "LocateAnything")
        roi_bbox = core.candidate_roi(marker_box, image.width, image.height)
        raw_roi, primitives, evidence_overlay = core.create_candidate_evidence(
            client,
            args.locate_model,
            image,
            marker,
            roi_bbox,
            raw_dir,
            index,
            use_tesseract=not args.no_tesseract,
        )
        related_candidate_dir = related_dir / f"candidate_{index:03d}"
        related_candidate_dir.mkdir(parents=True, exist_ok=True)
        raw_roi.save(related_candidate_dir / "roi.png")
        evidence_overlay.save(related_candidate_dir / "evidence.png")
        save_json(
            related_candidate_dir / "primitives.json",
            [item.prompt_record() for item in primitives],
        )

        compact_evidence = build_compact_semantic_evidence(raw_roi, primitives)
        compact_evidence.image.save(related_candidate_dir / "semantic.png")
        save_json(
            related_candidate_dir / "semantic.json",
            {
                "records": compact_evidence.records,
                "leader_path_segments": compact_evidence.path_segments,
                "input_image_count": 1,
            },
        )

        core.log(
            f"[4/8] Candidate {index}: one Qwen semantic association "
            f"({len(primitives)} primitives -> {len(compact_evidence.records)} compact "
            f"records, {len(compact_evidence.paths)} LP path(s), 1 image)"
        )
        mapping = run_compact_semantic_mapping(
            client,
            args.qwen_model,
            compact_evidence,
            related_candidate_dir / "semantic_mapping.txt",
            related_candidate_dir / "semantic_mapping_meta.json",
            max_tokens=args.semantic_max_tokens,
        )
        semantic_ids = selected_component_ids(mapping)
        core.log(
            f"[4/8] Candidate {index}: Qwen selected {len(semantic_ids)} compact "
            f"component(s): {', '.join(semantic_ids) if semantic_ids else 'none'}"
        )
        minimum_crop, minimum_source, chosen_ids = exact_or_bootstrap_crop(
            mapping, primitives, roi_bbox, image.size
        )
        fai_name = core.safe_fai_name(mapping.get("fai_number"))
        base = f"FAI{fai_name}_{index:03d}"
        selected_overlay = build_compact_selection_image(
            compact_evidence,
            chosen_ids,
            mapping.get("leader_path_ids", []),
        )
        selected_path = related_candidate_dir / "qwen_selected.png"
        minimum_path = related_candidate_dir / "minimum.png"
        selected_overlay.save(selected_path)
        image.crop(minimum_crop.to_int_tuple()).save(minimum_path)
        save_json(
            related_candidate_dir / "selection.json",
            {
                "mapping": mapping,
                "qwen_selected_compact_ids": semantic_ids,
                "selected_ids": chosen_ids,
                "selected_leader_path_ids": mapping.get("leader_path_ids", []),
                "leader_path_segments": compact_evidence.path_segments,
                "compact_evidence_records": compact_evidence.records,
                "minimum_source": minimum_source,
                "minimum_crop_bbox": minimum_crop.to_list(),
            },
        )
        core.log(
            f"[5/8] Candidate {index}: {minimum_source} saved "
            f"with {len(chosen_ids) - 1} component(s)"
        )

        record: dict[str, Any] = {
            "candidate_index": index,
            "marker_bbox": marker_box.to_list(),
            "roi_bbox": roi_bbox.to_list(),
            "mapping": mapping,
            "qwen_selected_compact_ids": semantic_ids,
            "semantic_selected_ids": chosen_ids,
            "minimum_source": minimum_source,
            "minimum_crop_bbox": minimum_crop.to_list(),
            "related_dir": str(related_candidate_dir),
            "selection_image_path": str(selected_path),
            "minimum_crop_path": str(minimum_path),
        }

        if args.no_verify:
            final_box = minimum_crop
            final_path = crop_dir / f"{base}_verification_skipped.png"
            image.crop(final_box.to_int_tuple()).save(final_path)
            record.update(
                {
                    "status": "verification_skipped",
                    "final_crop_bbox": final_box.to_list(),
                    "final_crop_path": str(final_path),
                    "expansion_dir": None,
                    "expansion_count": 0,
                    "recovery": None,
                }
            )
        else:
            assert skill is not None
            core.log(f"[6/8] Candidate {index}: starting agentic crop recovery")
            evidence = global_evidence(primitives, roi_bbox, chosen_ids)
            config = RecoveryConfig(
                max_turns=args.max_turns,
                max_format_retries=args.max_format_retries,
                max_subagents=args.max_subagents,
                max_content_bytes=args.max_content_bytes,
                max_direction_norm=args.max_direction_norm,
                max_crop_area_ratio=args.max_crop_area_ratio,
                max_crop_growth=args.max_crop_growth,
                recovery_max_tokens=args.recovery_max_tokens,
                subagent_confidence_threshold=args.subagent_confidence_threshold,
                context_fraction=args.context_fraction,
                max_image_edge=args.max_image_edge,
            )
            expansion_candidate_dir = expand_dir / f"candidate_{index:03d}"
            recovery = run_crop_recovery(
                client,
                args.qwen_model,
                image,
                CropBox.from_values(marker_box),
                CropBox.from_values(minimum_crop),
                mapping,
                evidence,
                skill,
                config,
                related_candidate_dir / "recovery",
                debug=args.debug,
                logger=lambda message, idx=index: core.log(
                    f"[7/8] Candidate {idx}: {message}"
                ),
                expansion_output_dir=expansion_candidate_dir,
            )
            final_box = core.BBox(*recovery.final_crop.to_int_tuple())
            if recovery.rejected:
                final_path = crop_dir / f"Candidate{index:03d}_rejected_not_fai.png"
            elif recovery.valid:
                final_path = crop_dir / f"{base}_validated.png"
            else:
                final_path = crop_dir / f"{base}_best_unvalidated.png"
            image.crop(final_box.to_int_tuple()).save(final_path)
            expansion_count = sum(
                1
                for turn in recovery.turns
                if turn.expansion is not None and turn.expansion.changed
            )
            record.update(
                {
                    "status": recovery.status,
                    "final_crop_bbox": final_box.to_list(),
                    "final_crop_path": str(final_path),
                    "expansion_dir": (
                        str(expansion_candidate_dir) if expansion_count else None
                    ),
                    "expansion_count": expansion_count,
                    "recovery": recovery.to_dict(),
                }
            )
        records.append(record)
        save_json(manifest_path, manifest_base(input_path, image, args, skill, records))
        core.log(
            f"[8/8] Candidate {index}: {record['status']} -> {record['final_crop_path']}"
        )

    if args.debug:
        write_overview(image, records, debug_dir / "overview.png")
    save_json(manifest_path, manifest_base(input_path, image, args, skill, records))
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect FAI groups and recover complete crops with a bounded agentic loop."
    )
    parser.add_argument("image", help="Path to the input PNG")
    parser.add_argument("-o", "--output", default="output_super_v5")
    parser.add_argument("--endpoint", default=core.DEFAULT_ENDPOINT)
    parser.add_argument(
        "--api-key", default=os.environ.get("LOCAL_VLM_API_KEY", core.DEFAULT_API_KEY)
    )
    parser.add_argument("--locate-model", default=core.DEFAULT_LOCATE_MODEL)
    parser.add_argument("--qwen-model", default=core.DEFAULT_QWEN_MODEL)
    parser.add_argument("--tile-size", type=int, default=1200)
    parser.add_argument("--tile-overlap", type=float, default=0.20)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-format-retries", type=int, default=2)
    parser.add_argument("--max-subagents", type=int, default=3)
    parser.add_argument("--semantic-max-tokens", type=int, default=8192)
    parser.add_argument("--recovery-max-tokens", type=int, default=8192)
    parser.add_argument("--max-content-bytes", type=int, default=4096)
    parser.add_argument("--max-direction-norm", type=int, default=500)
    parser.add_argument("--max-crop-area-ratio", type=float, default=0.45)
    parser.add_argument("--max-crop-growth", type=float, default=12.0)
    parser.add_argument("--subagent-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--context-fraction", type=float, default=0.50)
    parser.add_argument("--max-image-edge", type=int, default=2400)
    parser.add_argument(
        "--recovery-skill",
        default=str(SCRIPT_DIR / "skills" / "fai-crop-recovery" / "SKILL.md"),
    )
    parser.add_argument("--no-tesseract", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.tile_size < 256:
        parser.error("--tile-size must be at least 256")
    if not 0.0 <= args.tile_overlap < 0.9:
        parser.error("--tile-overlap must be in [0.0, 0.9)")
    if args.max_candidates < 1:
        parser.error("--max-candidates must be at least 1")
    if args.max_turns < 1:
        parser.error("--max-turns must be at least 1")
    if args.max_format_retries < 0:
        parser.error("--max-format-retries cannot be negative")
    if not 0 <= args.max_subagents <= 3:
        parser.error("--max-subagents must be between 0 and 3")
    if args.semantic_max_tokens < 1024:
        parser.error("--semantic-max-tokens must be at least 1024")
    if args.recovery_max_tokens < 1024:
        parser.error("--recovery-max-tokens must be at least 1024")
    if not 512 <= args.max_content_bytes <= 16384:
        parser.error("--max-content-bytes must be between 512 and 16384")
    if not 1 <= args.max_direction_norm <= 500:
        parser.error("--max-direction-norm must be between 1 and 500")
    if not 0.0 < args.max_crop_area_ratio <= 1.0:
        parser.error("--max-crop-area-ratio must be in (0, 1]")
    if args.max_crop_growth < 1.0:
        parser.error("--max-crop-growth must be at least 1")
    if not 0.0 <= args.subagent_confidence_threshold <= 1.0:
        parser.error("--subagent-confidence-threshold must be in [0, 1]")
    if not 0.05 <= args.context_fraction <= 1.0:
        parser.error("--context-fraction must be in [0.05, 1]")
    if args.max_image_edge < 512:
        parser.error("--max-image-edge must be at least 512")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    try:
        process_image_v5(args)
    except Exception as exc:
        core.log(f"ERROR: {exc}")
        if args.debug:
            raise
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
