# FAI Detection Pipeline Progress

Last updated: 2026-09-03

## Version 4 implementation

`FAI_DET_CROP_4.py` now implements the ID-only, one-call semantic-association design while reusing the stable detection helpers from Version 3.

Implemented behavior:

- one Qwen semantic-association generation per candidate;
- request-level strict JSON Schema;
- no semantic output coordinates and no `fallback_boxes_norm`;
- explicit `finish_reason`, token-usage, content-presence, and mapping-validity metadata;
- Python validation of required fields, evidence IDs, and evidence kinds;
- merging of overlapping/collinear OpenCV line evidence before Qwen sees it;
- no class-wide `A/H/G/R` fallback during minimum-crop calculation;
- a visible Qwen-selection overlay saved in `crops/`;
- a raw Python minimum-enclosing crop saved in `crops/`;
- an annotated minimum crop showing only Qwen-selected evidence;
- iterative crop validation with up to three attempts by default;
- failed semantic mappings and unresolved validations separated under `failed/`;
- incremental `results.json` updates after each candidate.

Version 4 output names distinguish artifact roles:

```text
FAI224_000_qwen_selected.png
FAI224_000_qwen_selection.json
FAI224_000_minimum.png
FAI224_000_minimum_selected.png
FAI224_000_validated.png
```

If Qwen returns a structurally valid but explicitly incomplete association, the selection and minimum-crop artifacts use an `_INCOMPLETE` marker, but the candidate now continues into crop validation. The semantic `missing` list is passed to the validator as visual recovery context. Only a structurally invalid mapping is rejected before crop validation.

## Current status

The pipeline can currently:

1. Split a large PNG drawing into overlapping tiles.
2. Use LocateAnything to propose FAI marker positions.
3. Convert tile-local boxes back to full-image coordinates and deduplicate them.
4. Build a candidate ROI around each selected FAI marker.
5. Collect candidate evidence with LocateAnything, Tesseract OCR, and OpenCV.
6. Draw an evidence overlay with stable IDs such as `F0`, `A0`, `T0`, `L0`, `H0`, `G0`, and `R0`.
7. Ask Qwen to associate the evidence with the selected `F0` marker.
8. Build and validate a crop.

The LocateAnything, OCR, OpenCV, overlay, and crop-validation portions are producing usable intermediate artifacts. The main unresolved problem is the Qwen semantic-association contract.

## Confirmed semantic-association failure

### Symptom

Generated crops are named like:

```text
FAIunknown_000.png
```

This happens even when Qwen correctly reads the FAI number during its reasoning. For example, `output_v9/raw_responses/candidate_000_mapping.txt` identifies the selected marker as FAI 224 and identifies its SPC value, but never emits the required final JSON object.

### Evidence

For `output_v9` Candidate 0:

- Qwen visually identified `F0` as FAI 224.
- The evidence set contained 122 primitives:
  - 1 selected FAI marker;
  - 36 OCR text records;
  - 70 OpenCV line segments;
  - 4 LocateAnything arrowhead proposals;
  - 11 OpenCV triangle proposals.
- The mapping response grew to about 25 KB and ended in the middle of its reasoning.
- The response never reached the final JSON object.
- `extract_json()` therefore returned `semantic_mapping_json` failure.
- `mapping.get("fai_number")` returned `None`.
- `safe_fai_name(None)` produced `unknown`.

The crop-validation responses in the same run are clean JSON. This indicates that the `qwen_3_5` reasoning parser can separate a completed reasoning response, while the larger semantic-mapping request is exhausting its output allowance before it completes thinking and produces the final answer.

### Root causes

The current semantic prompt asks one response to do too much low-level work:

1. Identify the selected FAI number and optional SPC letter.
2. Interpret OCR text.
3. Select measurement and description evidence.
4. Examine many raw and often duplicate Hough line segments.
5. Distinguish real arrowheads from false triangle proposals.
6. Select a touched target part.
7. Decide completeness.
8. Calculate normalized fallback coordinates for annotation, target, and complete-group boxes.

The last requirement is especially problematic. Qwen spends reasoning tokens estimating ROI dimensions and manually converting pixel positions to normalized coordinates. Coordinate calculation is deterministic work and should belong to Python, not the semantic model.

### Downstream consequences

When semantic mapping has no valid JSON:

- the FAI number is lost even if it appeared in reasoning;
- the crop filename becomes `FAIunknown_*`;
- selected `A/T/L/H/G/R` IDs are unavailable;
- the initial-crop code falls back to broad proposal classes instead of a verified coherent group;
- OCR text and leader lines may be omitted from the crop calculation;
- unrelated arrows or nearby annotations may enlarge the crop;
- crop validation can only inspect or expand the resulting crop and cannot reconstruct the missing semantic mapping;
- multiple records with `fai_number=None` can interact incorrectly with final deduplication;
- a crop may still be saved even though mapping or the final validation is incomplete.

## Architecture decision

Keep exactly one Qwen semantic-association call per FAI candidate.

The evidence-producing stages and their marked overlay exist specifically so Qwen can see the complete candidate context in one pass. The semantic request must therefore continue to receive:

- the clean candidate ROI;
- the complete evidence overlay;
- the evidence dictionary.

Qwen will only perform semantic interpretation and select existing evidence IDs. It will not output boxes, points, normalized coordinates, crop coordinates, padding, or edge adjustments.

Python will retain exclusive ownership of:

- coordinate conversion;
- evidence-ID validation;
- geometric union;
- padding;
- crop construction;
- boundary adjustment;
- physical image cropping.

## Required changes

### Semantic prompt

- Remove `fallback_boxes_norm` completely.
- Explicitly prohibit coordinate output.
- Require the fixed marker ID `F0`.
- Require exactly one coherent annotation group.
- Require selection of existing `A/T/L/H/G/R` IDs only.
- Require empty lists plus `missing` entries when required evidence is unavailable.
- Require a single final JSON object with no prose in `content`.

### API response contract

- Keep Qwen thinking enabled.
- Keep the oMLX reasoning parser set to `qwen_3_5`.
- Use a request-specific JSON Schema or JSON-object response format when supported.
- Record `finish_reason`, completion-token usage, and whether final `content` was present.
- Treat `finish_reason="length"`, empty `content`, invalid JSON, and invalid IDs as mapping failures.

### Evidence cleanup

The marked image must remain complete, but redundant evidence should be consolidated before Qwen sees it:

- merge overlapping and collinear Hough segments;
- remove near-identical OCR boxes;
- reject triangle proposals that lie inside FAI/SPC circles or hatching and are not near a plausible leader endpoint;
- preserve every unique connection needed to trace a leader path;
- assign stable IDs only after cleanup.

### Crop construction

- Build the initial crop only from `F0` and Qwen-selected IDs.
- Do not silently add every annotation, arrowhead, triangle, or target proposal when mapping is incomplete.
- Compute the minimum enclosing rectangle in Python.
- Add explicit, deterministic safety padding in Python.
- Do not save a normal crop when semantic mapping is invalid.

### Crop validation

- Validate the proposed crop against the wider candidate context.
- Allow repeated directional adjustment up to a configured attempt limit.
- Apply the final validator-requested adjustment instead of ignoring it.
- Save unresolved cases under a failure/debug path, not as successful crops.

## Acceptance criteria

A candidate is allowed into the normal `crops/` output only when:

1. Qwen returns a parseable final JSON object.
2. `finish_reason` indicates normal completion rather than length exhaustion.
3. `marker_id` is exactly `F0`.
4. `fai_number` is readable and non-empty, or the result is explicitly classified as unreadable.
5. Every selected evidence ID exists and has an allowed type.
6. The selected evidence represents one coherent FAI annotation group.
7. Python produces the crop from selected evidence rather than unverified class-wide fallback proposals.
8. Crop validation returns `valid=true` within the allowed adjustment attempts.

## Next verification step

Run `FAI_DET_CROP_4.py` against `p11.png` with the oMLX Qwen model configured for thinking plus the `qwen_3_5` reasoning parser. Inspect the first candidate's mapping metadata, Qwen-selection overlay, selection JSON, and minimum crop before committing to a full multi-candidate run.
