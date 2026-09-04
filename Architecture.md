# FAI Detection and Crop Architecture

Implementation status: the Version 4 flow described here is implemented in `FAI_DET_CROP_4.py`. Version 4 imports stable detection/evidence helpers from Version 3 but owns the semantic prompt, strict response contract, line-evidence consolidation, Python minimum-crop construction, visible selection artifacts, validation loop, and manifest.

## Purpose

The system finds every FAI inspection annotation group in a large engineering drawing and produces a compact crop for each group. A valid crop should contain the selected FAI marker, its readable number, associated measurement information, complete leader geometry, terminal arrowheads, and a useful local portion of the referenced component.

In this document, **minimum enclosing crop** is the precise term for the idea sometimes described as the “minimum common image” or “最小公约图.” It means the smallest axis-aligned rectangle enclosing all evidence selected for one FAI group, plus deterministic safety padding.

## Core ownership rule

Models propose and associate evidence. Python owns coordinates and cropping.

### Model responsibilities

- LocateAnything proposes visual regions.
- Tesseract proposes OCR text and text boxes.
- OpenCV proposes lines and geometric arrowhead evidence.
- Qwen decides which existing evidence belongs to the selected FAI group.
- Qwen crop validation judges semantic completeness and recommends directional boundary adjustment.

### Python responsibilities

- tile generation;
- local-to-global coordinate conversion;
- deduplication and geometric cleanup;
- candidate-ROI calculation;
- stable evidence IDs;
- validation of Qwen-selected IDs;
- minimum enclosing rectangle calculation;
- padding and boundary clamping;
- directional crop adjustment;
- image cropping and file output.

Qwen semantic association must never invent or modify coordinates.

## End-to-end flow

```text
Input engineering-drawing PNG
        |
        v
Split into overlapping tiles
        |
        v
LocateAnything finds high-recall FAI marker candidates per tile
        |
        v
Convert tile-local coordinates to full-image coordinates
        |
        v
Merge and deduplicate FAI candidates
        |
        v
For each FAI: expand a candidate ROI around its center
        |
        v
LocateAnything + OCR + OpenCV collect local evidence
        |
        v
Build one complete evidence overlay with F/A/T/L/H/G/R IDs
        |
        v
One Qwen semantic-association call selects existing IDs only
        |
        v
Python validates IDs and computes the minimum enclosing crop
        |
        v
Qwen crop-completeness cross-check
        |
        v
Python adjusts left/top/right/bottom boundaries if requested
        |
        v
Validated per-FAI crop and manifest record
```

## Stage 1: Tile the drawing

Large drawings are divided into overlapping image tiles so small FAI circles remain visible to the detector.

Default behavior:

- tile size: up to `1200 x 1200` pixels;
- overlap: 20%;
- approximate stride: 960 pixels;
- edge tiles are repositioned so the full drawing is covered.

Tile overlap intentionally creates duplicate detections. Python later removes them.

## Stage 2: Locate FAI markers

LocateAnything examines each tile and proposes every visual marker matching an FAI balloon: a circle containing the literal `FAI` and an identification number.

The detector returns tile-local boxes. Python:

1. converts normalized coordinates to tile pixels;
2. adds the tile origin to obtain full-image coordinates;
3. clamps boxes to the image;
4. merges boxes using overlap and center-distance rules;
5. sorts candidates in a stable order.

OpenCV circle-pair proposals and a Qwen validation/fallback path may supplement LocateAnything when necessary. Candidate provenance should remain explicit instead of being overwritten with a hard-coded source.

## Stage 3: Expand a candidate ROI around each FAI

For every selected FAI marker, Python creates a larger local region centered on the marker. The ROI must be large enough to include nearby measurement text, leader lines, arrowheads, and the referenced component, while excluding as much unrelated drawing content as practical.

All evidence coordinates inside semantic association use the candidate-ROI coordinate space. Python later translates the final local crop back into full-image coordinates.

The selected marker receives the stable local ID `F0`. `Candidate 0` is a processing index; `F0` is the selected marker inside that candidate's semantic-association problem.

## Stage 4: Collect component evidence

The local evidence stages are complementary rather than duplicated.

| Prefix | Evidence | Producer | Purpose |
|---|---|---|---|
| `F*` | selected FAI marker | initial FAI proposal stage | anchors the current group |
| `A*` | measurement/tolerance annotation region | LocateAnything | proposes semantic annotation blocks |
| `T*` | OCR text line | Tesseract | supplies text, text position, and confidence |
| `H*` | semantic arrowhead proposal | LocateAnything | finds visually meaningful leader arrowheads |
| `L*` | line or merged leader-path segment | OpenCV | supplies geometric connectivity |
| `G*` | geometric triangle proposal | OpenCV | backs up or confirms arrowhead detection |
| `R*` | touched local component/part region | LocateAnything with geometric guidance | proposes the referenced part |

OpenCV may run early enough to guide LocateAnything target-part detection. A useful dependency order is:

```text
A annotation proposals
T OCR evidence
H semantic arrowhead proposals
preliminary OpenCV L/G geometry
R target-part proposal guided by H/G and selected line context
final OpenCV line cleanup using R as an additional anchor
```

This order lets OpenCV assist LocateAnything without making OpenCV responsible for semantic interpretation.

## Stage 5: Clean evidence and build the overlay

Before IDs are assigned, Python should remove redundant proposals while preserving unique semantic paths.

Important cleanup operations:

- merge collinear Hough segments that describe the same physical line;
- collapse near-identical boxes;
- filter line fragments that are remote from all candidate anchors;
- suppress triangle false positives inside text, FAI/SPC circles, or repetitive hatching;
- keep all unique leader branches and terminal arrows;
- assign stable IDs after cleanup.

The evidence overlay uses consistent colors:

- red: selected `F0`;
- orange: `A*` annotation proposals;
- purple: `T*` OCR text;
- cyan: `H*` LocateAnything arrowheads;
- blue: `L*` OpenCV line geometry;
- teal: `G*` OpenCV triangle proposals;
- green: `R*` target-part proposals.

The clean ROI and marked overlay are both passed to Qwen so colored evidence never replaces the original pixels.

## Stage 6: One Qwen semantic association

There is exactly one semantic-association request per candidate. Its purpose is to select one coherent FAI group from the complete evidence image.

Qwen receives:

1. the clean candidate ROI;
2. the same ROI with all evidence IDs;
3. a compact evidence dictionary containing IDs, types, source metadata, OCR text, confidence, and geometric endpoints where applicable.

Qwen returns semantic values and existing evidence IDs only. It does not return any coordinate field.

### Proposed semantic-association prompt

```text
You are the single semantic-association stage of an engineering-drawing FAI pipeline.

You receive two images of the same candidate ROI:
- IMAGE 1 is the clean ROI.
- IMAGE 2 is the evidence overlay.

Evidence ID types in IMAGE 2:
- F*: FAI marker, red
- A*: measurement or tolerance annotation proposal, orange
- T*: OCR text line, purple
- H*: LocateAnything arrowhead proposal, cyan
- L*: OpenCV line or merged line-path proposal, blue
- G*: OpenCV triangular arrowhead proposal, teal
- R*: touched local component or part proposal, green

The selected marker is exactly F0. Associate exactly one coherent annotation group
with F0. Nearby FAI, SPC, datum, section, grid, dimension, and part evidence may belong
to other groups. Do not select them unless the clean image shows that they are part of
F0's measurement annotation or connected leader path.

Select the minimum sufficient set of existing evidence IDs that completely represents
F0's group:
1. Read the FAI number inside F0. Preserve leading zeros by returning it as a string.
2. Read the associated SPC letter only if one is present for F0.
3. Select the annotation region IDs belonging to F0.
4. Select all parameter, tolerance, feature-control-frame, and directly associated
   description text IDs.
5. Select every non-duplicate line ID required to form the complete connected leader
   path from F0's annotation to its terminal arrowhead or terminal feature.
6. Select all terminal arrowhead IDs on that path. H* and G* are alternative evidence
   sources; select the visually correct evidence and do not select a 100% sampling
   triangle, text glyph, circle fragment, hatch fragment, or ordinary part corner.
7. Select the smallest useful R* region touched by each selected terminal arrowhead.

OpenCV line proposals include ordinary part outlines, dimension lines, table borders,
and hatch lines. Select an L* ID only when it belongs to F0's connected measurement path.
Do not select duplicate L* IDs that describe the same physical segment.

Use IMAGE 1 to read semantics and verify physical connections. Use IMAGE 2 and the
Evidence dictionary only to identify the corresponding IDs. OCR text may be incorrect;
prefer the clean image when OCR and the visible drawing disagree.

Coordinate ownership rule:
- Never output a bbox, point, polygon, pixel coordinate, normalized coordinate,
  crop coordinate, padding value, fallback box, or edge adjustment.
- Never alter an evidence ID or the geometry behind it.
- Every selected ID must exist verbatim in the Evidence dictionary.
- If required evidence is not represented by an existing ID, leave the relevant list
  empty, add a concise entry to missing, and set complete to false. Never invent an ID.

measurement_description is text directly associated with the selected measurement.
A section title such as SECTION A-A is not a measurement description unless the drawing
explicitly uses it as part of F0's measurement instruction.

is_range_measurement is true only when the selected annotation explicitly expresses a
range, min/max interval, or limit pair.

Set complete=true only if the selected IDs and semantic values are sufficient for Python
to construct a crop containing the full F0 group. Optional SPC or description fields may
be null when the drawing genuinely has none; absence of an optional field alone does not
make the mapping incomplete.

Reason internally. After reasoning, the response content must be exactly one JSON object
matching the required schema. Do not include prose, Markdown, code fences, XML, analysis,
or coordinate fields in the final content.

Evidence dictionary:
{{EVIDENCE_DICTIONARY_JSON}}

Required final JSON shape:
{
  "marker_id": "F0",
  "fai_number": null,
  "spc_letter": null,
  "annotation_ids": [],
  "parameter_text_ids": [],
  "description_text_ids": [],
  "leader_ids": [],
  "arrowhead_ids": [],
  "target_ids": [],
  "parameter_values": [],
  "measurement_description": null,
  "is_range_measurement": false,
  "complete": false,
  "missing": [],
  "confidence": 0.0
}
```

### Output contract

The prompt must be paired with a request-specific JSON Schema. Prompt wording alone is not a sufficient format guarantee.

Required types:

- `marker_id`: constant string `F0`;
- `fai_number`: string or null;
- `spc_letter`: string or null;
- all `*_ids`: arrays of strings;
- `parameter_values`: array of strings;
- `measurement_description`: string or null;
- `is_range_measurement`: boolean;
- `complete`: boolean;
- `missing`: array of strings;
- `confidence`: number from 0 through 1;
- additional properties: forbidden.

The server keeps thinking enabled and uses the `qwen_3_5` reasoning parser. Internal reasoning belongs in `reasoning_content`; the final JSON belongs in `content`.

Python accepts the result only when:

- generation finishes normally rather than because of a token limit;
- final `content` exists;
- JSON parsing and schema validation succeed;
- every selected ID exists;
- every selected ID has an allowed type;
- the mapping describes `F0` and only one coherent group.

## Stage 7: Python computes the minimum enclosing crop

Python gathers:

```text
F0
+ selected A* annotation boxes
+ selected T* text boxes
+ selected L* line extents
+ selected H*/G* arrowhead boxes
+ selected R* target-part boxes
```

It then:

1. computes the geometric union;
2. converts the union to the smallest axis-aligned enclosing rectangle;
3. adds deterministic arrow context and overall padding;
4. translates the rectangle from ROI coordinates to full-image coordinates;
5. clamps it to the source image.

If semantic mapping is structurally invalid, Python rejects the candidate before crop construction. If the mapping is structurally valid but `complete=false`, Python must not silently add every proposal of a missing type; it constructs the minimum crop from the valid selected IDs and continues into crop validation. The semantic `missing` list is supplied to the validator as visual recovery context rather than treated as ground truth.

## Stage 8: Crop-completeness cross-check

This is a semantic cross-check, not statistical machine-learning cross-validation.

Qwen receives:

1. the wider candidate ROI with the selected FAI in red and proposed crop in magenta;
2. the proposed crop itself.
3. the semantic association's completion state and `missing` list as diagnostic context.

It checks whether the proposed crop contains one complete and coherent group:

- readable FAI number;
- associated SPC marker when present;
- complete parameter, tolerance, feature-control-frame, and description text;
- every connected leader segment;
- every terminal arrowhead;
- a useful local portion of each touched part;
- no dominant unrelated neighboring annotation group.

The validator does not crop pixels. It returns a semantic verdict and directional adjustment request. The current expansion-oriented interface is:

```json
{
  "valid": false,
  "missing": ["description text clipped on the right"],
  "expand_norm": {
    "left": 0,
    "top": 0,
    "right": 200,
    "bottom": 0
  },
  "confidence": 0.9
}
```

Positive values move the corresponding crop boundary outward. This provides the intended left, top, right, and bottom adjustment loop while Python retains coordinate control.

If future requirements include shrinking an oversized crop or translating it without resizing, the contract should add explicit `shrink_norm` or `shift_norm` fields rather than overloading `expand_norm`.

## Stage 9: Iterative adjustment and final output

Python applies the requested directional adjustment and validates again, up to a configured maximum number of attempts.

```text
proposed crop
    -> validate
    -> adjust left/top/right/bottom
    -> validate again
    -> save only when valid
```

Successful output includes:

- a crop named with the verified FAI number;
- selected evidence IDs;
- semantic values;
- initial and final crop boxes;
- validation history;
- confidence and completion status.

Candidates that cannot achieve a valid semantic mapping or valid crop are written to a failure/debug location with their raw responses and overlays. They are not mixed with successful crops.

## Required invariants

1. One candidate corresponds to one selected `F0` marker.
2. One Qwen semantic-association call selects the complete group.
3. Qwen semantic association returns IDs and semantic values, never coordinates.
4. Python alone computes and modifies crop coordinates.
5. Every selected ID exists and has the expected type.
6. A normal crop is never saved from an invalid mapping.
7. A normal crop is never saved after an unresolved validation failure.
8. Candidate provenance, raw responses, and validation history remain auditable.
