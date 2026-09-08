---
name: fai-crop-recovery
description: Recover and validate incomplete FAI engineering-drawing crops by tracing annotation text, leaders, terminal arrowheads, and touched target features. Use when a candidate crop may be clipped, semantically incomplete or unmapped, contaminated by neighboring annotations, or not a true FAI marker.
---

# Recover an FAI Crop

Treat the red rectangle as the selected marker and the magenta rectangle as the
current crop. Inspect the clean crop and its wider context before deciding.

Follow this order exactly:

1. Confirm that the red rectangle contains an actual `FAI` marker and its
   inspection number. Reject SPC balloons, ordinary holes, circled dimensions,
   datum symbols, and number text without `FAI`.
2. Identify the parameter, tolerance, feature-control frame, and description
   belonging to this FAI. Do not borrow content from a neighboring group.
3. Trace every associated leader from that annotation, segment by segment.
4. Find each terminal arrowhead on those leaders.
5. Confirm that a useful local portion of every part, surface, hole, or
   cross-section touched by an arrowhead is visible.
6. Inspect all four crop borders for clipped relevant text, leaders,
   arrowheads, or target features.
7. Ignore a semantic missing item when the pixels show it is already present
   or belongs to another annotation.

Choose exactly one action:

- Use `expand_crop` when the marker is a real FAI but related content lies
  outside or is clipped by the current crop. Expand only the required sides and
  make one bounded request.
- Use `finish` only when the complete coherent FAI group is visible and no
  related leader reaches a crop border.
- Use `reject_candidate` only when the selected red marker is not a real FAI.

Interpret the two validity fields differently:

- `candidate_valid` answers only whether the selected red rectangle contains a
  real FAI marker. It does not mean that the current crop is complete.
- `valid` answers whether the current crop is complete and contains all related
  FAI content needed for the final output. It may be `true` only when the crop
  can be accepted without any further expansion.

The action fields must use exactly these combinations:

- `expand_crop`: set `candidate_valid=true`, `valid=false`, and make at least
  one of `left_norm`, `top_norm`, `right_norm`, or `bottom_norm` greater than
  zero. List the clipped or outside evidence in `missing`.
- `finish`: set `candidate_valid=true`, `valid=true`, `missing=[]`, and set all
  four expansion directions to zero.
- `reject_candidate`: set `candidate_valid=false`, `valid=false`, and set all
  four expansion directions to zero.

Never return `expand_crop` with every expansion direction set to zero. Never
set `valid=true` when `missing` is non-empty or when requesting expansion.

Never reject a real FAI merely because semantic association selected no
components. Never claim completion to hide uncertainty. Never request expansion
for unrelated FAI/SPC groups, title blocks, drawing views, or nearby dimensions.
On the final turn, do not request another expansion; choose `finish` only when
complete or `reject_candidate` only when the marker is false.
