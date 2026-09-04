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

Never reject a real FAI merely because semantic association selected no
components. Never claim completion to hide uncertainty. Never request expansion
for unrelated FAI/SPC groups, title blocks, drawing views, or nearby dimensions.
On the final turn, do not request another expansion; choose `finish` only when
complete or `reject_candidate` only when the marker is false.
