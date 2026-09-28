# small-cone failure study plan

## question and hypothesis

Can stronger scale augmentation improve detection in images containing small,
likely distant cones without materially reducing performance on ordinary test
images?

Box size is only a proxy for distance. This study does not claim to measure
distance or real-world robustness.

## data and split

Use the full FSOCO dataset. Prepare it once with seed `42` and group images by
their top-level source team before assigning train, validation, and test. This
is the strongest source grouping exposed by FSOCO's downloaded directory
layout. The dataset does not identify recording sessions or tracks, so those
cannot be verified.

The study command will also check exact file hashes and perceptual hashes across
splits. Any cross-split source group, exact duplicate, or flagged near duplicate
is a failed leakage check and must be resolved before interpreting results.

Partition test images into two fixed, disjoint slices:

- `small_cones`: at least one ground-truth box has normalized area at most
  `0.005`.
- `ordinary`: every ground-truth box has normalized area greater than `0.005`.

Record the rule, image membership, image count, annotated-cone count, and
per-class cone count in the study manifest. The threshold is fixed before
looking at model results. Metrics on `small_cones` still include every annotated
cone in each selected image; this limitation must appear in the report.

## baseline and intervention

Use YOLO11n initialized from `yolo11n.pt`, image size `640`, batch size `16`,
`50` epochs, and seed `42`. The baseline sets Ultralytics `scale=0.5`
explicitly. The intervention changes only `scale` to `0.9`, increasing the
range of random image scaling during training. Both configurations disable
external loggers and use the same prepared dataset and split manifest.

One run per condition is the required local path because this repository does
not include the dataset, weights, or a compute budget. Seeds `43` and `44` are
recommended repeats when compute permits. Do not report uncertainty from a
single run. Repeat runs keep dataset `split_seed=42` while changing only the
training seed.

## measurements and acceptance criteria

Evaluate the best validation-selected checkpoint on the full test set and both
slices with identical inference settings. Report mAP50-95 and recall, plus
per-class results only for classes with at least ten annotations in that slice.

The intervention supports the hypothesis only if its `small_cones` mAP50-95
improves by at least `0.02` absolute while `ordinary` mAP50-95 falls by no more
than `0.01` absolute. Always show raw deltas for mAP50-95 and recall. These are
study decision thresholds, not statistical significance claims.

Save canonical predictions and use confidence `0.25` and IoU `0.5` for failure
matching. Generate a deterministic gallery containing false negatives and
false positives, including examples where the intervention helps, hurts, and
has no clear effect when such examples exist.

## deliverables

The implementation will provide commands to prepare and inspect the slices,
run slice evaluation for either experiment, and build a Markdown report and
gallery entirely from saved manifests, predictions, and result JSON. With no
local dataset or completed training runs, the empirical conclusion remains
open; only actual run artifacts may populate result tables and figures.
