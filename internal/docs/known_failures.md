## Known and resolved failures

Every failure recorded for this run is explained here. None of them affected a
reported number: each was diagnosed, fixed, and the affected stage re-run.

### `prepare_cache` task 10 --- FAILED (data defect, resolved)
`libpng error: Read Error` on
`Patient_27/IOP_Left_RawImage_27.png`. The file is a truncated PNG: its IHDR
parses and reports 5371x3040, but it has no IEND chunk and cannot be decoded.
This is the defect that motivated `scripts/verify_decodable.py`, which now
attempts a full decode of every image and writes
`derived/undecodable_images.json`; `iop_compass.data.splits.eligibility`
consults it, so Patient_27 is excluded from every split with a recorded reason.
The remaining 15 shards completed and the missing shard was re-run.

### `audit` (job 77822) --- FAILED (intended gate, resolved)
Exit 1 from `--fail-on-fatal` on the finding
`contour_out_of_bounds_severe`: Patient_105 has annotations drawn against a
differently sized image, overshooting by up to 393 px. The gate did its job. The
audit now separates rounding-level overshoot (a warning; up to
`contour_overshoot_tolerance_px`) from an annotation drawn against the wrong
image size (fatal), records the affected images in
`derived/unusable_images.json`, and the split builder excludes their patients.
Re-run after the fix, the audit reports no fatal findings.

### `roi_eval_val` (job 77964) --- FAILED (code defect, resolved)
`ValueError: operands could not be broadcast together with shapes (807,1024)
(806,1024)`. The evaluation grid was recomputed from the already-downscaled
inference image, rounding twice and landing one pixel off the reference raster.
Predictions built the same way would have been silently skipped by the evaluator
as a shape mismatch. Both paths now derive the grid from the native image size,
exactly as the reference rasters were built; `segment_val` and `roi_eval_val` were
re-run from scratch afterwards.

### `segment_val` (job 77963) --- CANCELLED (by us)
Cancelled deliberately, together with the failure above, so that no prediction
written on the wrong grid could reach an evaluation.
