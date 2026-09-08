# The viewer

`app/` is a browser interface for reviewing and correcting tooth instance
segmentations. It is the clinical face of the benchmark: a clinician uploads a
patient's photographs, the pipeline labels and segments them, and the clinician fixes
what the model got wrong.

## What it runs

```
photographs → view classification → segmenter (full image) → post-processing → FDI instances
                  ResNet18             SAT or Mask R-CNN       frozen config
```

The viewer does not implement any of these stages. It calls the same modules the
benchmark called — `classification.models`, `classification.constrained_assignment`,
`segmentation.segmentanytooth_adapter`, `segmentation.mask_rcnn`,
`segmentation.postprocessing` — so what a clinician sees is what was measured.

**No ROI stage runs.** The benchmark's four ROI levels all scored below the full
image on the primary outcome, so the two cells the viewer serves are
`no-roi+sat+post` and `no-roi+mask-rcnn+post`. Every `/segment` response carries its
`cell`, so a clinical result can be traced to a row of the benchmark table.

**Mirror acquisition is a per-case tick, and it is not cosmetic.** The cohort's buccal
labels agree with the FDI quadrants its annotations contain — `src/iop_compass/data/validation.py`
declares `left_buccal → (2, 3)` and the audit excludes any image whose annotation
quadrants indicate the opposite view — so there is exactly one left/right convention
the view classifier and SegmentAnyTooth's per-view detectors were trained in.

Ticking the box says the photographs were taken through an intraoral mirror, and the
backend then flips each image once at ingest, before any model sees it. That is
deliberately a flip of the image rather than a relabelling of the prediction:
SegmentAnyTooth picks a per-view detector and, for the left view, applies a horizontal
flip and a remapped class table, so a merely renamed view would still yield the wrong
tooth *numbers*. The uploaded bytes are never modified and `/segment` reflects the
contours back, so everything the clinician sees and everything `save` writes stays in
the uploaded photograph's frame; `mirror_acquisition` is recorded in each
`_annotations.json` as provenance. The tick locks once a case is open, because it is
applied at upload.

**View classification is joint when it can be.** With exactly five images the labels
are resolved under a one-to-one image/view constraint rather than independently: a
standardised acquisition contains each view once, so an argmax that returns two
frontals is knowably wrong. That configuration measured 1.0000 ± 0.0000 accuracy.
Any other image count falls back to the per-image argmax.

## Running it

```bash
cd app
make install     # frontend dependencies, once
make check       # validate every weight the enabled segmenters need
make dev         # API on :5000, UI on http://localhost:5173
```

`make check` fails loudly and names the missing file together with the environment
variable that relocates it. To serve one segmenter only, so the other's checkpoint is
not required: `make backend SEGMENTERS=sat`.

The SegmentAnyTooth path needs the upstream source as well as the weights: `sat`
loads `sam_load` / `sam_predict` from it. The code is MIT and is cloned into
`inputs/SegmentAnyTooth`, the weights go in `weights/SegmentAnyTooth weights/`, and
`scripts/fetch_third_party.py` links both into `third_party/` —
[Where everything goes](../README.md#where-everything-goes) is the full layout,
including the two names the release checkpoints must be renamed to.

| variable | default |
| --- | --- |
| `IOPC_WEIGHTS_DIR` | `weights/` |
| `IOPC_CLASSIFIER_WEIGHTS` | `weights/view_classifier.pt` |
| `IOPC_MASKRCNN_WEIGHTS` | `weights/maskrcnn.pt` |
| `SAT_WEIGHT_DIR`, `SAT_CODE_DIR` | `third_party/sat_weights`, `third_party/segmentanytooth_src` (both symlinks made by `scripts/fetch_third_party.py`) |
| `IOPC_POSTPROCESSING_CONFIG` | `configs/segmentation/postprocessing.yaml` |
| `IOPC_OUTPUT_DIR` | `app/backend/outputs` |
| `IOPC_DEVICE` | `cuda` (falls back to CPU with a warning) |
| `IOPC_INFERENCE_LONG_SIDE` | `2048` |
| `IOPC_CORS_ORIGINS` | `http://localhost:5173` |
| `IOPC_MAX_UPLOAD_BYTES` | `40000000` |

## API

| endpoint | does |
| --- | --- |
| `GET /api/health` | readiness, device, which segmenters are loaded |
| `GET /api/config` | views, segmenters and their cells, the FDI vocabulary |
| `POST /api/cases` | upload N photographs (form field `mirror_acquisition`, default true), get a case and per-image ids |
| `GET /api/cases/<case>/images/<image>` | the original photograph |
| `POST /api/cases/<case>/classify` | predict the view of every image of the case |
| `POST /api/cases/<case>/segment` | segment one image with a chosen segmenter |
| `POST /api/cases/<case>/save` | write the corrected annotations |

`save` writes one directory per patient under `IOPC_OUTPUT_DIR`, and per image an
`_original.png`, a `_mask.png` label raster and an `_annotations.json` in the schema
`iop_compass.data.adapter` already reads — so a corrected case folds straight back
into the dataset with no conversion step.

## Editing

| | |
| --- | --- |
| `V` select · `H` pan | click an instance, drag the image |
| `N` new | trace a tooth the model missed |
| `A` add · `E` erase | extend or trim the selected instance |
| Split · Merge · Delete | fix a merged crown, a split crown, a false positive |
| `Ctrl+Z` / `Ctrl+Shift+Z` | undo, redo |
| FDI keypad | assign or correct a tooth number |

Codes already used in the image are dimmed but stay clickable, because reassigning a
number is exactly how a swap gets fixed.

**Resolution.** Inference runs at `inference_long_side`; contours are mapped back to
the uploaded photograph's coordinates before they leave the API. A saved mask is at
sensor resolution, not at the working resolution.

**Add and erase are vertex-level, not boolean polygon operations.** Erasing drops the
vertices the stroke covers rather than computing an exact difference. At the scale a
clinician corrects a crown boundary the difference is not visible, and it keeps the
editing surface dependency-free. If exact boolean geometry is ever needed, that is
the place to change.

## Limits

Stated rather than implied, because the manuscript's claims depend on them.

- **No reviewer identity, no timing log, no revision history.** The backend records
  no reviewer, no start or finish time, no active editing time and no action counts.
  An annotation-effort study cannot be run against this build, and the paper
  therefore makes **no claim about annotation time**.
- **No authentication.** Anyone who can reach the port can read and write cases. It
  binds to `127.0.0.1` by default for that reason.
- **No task queue or assignment.** A clinician uploads one patient at a time; there
  is no manifest-driven work list and no "mark complete".
- **Cases live in memory**, bounded to the eight most recent, and are lost on
  restart. Saving writes to disk; not saving loses the corrections.
- **One patient at a time on one GPU.** All model calls serialise on a single lock,
  so a second concurrent user waits rather than corrupting CUDA state.
- Undo history is in the browser and is not persisted.

## Provenance of the reference annotations

The masks in the released dataset were **not** produced with this viewer. They were
produced with a predecessor correction tool, which recorded no reviewer identity or
timing either. The audit reports which annotation files still have the raw pipeline
shape — FDI codes and boxes with no polygon geometry — which is direct evidence that
they never went through a correction interface.

## Checking that the viewer serves the benchmarked pipeline

```bash
cd app && make smoke
```

**This needs the dataset.** It reads `<derived_root>/manifest.csv`,
`<derived_root>/splits.json` and the patient images named by the manifest, all
resolved from `configs/dataset_final_1000.yaml` — so it runs against a prepared
cohort, not against a fresh checkout. Without the dataset it exits with
`no complete val patient with readable images`.

It uploads one held-out patient's five photographs through the real HTTP API, runs both
segmenters, and asserts that the predicted views match the manifest, that every FDI
code is in the permanent-dentition vocabulary, that the instance counts are within
tolerance of the reference, and that a saved mask comes back at the uploaded
resolution. It is a wiring check, not a metric — the numbers come from
`scripts/evaluate_segmentation.py` over the whole split. What it catches is a viewer
that has drifted into running a different pipeline than the one that was measured.
