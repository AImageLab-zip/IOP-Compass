# 🦷 IOP-Compass

<p align="center">
  <img src="Assets/UI.png" alt="The IOP-Compass viewer: view classification, segmentation and FDI correction" width="960"/>
</p>

A clinical viewer and annotation tool for multi-view intraoral photographs.
Standardised five-view photographs are assigned their clinical view, segmented
into FDI-numbered tooth instances, and presented in a correction interface where a
clinician verifies and fixes the result. Built on human-verified FDI tooth-instance
annotations, with baselines for view classification and tooth instance
segmentation.

---

## The pipeline

A clinician uploads a patient's photographs. Each one is assigned its clinical view,
then segmented into FDI-numbered tooth instances:

```
photographs → view classification → SegmentAnyTooth (full image) → post-processing → FDI instances
                  ResNet18              no ROI stage                  frozen rules
```

Two design choices in that chain:

- **No ROI stage.** The viewer segments the full image and crops nothing.
- **Post-processing on.** Frozen rules clean up the segmenter output; they run after
  the forward pass in both segmenter paths.

**Mask R-CNN** is offered as the second segmenter, also on the full image, and is
faster. Both are selectable in the viewer.

---

## Dataset & Weights

> **Download:** <https://ditto.ing.unimore.it/iop-compass/>
>
> Note: for model weights drop us an email at federico.bolelli[at]unimore.it 

### Where everything goes

Four things have to be put in place by hand. This is the layout to end up with —
`third_party/` is *generated* and should never be populated manually:

```
IOP-Compass/
├── weights/
│   ├── view_classifier.pt              ← rename/symlink of classifier__*.pt
│   ├── maskrcnn.pt                     ← rename/symlink of maskrcnn__*.pt
│   └── SegmentAnyTooth weights/        ← the SAT weight directory, unpacked as-is
│       ├── segmentanytooth_vit_tiny.pt
│       ├── segmentanytooth_yolo11_front.pt
│       ├── segmentanytooth_yolo11_upper.pt
│       ├── segmentanytooth_yolo11_lower.pt
│       └── segmentanytooth_yolo11_right.pt
├── inputs/
│   └── SegmentAnyTooth/                ← the MIT vendor code, git-cloned
│       ├── sam.py
│       └── utils.py
└── third_party/                        ← symlinks, created by fetch_third_party.py
    ├── sat_weights      -> ../weights/SegmentAnyTooth weights
    ├── segmentanytooth_src -> ../inputs/SegmentAnyTooth
    └── torch_home
```

`weights/` and `inputs/` are the only two directories you touch. Both are
git-ignored, so nothing you put there can be committed by accident.

#### 1. The two release weights

| file | what it is |
| --- | --- |
| `view_classifier.pt` | ResNet18 five-view orientation classifier — assigns each photograph its clinical view |
| `maskrcnn.pt` | Mask R-CNN R50-FPN tooth-instance segmenter |

They are emailed under the name of the training run that produced them, so rename or
symlink them to the two names above — the viewer does no filename guessing and will
refuse to start otherwise:

```bash
cd weights
ln -s classifier__*.pt view_classifier.pt
ln -s maskrcnn__*.pt   maskrcnn.pt
cd ..
```

#### 2. SegmentAnyTooth — code and weights are licensed separately

The **code** is MIT. It is not redistributed here only because it is upstream's, so
clone it yourself; no agreement and no email are needed. It must land in
`inputs/SegmentAnyTooth`, with `sam.py` at the top level of that directory:

```bash
git clone https://github.com/thangngoc89/SegmentAnyTooth inputs/SegmentAnyTooth
```

The **weights** are covered by a separate non-commercial licence — email the
maintainers. They arrive as a directory literally named `SegmentAnyTooth weights`
(with the space); move it under `weights/` without renaming it or flattening it:

```bash
mv ~/Downloads/'SegmentAnyTooth weights' weights/
```

#### 3. Link them

```bash
python scripts/fetch_third_party.py
```

This downloads nothing and accepts no licence. It links the two locations above into
`third_party/`, records their SHA-256, creates `third_party/torch_home`, and exits
non-zero naming anything still missing. `SAT_CODE_DIR` and `SAT_WEIGHT_DIR` override
the source locations if you keep them elsewhere; `cd app && make check` then confirms
every file the enabled segmenters need.

The viewer applies no ROI stage, so no ROI-detector, geometric-prior or SAM 3 weights
are needed. SAM 3 is used only by the ROI arm of the benchmark, and
`fetch_third_party.py` therefore treats it as optional.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt    # pins the +cu126 torch index itself
python scripts/fetch_third_party.py     # link the SegmentAnyTooth code and weights
```

The lock file carries its own `--extra-index-url`, because the pinned torch and
torchvision wheels have a `+cu126` local version and are not on PyPI. It installs
everything the viewer needs, Flask included.

The third step assumes the weights and the vendor code are already in place — see
[Where everything goes](#where-everything-goes) above.

**Run the viewer** (API on :5000, UI on <http://localhost:5173>):

```bash
cd app && make install     # frontend dependencies, once
make check                 # confirm every weight is present before starting
make dev                   # backend and UI together
```

`make check` names any missing file and the environment variable that relocates it,
rather than failing on the first clinical image. To serve one segmenter only:
`make backend SEGMENTERS=sat`.

**Run the models on your own images**, without the viewer — see
[`docs/inference.md`](docs/inference.md) for the copy-pasteable version:

```python
from iop_compass.segmentation.segmentanytooth_adapter import SegmentAnyToothRunner
from iop_compass.segmentation.postprocessing import PostProcessParams, postprocess_instances

runner = SegmentAnyToothRunner(weight_dir="third_party/sat_weights",
                               code_dir="third_party/segmentanytooth_src",
                               device="cuda")
prediction = runner.predict(image_bgr, view_label="frontal")
masks, fdis, scores, _ = postprocess_instances(
    prediction.masks, prediction.fdis, prediction.scores,
    roi_mask=None, exclusion_mask=None,
    params=PostProcessParams.from_yaml("configs/segmentation/postprocessing.yaml"),
    view_label="frontal",
)
```

---

## Layout

```
app/                  the viewer: Flask backend + React frontend
configs/              dataset, classification and segmentation configuration
docs/                 inference and the viewer
internal/             cluster-specific submission tooling, not needed to use the release
scripts/              audit, manifest, splits, training, inference, evaluation, figures
src/iop_compass/
  data/               adapter, manifest, rasterisation, validation, splits, imaging
  classification/     dataset, model, training, constrained assignment, metrics
  roi/                the four ROI strategies, factory, ROI-only evaluation
  segmentation/       SegmentAnyTooth adapter, Mask R-CNN, post-processing,
                      matching, metrics, the benchmark grid, inference
  reporting/          aggregation, bootstrap, figures, labels, qualitative panels
tests/                leakage, manifest, label mapping, matching, metrics, grid
```

`results/` and `runs/` are produced by a campaign and are not tracked.

---

## Benchmark tasks

| task | variants |
| --- | --- |
| View classification | C0 pretrained, no augmentation · C1 pretrained + clinically plausible augmentation · C2 from scratch. C1 also evaluated with patient-set constrained assignment. |
| ROI extraction | full image · view-specific geometric prior fitted on training annotations · learned single-box detector · SAM 3 concept prompts with a documented full-image fallback. |
| Tooth instance segmentation | The full ROI × segmenter × post-processing grid: 4 × 2 × 2 = 16 cells, every cell through one code path so the axes are not confounded with a per-backend crop convention. |

---

## Citation

The dataset is publicly available at
<https://ditto.ing.unimore.it/iop-compass/>.
