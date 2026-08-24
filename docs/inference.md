# Running the models on your own images

This is the released pipeline without the viewer and without the benchmark
machinery: five-view classification, then tooth instance segmentation with
post-processing.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt
export PYTHONPATH=$PWD/src:$PWD/third_party
export TORCH_HOME=$PWD/third_party/torch_home
```

Download the release weights into `weights/` (see the weights table in the
[README](../README.md)) and obtain the SegmentAnyTooth assets separately — they are
covered by a non-commercial licence and are not redistributed here. Then:

```bash
python scripts/fetch_third_party.py
```

It downloads nothing and accepts no licence for you: it links assets already present,
records their SHA-256, and exits non-zero listing anything missing.

## Classify the view

The classifier expects a 256×256 RGB crop with ImageNet normalisation.

```python
import cv2, numpy as np, torch
from iop_compass.classification.dataset import IMAGENET_MEAN, IMAGENET_STD, VIEW_CLASSES
from iop_compass.classification.models import build_resnet18, load_checkpoint

model = build_resnet18(num_classes=len(VIEW_CLASSES), pretrained=False)
load_checkpoint(model, "weights/view_classifier.pt")
model.eval().cuda()

def as_tensor(image_bgr, size=256):
    rgb = cv2.cvtColor(cv2.resize(image_bgr, (size, size)), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std

images = [cv2.imread(p) for p in paths]              # five views of one patient
batch = torch.stack([as_tensor(i) for i in images]).cuda()
with torch.no_grad():
    probs = torch.softmax(model(batch), dim=1).cpu().numpy()
```

**If you have exactly the five standard views of one patient, resolve them jointly.**
A standardised acquisition contains each view once, so a one-to-one assignment beats
five independent argmaxes — 1.0000 ± 0.0000 versus 0.9987 ± 0.0011 accuracy on
validation:

```python
from iop_compass.classification.constrained_assignment import constrained_assign

views = [VIEW_CLASSES[i] for i in constrained_assign(probs)]
```

Otherwise use `probs.argmax(axis=1)`.

## Segment the teeth

The view label selects the SegmentAnyTooth detector, so it must be right.

```python
from iop_compass.data.imaging import target_size
from iop_compass.segmentation.postprocessing import PostProcessParams, postprocess_instances
from iop_compass.segmentation.segmentanytooth_adapter import SegmentAnyToothRunner

runner = SegmentAnyToothRunner(weight_dir="third_party/sat_weights",
                               code_dir="third_party/segmentanytooth_src",
                               device="cuda")
params = PostProcessParams.from_yaml("configs/segmentation/postprocessing.yaml")

def segment(image_bgr, view, long_side=2048):
    h, w = image_bgr.shape[:2]
    tw, th = target_size(w, h, long_side)
    work = image_bgr if (tw, th) == (w, h) else cv2.resize(image_bgr, (tw, th),
                                                           interpolation=cv2.INTER_AREA)
    pred = runner.predict(work, view)
    masks, fdis, scores, stats = postprocess_instances(
        pred.masks, pred.fdis, pred.scores,
        roi_mask=None, exclusion_mask=None,   # no ROI: the full image is the region
        params=params, view_label=view,
    )
    return masks, fdis, scores            # masks are at (th, tw), not (h, w)
```

`roi_mask=None` and `exclusion_mask=None` are the point: **no ROI stage**. Every ROI
level in the benchmark scored below the full image, so cropping costs accuracy.

Masks come back at the working resolution. To get them at your sensor resolution,
scale the contours rather than the rasters — `app/backend/contours.py` does exactly
this and is reusable.

## Mask R-CNN instead

Roughly 12× faster, measurably less accurate (0.9220 ± 0.0089 versus 0.9495 ± 0.0046
FDI-aware F1). It needs no view label:

```python
from iop_compass.segmentation.mask_rcnn import MaskRcnnPredictor

predictor = MaskRcnnPredictor("weights/maskrcnn.pt", device="cuda")
masks, fdis, scores = predictor.predict(work)
masks, fdis, scores, stats = postprocess_instances(
    masks, fdis, scores, roi_mask=None, exclusion_mask=None,
    params=params, view_label=view,
)
```

## What to expect

On held-out validation patients, per image: FDI-aware instance F1 0.9495 ± 0.0046,
FDI accuracy among matches 0.9822 ± 0.0046, matched Dice 0.9850 ± 0.0001.

**Roughly 59 % of images and only about 10 % of patients come back with no error of
any kind.** Per-image accuracy near 0.95 does not survive being asked for five images
at once. Plan for review rather than for unattended use — that is what `app/` is for.

## Scoring your own predictions

If you have reference annotations, the benchmark's evaluator works on any pair of
instance lists:

```python
from iop_compass.segmentation.metrics import aggregate, evaluate_image

evals = [evaluate_image(image_id, patient_id, view, gt_masks, gt_fdis,
                        pred_masks, pred_fdis, pred_scores)[0]
         for ... in ...]
print(aggregate(evals)["fdi_instance_f1"])
```

Matching is Hungarian and class-agnostic, so a correct mask with the wrong number
counts as one FDI error rather than two errors. For confidence intervals use
`iop_compass.reporting.bootstrap`, which resamples **patients**, not images — all
five views of a patient move together.
