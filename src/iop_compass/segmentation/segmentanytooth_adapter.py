"""Adapter around the SegmentAnyTooth reference implementation.

The upstream ``segmentanytooth.predict`` entry point is kept as the definition of
the method, but it is unusable for a benchmark as written:

* it reloads the YOLO11 detector *and* the SAM-HQ ViT-Tiny segmenter on every
  call (``segmentanytooth.py:60-62``);
* ``sam_load`` never moves the model off the CPU, so ``sam.device`` is ``cpu``
  and every mask is decoded on the CPU (``sam.py:29-36``);
* it collapses the per-instance masks into a single ``uint8`` raster keyed by FDI
  code (``segmentanytooth.py:105-110``), so two detections sharing an FDI code
  silently merge and per-instance confidences are discarded.

This adapter keeps the model weights, the detector, the class-name mapping and
the left-view mirroring **exactly** as upstream, and only

1. caches the loaded models across calls,
2. places both models on the requested device,
3. returns the per-instance masks together with their detector confidence, plus a
   flat FDI raster (one uint8 label per pixel) for the rule-based post-processing
   and the qualitative figures.

No weight is modified and no thresholds are introduced: YOLO runs with its own
defaults, as upstream.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# SAT views, keyed by the canonical view label used throughout this repo.
VIEW_TO_SAT = {
    "frontal": "front",
    "left_buccal": "left",
    "right_buccal": "right",
    "upper_occlusal": "upper",
    "lower_occlusal": "lower",
}

# Verbatim from segmentanytooth.py:29-33 - the left lateral detector is the right
# lateral model applied to a horizontally flipped image, with remapped classes.
LEFT_CLASSES = [
    "le28", "le27", "le26", "le25", "le24", "le23", "le22", "le21",
    "le38", "le37", "le36", "le35", "le34", "le33", "le32", "le31",
    "le11", "le12", "le13", "le14", "le41", "le42", "le43", "le44",
]


@dataclass
class SatInstance:
    mask: np.ndarray  # bool, full image size
    fdi: int | None
    score: float
    bbox: tuple[float, float, float, float]


@dataclass
class SatPrediction:
    instances: list[SatInstance] = field(default_factory=list)
    fdi_raster: np.ndarray | None = None
    n_detections: int = 0

    @property
    def masks(self) -> list[np.ndarray]:
        return [i.mask for i in self.instances]

    @property
    def fdis(self) -> list[int | None]:
        return [i.fdi for i in self.instances]

    @property
    def scores(self) -> list[float]:
        return [i.score for i in self.instances]


class SegmentAnyToothRunner:
    """Cached SegmentAnyTooth runner."""

    def __init__(
        self,
        weight_dir: str | Path,
        code_dir: str | Path | None = None,
        device: str = "cuda",
        sam_batch_size: int = 16,
    ):
        self.weight_dir = Path(weight_dir)
        self.device = device
        self.sam_batch_size = sam_batch_size
        if code_dir is not None:
            code_dir = str(Path(code_dir).resolve())
            if code_dir not in sys.path:
                sys.path.insert(0, code_dir)
        self._yolo: dict[str, object] = {}
        self._sam = None
        self._sam_predict = None

    # ------------------------------------------------------------------ models
    def _weight_path(self, kind: str) -> Path:
        if kind == "sam":
            return self.weight_dir / "segmentanytooth_vit_tiny.pt"
        if kind == "left":
            kind = "right"
        return self.weight_dir / f"segmentanytooth_yolo11_{kind}.pt"

    def _get_sam(self):
        if self._sam is None:
            from sam import sam_load, sam_predict  # vendored SAT module

            path = self._weight_path("sam")
            if not path.exists():
                raise FileNotFoundError(f"missing SegmentAnyTooth SAM weights: {path}")
            sam = sam_load(str(path))
            sam = sam.to(self.device)
            sam.eval()
            self._sam = sam
            self._sam_predict = sam_predict
        return self._sam, self._sam_predict

    def _get_yolo(self, sat_view: str):
        key = "right" if sat_view == "left" else sat_view
        if key not in self._yolo:
            from ultralytics import YOLO

            path = self._weight_path(key)
            if not path.exists():
                raise FileNotFoundError(f"missing SegmentAnyTooth detector: {path}")
            self._yolo[key] = YOLO(model=str(path))
        return self._yolo[key]

    def weight_checksums(self) -> dict[str, str]:
        from ..data.adapter import sha256_file

        out = {}
        for name in sorted(self.weight_dir.glob("*.pt")):
            out[name.name] = sha256_file(name)
        return out

    # --------------------------------------------------------------- inference
    def predict(self, image_bgr: np.ndarray, view_label: str) -> SatPrediction:
        """Run SegmentAnyTooth on a BGR image.

        The control flow mirrors ``segmentanytooth.predict`` exactly, including
        the left-view flip/unflip and the class-id sort.
        """
        sat_view = VIEW_TO_SAT.get(view_label, view_label)
        should_flip = sat_view == "left"

        image = image_bgr
        if should_flip:
            image = cv2.flip(image, 1)

        yolo = self._get_yolo(sat_view)
        result = yolo.predict(
            image,
            save=False,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            project=None,
            verbose=False,
            device=self.device,
        )[0]

        h, w = image.shape[:2]
        if result.boxes is None or len(result.boxes) == 0:
            return SatPrediction(
                instances=[], fdi_raster=np.zeros((h, w), dtype=np.uint8), n_detections=0
            )

        names = result.names if not should_flip else LEFT_CLASSES
        boxes = result.boxes.xyxy.squeeze(0).cpu().numpy().reshape(-1, 4)
        classes = result.boxes.cls.squeeze(0).cpu().numpy().astype(np.int32).reshape(-1)
        scores = result.boxes.conf.squeeze(0).cpu().numpy().astype(np.float32).reshape(-1)

        order = np.argsort(classes)
        classes, boxes, scores = classes[order], boxes[order], scores[order]

        if should_flip:
            image = cv2.flip(image, 1)
            flipped = boxes.copy()
            flipped[:, [0, 2]] = w - flipped[:, [2, 0]]
            boxes = flipped

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam, sam_predict = self._get_sam()
        sam_masks = sam_predict(
            sam=sam, boxes_xyxy=boxes, image=rgb, batch_size=self.sam_batch_size
        )

        instances: list[SatInstance] = []
        raster = np.zeros((h, w), dtype=np.uint8)
        for cls_id, mask, score, box in zip(classes, sam_masks, scores, boxes):
            name = names[int(cls_id)]
            try:
                fdi = int(str(name)[-2:])
            except ValueError:
                fdi = None
            binary = np.asarray(mask).astype(bool)
            if not binary.any():
                continue
            instances.append(
                SatInstance(
                    mask=binary,
                    fdi=fdi,
                    score=float(score),
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                )
            )
            if fdi is not None:
                raster[binary] = fdi

        return SatPrediction(
            instances=instances, fdi_raster=raster, n_detections=len(classes)
        )


def resolve_weight_dir(explicit: str | None = None) -> Path:
    """Locate the SegmentAnyTooth weights.

    Order: explicit argument, ``SAT_WEIGHT_DIR`` environment variable, the
    repository ``third_party/sat_weights`` link.  The weights are covered by a
    non-commercial licence and are therefore never committed.
    """
    if explicit:
        return Path(explicit)
    env = os.environ.get("SAT_WEIGHT_DIR")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[3]
    return repo / "third_party" / "sat_weights"


def resolve_code_dir(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("SAT_CODE_DIR")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[3]
    return repo / "third_party" / "segmentanytooth_src"
