"""The clinical pipeline: view classification, then segmentation, then post-processing.

No ROI stage runs in either segmenter path: both segment the full image, and
post-processing runs on the result.

Both segmenters are offered:

``sat``
    SegmentAnyTooth on the full image with post-processing — the viewer's default.
``mask-rcnn``
    Mask R-CNN R50-FPN trained in-dataset; faster.

Every stage calls the shared model modules
(:class:`~iop_compass.segmentation.segmentanytooth_adapter.SegmentAnyToothRunner`,
:class:`~iop_compass.segmentation.mask_rcnn.MaskRcnnPredictor`,
:func:`~iop_compass.segmentation.postprocessing.postprocess_instances`), not a
re-implementation of them.

Model calls are serialised by :attr:`Pipeline.lock`.  One process holds one copy of
each model on one device; concurrent requests would interleave CUDA work on shared
module state.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from config import Settings
from contours import (
    mask_to_polygons,
    mirror_polygons_x,
    polygons_area,
    polygons_bbox,
    polygons_centroid,
)
from iop_compass.classification.constrained_assignment import constrained_assign
from iop_compass.classification.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    VIEW_CLASSES,
)
from iop_compass.classification.models import build_resnet18, load_checkpoint
from iop_compass.data.imaging import target_size
from iop_compass.roi.base import full_image_result
from iop_compass.segmentation.postprocessing import (
    PostProcessParams,
    postprocess_instances,
)

log = logging.getLogger(__name__)

SEGMENTERS = ("sat", "mask-rcnn")
SEGMENTER_LABELS = {
    "sat": "SegmentAnyTooth",
    "mask-rcnn": "Mask R-CNN",
}
#: The grid cell each viewer segmenter corresponds to, recorded in every response so
#: a result can always be traced to the row of the benchmark table it came from.
SEGMENTER_CELLS = {
    "sat": "no-roi+sat+post",
    "mask-rcnn": "no-roi+mask-rcnn+post",
}

#: The full FDI permanent-dentition vocabulary; 19, 20, 29, 30, 39, 40 do not exist.
FDI_CODES = tuple(
    int(f"{quadrant}{tooth}")
    for quadrant in (1, 2, 3, 4)
    for tooth in range(1, 9)
)


@dataclass
class ViewPrediction:
    image_id: str
    view: str
    confidence: float
    probabilities: dict[str, float]
    #: True when the label came from the patient-set constrained assignment rather
    #: than from this image's own argmax.
    constrained: bool = False


@dataclass
class Instance:
    instance_id: str
    fdi: int | None
    score: float
    contours: list[list[float]]
    area: float
    bbox: tuple[float, float, float, float]
    centroid: tuple[float, float]

    def to_dict(self) -> dict:
        x0, y0, x1, y1 = self.bbox
        cx, cy = self.centroid
        return {
            "instance_id": self.instance_id,
            "fdi": self.fdi,
            "score": round(self.score, 4),
            "contours": self.contours,
            "area": round(self.area, 1),
            "bbox": [round(v, 1) for v in (x0, y0, x1, y1)],
            "centroid": [round(cx, 1), round(cy, 1)],
        }


@dataclass
class SegmentationResult:
    instances: list[Instance]
    segmenter: str
    cell: str
    postprocess: bool
    width: int
    height: int
    timings: dict[str, float] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "instances": [i.to_dict() for i in self.instances],
            "segmenter": self.segmenter,
            "segmenter_label": SEGMENTER_LABELS[self.segmenter],
            "cell": self.cell,
            "postprocess": self.postprocess,
            "width": self.width,
            "height": self.height,
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
            "stats": self.stats,
        }


class Pipeline:
    """Holds the loaded models and runs the two stages.

    Models load lazily on first use so start-up is quick, but their weights are
    validated eagerly by :meth:`Settings.missing` before the server binds a port.
    """

    def __init__(self, settings: Settings, segmenters: tuple[str, ...] = SEGMENTERS):
        self.settings = settings
        self.segmenters = tuple(s for s in segmenters if s in SEGMENTERS)
        self.lock = threading.Lock()
        self.device = self._resolve_device(settings.device)
        self.post_params = self._load_post_params()

        self._classifier: torch.nn.Module | None = None
        self._sat = None
        self._maskrcnn = None

    # ------------------------------------------------------------------ set-up
    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA requested but unavailable; falling back to CPU")
            return "cpu"
        return requested

    def _load_post_params(self) -> PostProcessParams:
        return PostProcessParams.from_yaml(self.settings.postprocessing_config)

    def device_name(self) -> str:
        if self.device.startswith("cuda"):
            return torch.cuda.get_device_name(0)
        return "CPU"

    def warm_up(self) -> None:
        """Load every enabled model now, so the first clinical request is not the slow one."""
        self.classifier()
        for segmenter in self.segmenters:
            model = self._segmenter(segmenter)
            # SegmentAnyToothRunner defers its own weights until the first predict, so
            # constructing it is not loading it.
            if hasattr(model, "load"):
                model.load()

    # ------------------------------------------------------------------ models
    def classifier(self) -> torch.nn.Module:
        if self._classifier is None:
            log.info("loading view classifier from %s", self.settings.classifier_weights)
            model = build_resnet18(num_classes=len(VIEW_CLASSES), pretrained=False)
            load_checkpoint(model, self.settings.classifier_weights)
            self._classifier = model.eval().to(self.device)
        return self._classifier

    def _segmenter(self, name: str):
        if name == "sat":
            if self._sat is None:
                from iop_compass.segmentation.segmentanytooth_adapter import (
                    SegmentAnyToothRunner,
                )

                log.info("loading SegmentAnyTooth from %s", self.settings.sat_weight_dir)
                self._sat = SegmentAnyToothRunner(
                    weight_dir=self.settings.sat_weight_dir,
                    code_dir=self.settings.sat_code_dir,
                    device=self.device,
                )
            return self._sat

        if name == "mask-rcnn":
            if self._maskrcnn is None:
                from iop_compass.segmentation.mask_rcnn import MaskRcnnPredictor

                log.info("loading Mask R-CNN from %s", self.settings.maskrcnn_weights)
                self._maskrcnn = MaskRcnnPredictor(
                    self.settings.maskrcnn_weights, device=self.device
                )
            return self._maskrcnn

        raise ValueError(f"unknown segmenter {name!r}; expected one of {SEGMENTERS}")

    # ------------------------------------------------------- stage 1: the view
    def classify(self, images: dict[str, np.ndarray]) -> list[ViewPrediction]:
        """Predict the clinical view of every image of one patient.

        With exactly five images the labels are resolved jointly under a one-to-one
        image/view constraint (:func:`constrained_assign`), which is the
        configuration measured at 100 % accuracy on the validation split: a
        standardised acquisition contains each view exactly once, so an independent
        argmax that returns two frontals is knowably wrong.  Any other count falls
        back to the per-image argmax.
        """
        image_ids = list(images)
        if not image_ids:
            return []

        batch = torch.stack(
            [self._classifier_tensor(images[i]) for i in image_ids]
        ).to(self.device)

        with self.lock, torch.no_grad():
            logits = self.classifier()(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        constrained = len(image_ids) == len(VIEW_CLASSES)
        indices = (
            constrained_assign(probs) if constrained else probs.argmax(axis=1)
        )

        return [
            ViewPrediction(
                image_id=image_id,
                view=VIEW_CLASSES[int(index)],
                confidence=float(row[int(index)]),
                probabilities={
                    view: float(p) for view, p in zip(VIEW_CLASSES, row)
                },
                constrained=constrained,
            )
            for image_id, index, row in zip(image_ids, indices, probs)
        ]

    def _classifier_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        size = self.settings.classifier_image_size
        resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (tensor - mean) / std

    # ------------------------------------------------ stage 2: the segmentation
    def segment(
        self,
        image_bgr: np.ndarray,
        view: str,
        segmenter: str = "sat",
        postprocess: bool = True,
        flipped: bool = False,
    ) -> SegmentationResult:
        """Segment one image and return instances in its original coordinates.

        The image is downscaled to ``inference_long_side`` for the forward pass, the
        same bound the benchmark used, and every contour is scaled back afterwards
        so an exported mask matches the uploaded photograph pixel for pixel.

        ``flipped`` says that ``image_bgr`` is already a horizontal mirror of the
        photograph the clinician uploaded -- the compensation for an acquisition made
        through an intraoral mirror, applied at ingest so the detector and its
        side-specific FDI class table run in the convention they were trained in.
        The contours are reflected back at the end, so a caller always receives the
        uploaded photograph's coordinates whichever way the image came in.
        """
        if segmenter not in self.segmenters:
            raise ValueError(
                f"segmenter {segmenter!r} is not enabled; available: {self.segmenters}"
            )
        if view not in VIEW_CLASSES:
            raise ValueError(f"unknown view {view!r}; expected one of {VIEW_CLASSES}")

        height, width = image_bgr.shape[:2]
        work_w, work_h = target_size(width, height, self.settings.inference_long_side)
        work = (
            image_bgr
            if (work_w, work_h) == (width, height)
            else cv2.resize(image_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
        )

        timings: dict[str, float] = {}
        with self.lock:
            start = time.perf_counter()
            masks, fdis, scores = self._forward(work, view, segmenter)
            timings["segment"] = time.perf_counter() - start

            stats: dict = {}
            if postprocess:
                # The full image is the region: no ROI mask constrains the result,
                # and no exclusion mask marks anything as definitely-not-tooth.
                roi = full_image_result((work_h, work_w), strategy="R0_full")
                start = time.perf_counter()
                masks, fdis, scores, post_stats = postprocess_instances(
                    masks,
                    fdis,
                    scores,
                    roi_mask=roi.roi_mask,
                    exclusion_mask=roi.exclusion_mask,
                    params=self.post_params,
                    view_label=view,
                )
                timings["postprocess"] = time.perf_counter() - start
                stats = post_stats.to_dict()

        scale_x = width / float(work_w)
        scale_y = height / float(work_h)

        instances: list[Instance] = []
        for index, (mask, fdi, score) in enumerate(zip(masks, fdis, scores)):
            contours = mask_to_polygons(mask, scale_x, scale_y)
            if not contours:
                continue
            if flipped:
                # Back into the uploaded photograph's frame, before the derived
                # geometry below is computed from these points.
                contours = mirror_polygons_x(contours, width)
            instances.append(
                Instance(
                    instance_id=f"i{index}",
                    fdi=int(fdi) if fdi is not None else None,
                    score=float(score),
                    contours=contours,
                    area=polygons_area(contours),
                    bbox=polygons_bbox(contours),
                    centroid=polygons_centroid(contours),
                )
            )

        instances.sort(key=lambda i: (i.fdi is None, i.fdi or 0))
        return SegmentationResult(
            instances=instances,
            segmenter=segmenter,
            cell=SEGMENTER_CELLS[segmenter],
            postprocess=postprocess,
            width=width,
            height=height,
            timings=timings,
            stats=stats,
        )

    def _forward(self, image_bgr: np.ndarray, view: str, segmenter: str):
        model = self._segmenter(segmenter)
        if segmenter == "sat":
            prediction = model.predict(image_bgr, view)
            return prediction.masks, prediction.fdis, prediction.scores
        masks, fdis, scores = model.predict(image_bgr)
        return masks, fdis, scores
