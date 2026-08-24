"""R2 - lightweight learned single-box ROI.

A Faster R-CNN with a MobileNetV3-Large 320 FPN backbone is trained to predict a
single box: the bounding box of the union of the reference tooth masks, expanded
by a fixed margin.  One class, one box per image.  Training uses training
patients only, and inference never sees reference data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .base import RoiResult, box_to_masks, clip_box, full_image_result


@dataclass
class LearnedRoiConfig:
    target_margin_frac: float = 0.05
    epochs: int = 12
    batch_size: int = 4
    lr: float = 5.0e-3
    weight_decay: float = 1.0e-4
    momentum: float = 0.9
    score_threshold: float = 0.30
    num_workers: int = 4
    amp: bool = True
    inference_margin_frac: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict | None) -> "LearnedRoiConfig":
        payload = dict(payload or {})
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


def build_detector(pretrained: bool = True) -> torch.nn.Module:
    weights = (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1 if pretrained else None
    )
    # weights_backbone must be pinned to None as well: torchvision otherwise
    # downloads the ImageNet backbone even when weights=None, which turns loading a
    # trained checkpoint into a network call on a compute node.
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights, weights_backbone=None if not pretrained else "DEFAULT"
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # two classes: background + intraoral region
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model


def target_box_from_raster(
    raster: np.ndarray, margin_frac: float
) -> tuple[int, int, int, int] | None:
    """Expanded bounding box of the tooth union in a reference instance raster."""
    ys, xs = np.nonzero(raster)
    if xs.size == 0:
        return None
    h, w = raster.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    mx = margin_frac * max(x1 - x0 + 1, 1)
    my = margin_frac * max(y1 - y0 + 1, 1)
    return clip_box((x0 - mx, y0 - my, x1 + 1 + mx, y1 + 1 + my), (h, w))


class LearnedRoi:
    """Callable ROI strategy backed by the trained single-box detector."""

    name = "R2_learned"

    def __init__(
        self,
        checkpoint: str | Path,
        config: LearnedRoiConfig | None = None,
        device: str = "cuda",
    ):
        self.config = config or LearnedRoiConfig()
        self.device = device
        self.checkpoint = Path(checkpoint)
        self.model = build_detector(pretrained=False)
        state = torch.load(str(self.checkpoint), map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model_state"])
        self.model.eval().to(device)

    def checkpoint_sha256(self) -> str:
        from ..data.adapter import sha256_file

        return sha256_file(self.checkpoint)

    @torch.no_grad()
    def __call__(self, image_bgr: np.ndarray, view_label: str) -> RoiResult:
        h, w = image_bgr.shape[:2]
        start = time.perf_counter()
        rgb = image_bgr[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        outputs = self.model([tensor.to(self.device)])[0]
        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()
        runtime = time.perf_counter() - start

        keep = scores >= self.config.score_threshold
        if not keep.any():
            result = full_image_result(
                (h, w), self.name, True, "no box above score threshold"
            )
            result.runtime_s = runtime
            return result

        best = int(np.argmax(scores[keep]))
        box = boxes[keep][best]
        margin = self.config.inference_margin_frac
        if margin:
            bw, bh = box[2] - box[0], box[3] - box[1]
            box = [
                box[0] - margin * bw,
                box[1] - margin * bh,
                box[2] + margin * bw,
                box[3] + margin * bh,
            ]
        clipped = clip_box(box, (h, w))
        roi, exclusion = box_to_masks(clipped, (h, w))
        return RoiResult(
            roi_mask=roi,
            exclusion_mask=exclusion,
            box=clipped,
            strategy=self.name,
            runtime_s=runtime,
            extra={"score": float(scores[keep][best])},
        )
