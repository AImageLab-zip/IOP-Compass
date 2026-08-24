"""Supervised in-dataset baseline: Mask R-CNN ResNet50-FPN.

COCO-pretrained backbone, classification and mask heads replaced for the FDI label
set actually present in the dataset (32 permanent-tooth codes plus background).
Trained on training patients only; the checkpoint is selected on validation
FDI-aware instance F1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from ..data.rasterize import load_instance_raster, load_instance_sidecar


@dataclass
class MaskRcnnConfig:
    epochs: int = 30
    batch_size: int = 2
    lr: float = 5.0e-3
    weight_decay: float = 1.0e-4
    momentum: float = 0.9
    num_workers: int = 4
    amp: bool = True
    score_threshold: float = 0.05
    mask_threshold: float = 0.5
    max_detections: int = 64
    hflip_prob: float = 0.0
    early_stopping_patience: int = 8
    trainable_backbone_layers: int = 3

    @classmethod
    def from_dict(cls, payload: dict | None) -> "MaskRcnnConfig":
        payload = dict(payload or {})
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


def build_maskrcnn(
    fdi_labels: tuple[int, ...],
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    max_detections: int = 64,
) -> torch.nn.Module:
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    # see build_detector: weights_backbone=None keeps checkpoint loading offline
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None if not pretrained else "DEFAULT",
        trainable_backbone_layers=trainable_backbone_layers,
        box_detections_per_img=max_detections,
    )
    num_classes = len(fdi_labels) + 1  # background + one class per FDI code
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, num_classes)
    return model


class ToothInstanceDataset(Dataset):
    """Cached image + reference instance raster -> torchvision detection target."""

    def __init__(
        self,
        items: list[tuple[Path, Path, Path]],
        fdi_to_index: dict[int, int],
        hflip_prob: float = 0.0,
        flip_fdi: dict[int, int] | None = None,
    ):
        self.items = items
        self.fdi_to_index = fdi_to_index
        self.hflip_prob = hflip_prob
        self.flip_fdi = flip_fdi or {}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_path, raster_path, sidecar_path = self.items[index]
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"cannot read {image_path}")
        raster = load_instance_raster(raster_path)
        sidecar = load_instance_sidecar(sidecar_path)
        if img.shape[:2] != raster.shape[:2]:
            img = cv2.resize(
                img, (raster.shape[1], raster.shape[0]), interpolation=cv2.INTER_AREA
            )

        masks, labels, boxes = [], [], []
        for row in sidecar["instances"]:
            fdi = row.get("fdi")
            if fdi is None or fdi not in self.fdi_to_index:
                continue
            mask = raster == int(row["instance_id"])
            if not mask.any():
                continue
            ys, xs = np.nonzero(mask)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            if x1 <= x0 or y1 <= y0:
                continue
            masks.append(mask)
            labels.append(self.fdi_to_index[fdi])
            boxes.append([x0, y0, x1 + 1, y1 + 1])

        if self.hflip_prob and np.random.random() < self.hflip_prob:
            img = img[:, ::-1].copy()
            w = img.shape[1]
            masks = [m[:, ::-1].copy() for m in masks]
            boxes = [[w - b[2], b[1], w - b[0], b[3]] for b in boxes]
            index_to_fdi = {v: k for k, v in self.fdi_to_index.items()}
            labels = [
                self.fdi_to_index.get(
                    self.flip_fdi.get(index_to_fdi[label], index_to_fdi[label]), label
                )
                for label in labels
            ]

        tensor = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255.0)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": (
                torch.as_tensor(np.stack(masks), dtype=torch.uint8)
                if masks
                else torch.zeros((0, *raster.shape), dtype=torch.uint8)
            ),
            "image_id": torch.tensor([index]),
        }
        return tensor, target


def collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def mirror_fdi(fdi: int) -> int:
    """FDI code of the contralateral tooth (quadrant 1<->2, 4<->3)."""
    quadrant, position = divmod(fdi, 10)
    mirrored = {1: 2, 2: 1, 3: 4, 4: 3}.get(quadrant, quadrant)
    return mirrored * 10 + position


class MaskRcnnPredictor:
    def __init__(self, checkpoint: str | Path, device: str = "cuda"):
        self.checkpoint = Path(checkpoint)
        state = torch.load(str(self.checkpoint), map_location="cpu", weights_only=False)
        self.fdi_labels: tuple[int, ...] = tuple(state["fdi_labels"])
        self.config = MaskRcnnConfig.from_dict(state.get("config"))
        self.index_to_fdi = {i + 1: fdi for i, fdi in enumerate(self.fdi_labels)}
        self.model = build_maskrcnn(
            self.fdi_labels,
            pretrained=False,
            trainable_backbone_layers=self.config.trainable_backbone_layers,
            max_detections=self.config.max_detections,
        )
        self.model.load_state_dict(state["model_state"])
        self.model.eval().to(device)
        self.device = device

    def checkpoint_sha256(self) -> str:
        from ..data.adapter import sha256_file

        return sha256_file(self.checkpoint)

    @torch.no_grad()
    def predict(
        self, image_bgr: np.ndarray
    ) -> tuple[list[np.ndarray], list[int | None], list[float]]:
        tensor = (
            torch.from_numpy(image_bgr[:, :, ::-1].copy())
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
            .to(self.device)
        )
        out = self.model([tensor])[0]
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        masks = out["masks"].cpu().numpy()[:, 0]
        keep = scores >= self.config.score_threshold
        out_masks, out_fdis, out_scores = [], [], []
        for mask, label, score in zip(masks[keep], labels[keep], scores[keep]):
            binary = mask >= self.config.mask_threshold
            if not binary.any():
                continue
            out_masks.append(binary)
            out_fdis.append(self.index_to_fdi.get(int(label)))
            out_scores.append(float(score))
        return out_masks, out_fdis, out_scores
