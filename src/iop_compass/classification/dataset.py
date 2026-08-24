"""View-classification dataset and augmentation.

Images are read from the pre-resized cache produced by
``scripts/build_image_cache.py`` so training never decodes 18 MB full-resolution
PNGs.

Horizontal flipping is the only geometric mirror used, and it is applied
*with* the label remap ``left_buccal <-> right_buccal``; the occlusal and frontal
labels are mirror-invariant.  Vertical flipping is not used: an upside-down
intraoral photograph is not a clinically plausible acquisition.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VIEW_CLASSES = (
    "frontal",
    "left_buccal",
    "right_buccal",
    "upper_occlusal",
    "lower_occlusal",
)

# label index -> label index under a horizontal mirror
FLIP_REMAP = {
    "frontal": "frontal",
    "left_buccal": "right_buccal",
    "right_buccal": "left_buccal",
    "upper_occlusal": "upper_occlusal",
    "lower_occlusal": "lower_occlusal",
}


@dataclass
class AugmentationConfig:
    """Clinically plausible augmentation envelope."""

    enabled: bool = True
    rotation_deg: float = 20.0
    translate_frac: float = 0.10
    scale_min: float = 0.90
    scale_max: float = 1.10
    brightness: float = 0.25
    contrast: float = 0.25
    saturation: float = 0.25
    hue: float = 0.03
    blur_prob: float = 0.20
    blur_sigma_max: float = 1.0
    noise_prob: float = 0.20
    noise_std: float = 0.02
    jpeg_prob: float = 0.20
    jpeg_quality_min: int = 60
    jpeg_quality_max: int = 95
    hflip_prob: float = 0.50
    vflip_prob: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict | None) -> "AugmentationConfig":
        payload = dict(payload or {})
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


def label_to_index(label: str) -> int:
    return VIEW_CLASSES.index(label)


FLIP_INDEX = tuple(label_to_index(FLIP_REMAP[c]) for c in VIEW_CLASSES)


@dataclass
class Sample:
    image_path: Path
    label: int
    patient_id: str
    image_id: str
    view_label: str


class ViewDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        image_size: int = 256,
        augment: AugmentationConfig | None = None,
        seed: int = 0,
    ):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment or AugmentationConfig(enabled=False)
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.to_tensor = transforms.ToTensor()
        self.affine = transforms.RandomAffine(
            degrees=self.augment.rotation_deg,
            translate=(self.augment.translate_frac, self.augment.translate_frac),
            scale=(self.augment.scale_min, self.augment.scale_max),
        )
        self.jitter = transforms.ColorJitter(
            brightness=self.augment.brightness,
            contrast=self.augment.contrast,
            saturation=self.augment.saturation,
            hue=self.augment.hue,
        )
        self._seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, img: Image.Image, label: int) -> tuple[Image.Image, int]:
        cfg = self.augment
        if cfg.hflip_prob and random.random() < cfg.hflip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = FLIP_INDEX[label]
        if cfg.vflip_prob and random.random() < cfg.vflip_prob:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = self.affine(img)
        img = self.jitter(img)
        if cfg.jpeg_prob and random.random() < cfg.jpeg_prob:
            quality = random.randint(cfg.jpeg_quality_min, cfg.jpeg_quality_max)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        if cfg.blur_prob and random.random() < cfg.blur_prob:
            sigma = random.uniform(0.1, cfg.blur_sigma_max)
            img = transforms.functional.gaussian_blur(img, kernel_size=5, sigma=sigma)
        return img, label

    def __getitem__(self, index: int):
        sample = self.samples[index]
        img = Image.open(sample.image_path).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        label = sample.label
        if self.augment.enabled:
            img, label = self._augment(img, label)
        tensor = self.to_tensor(img)
        if self.augment.enabled and self.augment.noise_prob:
            if random.random() < self.augment.noise_prob:
                tensor = torch.clamp(
                    tensor + torch.randn_like(tensor) * self.augment.noise_std, 0.0, 1.0
                )
        tensor = self.normalize(tensor)
        return tensor, label, sample.image_id, sample.patient_id


def samples_from_manifest(
    rows: list[dict[str, str]],
    split: str,
    cache_dir: Path,
) -> list[Sample]:
    """Build the sample list for one split from the manifest rows."""
    out: list[Sample] = []
    for row in rows:
        if row["split"] != split:
            continue
        if row["annotation_status"] not in ("annotated",):
            continue
        path = cache_dir / f"{row['image_id']}.png"
        out.append(
            Sample(
                image_path=path,
                label=label_to_index(row["view_label"]),
                patient_id=row["patient_id"],
                image_id=row["image_id"],
                view_label=row["view_label"],
            )
        )
    out.sort(key=lambda s: s.image_id)
    return out
