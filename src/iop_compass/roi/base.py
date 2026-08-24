"""Common interface for the four ROI strategies.

Every strategy returns the same :class:`RoiResult`, so the segmentation driver and
the ROI evaluation script are agnostic to how the region was obtained.

Strategies
    ``R0`` full image (no crop)
    ``R1`` view-specific geometric prior fitted on training annotations
    ``R2`` lightweight learned single-box detector
    ``R3`` SAM 3 concept-prompted intraoral region

Box convention: ``(x0, y0, x1, y1)`` with ``x1`` / ``y1`` **exclusive**, matching
the vendored SAM 3 ROI code (``_sam3_upstream._bbox_from_mask`` adds one) so a box
can be used directly as a numpy slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class RoiResult:
    """Region of interest for one image.

    ``roi_mask``
        binary region kept; post-processing clips predictions to it.
    ``allowed_mask``
        slightly dilated region used for the photometric guidance applied before
        inference; defaults to ``roi_mask``.
    ``exclusion_mask``
        region treated as definitely-not-tooth by post-processing.
    ``box``
        axis-aligned crop, ``x1``/``y1`` exclusive.
    ``fallback``
        the strategy failed and the full image was used instead.
    """

    roi_mask: np.ndarray
    exclusion_mask: np.ndarray
    box: tuple[int, int, int, int]
    strategy: str
    allowed_mask: np.ndarray | None = None
    fallback: bool = False
    fallback_reason: str = ""
    runtime_s: float = 0.0
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.allowed_mask is None:
            self.allowed_mask = self.roi_mask

    @property
    def area(self) -> int:
        return int((self.roi_mask > 0).sum())

    def crop(self, array: np.ndarray) -> np.ndarray:
        x0, y0, x1, y1 = self.box
        return array[y0:y1, x0:x1]


class RoiStrategy(Protocol):
    name: str

    def __call__(self, image_bgr: np.ndarray, view_label: str) -> RoiResult:
        ...


def full_image_result(
    shape: tuple[int, int], strategy: str, fallback: bool = False, reason: str = ""
) -> RoiResult:
    h, w = shape
    return RoiResult(
        roi_mask=np.ones((h, w), dtype=np.uint8),
        exclusion_mask=np.zeros((h, w), dtype=np.uint8),
        box=(0, 0, w, h),
        strategy=strategy,
        fallback=fallback,
        fallback_reason=reason,
    )


def clip_box(box, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Clip an exclusive-end box to the image, keeping it non-empty."""
    h, w = shape
    x0, y0, x1, y1 = box
    x0 = int(max(0, min(round(x0), w - 1)))
    y0 = int(max(0, min(round(y0), h - 1)))
    x1 = int(max(x0 + 1, min(round(x1), w)))
    y1 = int(max(y0 + 1, min(round(y1), h)))
    return x0, y0, x1, y1


def box_to_masks(
    box: tuple[int, int, int, int], shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    x0, y0, x1, y1 = clip_box(box, (h, w))
    roi = np.zeros((h, w), dtype=np.uint8)
    roi[y0:y1, x0:x1] = 1
    return roi, (1 - roi).astype(np.uint8)


class FullImageRoi:
    """R0 - identity strategy, used so R0 goes through the same code path."""

    name = "R0_full"

    def __call__(self, image_bgr: np.ndarray, view_label: str) -> RoiResult:
        return full_image_result(image_bgr.shape[:2], self.name)
