"""Image loading and resolution handling.

The source photographs are full-resolution clinical DSLR images (up to
6016x4016).  Two resolutions are used throughout, both configurable:

``inference_long_side`` (default 2048)
    resolution at which the segmentation models see the image.  Both stages of
    SegmentAnyTooth already work internally at fixed sizes (the YOLO11 detector
    letterboxes to 640, SAM-HQ embeds at 1024), so this is not a quality knob;
    it only removes the cost of shuffling 24-megapixel buffers.

``eval_long_side`` (default 1024)
    resolution at which every mask - reference and predicted - is compared.
    Fixing it makes the metric independent of the per-image sensor resolution,
    which varies across 2,283 distinct sizes in this dataset.

Both values are recorded in every result file.
"""

from __future__ import annotations

import cv2
import numpy as np

INFERENCE_LONG_SIDE = 2048
EVAL_LONG_SIDE = 1024


def target_size(width: int, height: int, long_side: int | None) -> tuple[int, int]:
    """Return ``(width, height)`` scaled so the long side equals ``long_side``.

    Never upscales: an image already smaller than ``long_side`` is left alone.
    """
    if not long_side:
        return width, height
    longest = max(width, height)
    if longest <= long_side:
        return width, height
    scale = long_side / longest
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def load_image_bgr(path, long_side: int | None = None) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read image: {path}")
    h, w = img.shape[:2]
    tw, th = target_size(w, h, long_side)
    if (tw, th) != (w, h):
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    return img


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a boolean mask, keeping pixels whose area coverage exceeds 0.5."""
    if mask.shape[1] == width and mask.shape[0] == height:
        return mask.astype(bool)
    resized = cv2.resize(
        mask.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA
    )
    return resized >= 0.5


def masks_to_raster(
    masks: list[np.ndarray], width: int, height: int, order_by_area: bool = True
) -> np.ndarray:
    """Pack boolean masks into a uint16 instance-ID raster at a given size.

    Larger instances are painted first so a small instance overlapping a large
    one survives, matching the ground-truth rasterisation policy.
    """
    raster = np.zeros((height, width), dtype=np.uint16)
    indexed = list(enumerate(masks, start=1))
    if order_by_area:
        indexed.sort(key=lambda t: -int(t[1].sum()))
    for inst_id, mask in indexed:
        small = resize_mask(mask, width, height)
        raster[small] = inst_id
    return raster


def raster_to_masks(raster: np.ndarray, instance_ids: list[int]) -> list[np.ndarray]:
    return [raster == int(i) for i in instance_ids]


def scale_boxes(boxes, scale_x: float, scale_y: float):
    out = []
    for x0, y0, x1, y1 in boxes:
        out.append((x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y))
    return out
