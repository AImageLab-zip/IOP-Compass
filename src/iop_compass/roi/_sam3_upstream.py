"""SAM 3 concept-prompted intraoral ROI extraction.

Vendored from the upstream ``sam3_roi_preprocess`` implementation so the repository
owns exactly one copy.  Only two things changed, both marked ``PATCHED``:

* ``build_sam3`` accepts an explicit local ``checkpoint_path`` and forwards the
  device to ``Sam3Processor`` (upstream dropped it, so the CPU path was broken
  and the checkpoint was always fetched from Hugging Face at runtime);
* the noisy ``print`` became a no-op unless ``verbose=True``.

Everything else - the prompt list, the candidate ranking, every threshold and the
glove/retractor heuristic - is byte-for-byte the original.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover - the vendored package is import-time optional
    from sam3.model.sam3_image_processor import Sam3Processor

# The vendored ``sam3`` package is linked in by ``scripts/fetch_third_party.py`` and is
# needed only by the R3 ROI strategy, which the released viewer never runs.  Importing
# it here would make the whole ``iop_compass.roi`` package - and the test suite that
# stubs this module - unimportable without it, so it is imported inside
# :func:`build_sam3`, the only place it is used.  ``sam3_roi`` already defers its own
# import of this module for the same reason.


@dataclass
class Sam3RoiParams:
    long_side: int = 1400

    roi_close_k: int = 7
    roi_open_k: int = 2
    min_roi_area: int = 16000
    min_exclusion_area: int = 1400
    fragment_min_area: int = 700

    crop_pad: int = 110
    allowed_expand_px: int = 14
    exclusion_expand_px: int = 4
    subtract_exclusion_from_roi: bool = False
    fill_holes_flag: bool = True

    score_thr_mouth: float = 0.16

    mouth_prompts: List[str] = field(default_factory=lambda: [
        "mouth",
        "oral cavity",
        "inside the mouth",
        "intraoral region",
        "mouth opening",
        "teeth and gums",
        "visible mouth interior",
    ])


def build_sam3(
    device: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    verbose: bool = False,
) -> Sam3Processor:
    """PATCHED: local checkpoint, device forwarded to the processor.

    Passing ``checkpoint_path`` keeps the job fully offline; without it the
    upstream builder downloads ``facebook/sam3`` from Hugging Face at runtime,
    which is neither reproducible nor available on a compute node.
    """
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_sam3_image_model(
        device=dev,
        checkpoint_path=checkpoint_path,
        load_from_HF=checkpoint_path is None,
    )
    processor = Sam3Processor(model, device=dev)
    if verbose:
        print("[OK] SAM3 loaded on", dev)
    return processor


def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_bool_mask(m: np.ndarray, hw=None) -> Optional[np.ndarray]:
    if m is None:
        return None
    m = np.squeeze(np.asarray(m))
    if m.ndim != 2 or m.size == 0:
        return None
    m = m.astype(bool)
    if hw is not None and m.shape != hw:
        H, W = hw
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    return m


def _resize_long_side(img: np.ndarray, long_side: int):
    h, w = img.shape[:2]
    s = long_side / float(max(h, w))
    if s >= 1.0:
        return img, 1.0
    out = cv2.resize(
        img,
        (int(round(w * s)), int(round(h * s))),
        interpolation=cv2.INTER_AREA,
    )
    return out, s


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8)


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    h, w = mask_u8.shape
    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    out = cv2.bitwise_or(mask_u8, holes)
    return (out > 0).astype(np.uint8)


def _smooth_mask(mask: np.ndarray, close_k: int, open_k: int, fill_holes_flag: bool) -> np.ndarray:
    out = (mask > 0).astype(np.uint8)

    if close_k > 1:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ker)

    if open_k > 1:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, ker)

    if fill_holes_flag:
        out = _fill_holes(out)

    return (out > 0).astype(np.uint8)


def _bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _expand_box(box, h, w, pad):
    x0, y0, x1, y1 = box
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    return x0, y0, x1, y1


def _touches_top(mask: np.ndarray, frac: float = 0.04) -> bool:
    h, _ = mask.shape
    hh = max(1, int(h * frac))
    return bool(mask[:hh, :].any())


def _remove_top_bright_glove_regions(img_rgb: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    h, w = roi_mask.shape
    glove_like = ((v > 150) & (s < 55)).astype(np.uint8)

    top_zone = np.zeros((h, w), np.uint8)
    top_zone[:max(20, int(0.22 * h)), :] = 1
    glove_like = ((glove_like > 0) & (top_zone > 0)).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(glove_like, 8)
    cleaned = roi_mask.copy().astype(np.uint8)

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 120:
            continue

        y = stats[i, cv2.CC_STAT_TOP]
        comp = (labels == i)
        overlap = int((comp & (roi_mask > 0)).sum())
        if overlap == 0:
            continue

        if y < int(0.18 * h):
            cleaned[comp] = 0

    return (cleaned > 0).astype(np.uint8)


def _mask_score_for_mouth(mask: np.ndarray, img_rgb: np.ndarray) -> float:
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return -1e9

    h, w = mask.shape
    area = float(mask.sum())
    area_frac = area / float(h * w)

    ys, xs = np.where(mask > 0)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    box_area_frac = (bw * bh) / float(h * w)

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    inside_sat = float(sat[mask > 0].mean())
    inside_val = float(val[mask > 0].mean())

    score = 0.0
    score += 5.0 * min(area_frac, 0.70)
    score += 2.0 * min(box_area_frac, 0.90)
    score += 0.0025 * inside_sat
    score += 0.0010 * inside_val

    if _touches_top(mask, frac=0.04):
        score -= 0.25

    return float(score)


def _dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    if px <= 0:
        return mask
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    return cv2.dilate(mask, ker, iterations=1)


def _merge_close_masks(masks: List[np.ndarray], dilate_px: int = 20) -> np.ndarray:
    if not masks:
        raise RuntimeError("No masks to merge.")

    masks_u8 = [(m > 0).astype(np.uint8) for m in masks]
    grown = [_dilate_mask(m, dilate_px) for m in masks_u8]

    used = [False] * len(masks_u8)
    merged_groups = []

    for i in range(len(masks_u8)):
        if used[i]:
            continue

        group = masks_u8[i].copy()
        group_grown = grown[i].copy()
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j in range(len(masks_u8)):
                if used[j]:
                    continue
                if np.any((group_grown > 0) & (grown[j] > 0)):
                    group = ((group > 0) | (masks_u8[j] > 0)).astype(np.uint8)
                    group_grown = ((group_grown > 0) | (grown[j] > 0)).astype(np.uint8)
                    used[j] = True
                    changed = True

        merged_groups.append(group)

    areas = [int(g.sum()) for g in merged_groups]
    best = merged_groups[int(np.argmax(areas))]
    return (best > 0).astype(np.uint8)


def sam3_text_masks(
    processor: Sam3Processor,
    pil_img: Image.Image,
    prompts: List[str],
    score_thr: float,
) -> List[np.ndarray]:
    state = processor.set_image(pil_img)
    out_masks: List[np.ndarray] = []

    for prompt in prompts:
        out = processor.set_text_prompt(state=state, prompt=prompt)
        masks = _to_numpy(out.get("masks"))
        scores = _to_numpy(out.get("scores"))

        if masks is None or len(masks) == 0:
            continue
        if scores is None:
            scores = np.ones((masks.shape[0],), dtype=np.float32)

        for m, s in zip(masks, scores):
            if float(s) < float(score_thr):
                continue
            m = np.squeeze(np.asarray(m))
            if m.ndim != 2 or m.size == 0:
                continue
            out_masks.append((m > 0.5).astype(np.uint8))

    return out_masks


def run_real_sam3_masks(
    processor: Sam3Processor,
    img_rgb: np.ndarray,
    params: Sam3RoiParams,
) -> List[np.ndarray]:
    img_small, scale = _resize_long_side(img_rgb, params.long_side)
    pil_small = Image.fromarray(img_small)

    small_masks = sam3_text_masks(
        processor=processor,
        pil_img=pil_small,
        prompts=params.mouth_prompts,
        score_thr=params.score_thr_mouth,
    )

    H, W = img_rgb.shape[:2]
    out_masks: List[np.ndarray] = []

    for m in small_masks:
        mb = _safe_bool_mask(m)
        if mb is None:
            continue

        if scale != 1.0:
            mb = cv2.resize(
                mb.astype(np.uint8),
                (W, H),
                interpolation=cv2.INTER_NEAREST,
            ) > 0

        mu8 = mb.astype(np.uint8)
        mu8 = _remove_small_components(mu8, params.fragment_min_area)
        if mu8.sum() == 0:
            continue

        out_masks.append(mu8)

    return out_masks


def make_sam3_roi_and_exclusion(
    processor,
    img_rgb: np.ndarray,
    params: Sam3RoiParams,
    view: str | None = None,
):
    h, w = img_rgb.shape[:2]

    sam_masks = run_real_sam3_masks(processor, img_rgb, params)
    if not sam_masks:
        raise RuntimeError("SAM3 returned no mouth candidates.")

    scored = []
    for m in sam_masks:
        mu8 = (m > 0).astype(np.uint8)
        if mu8.sum() < params.fragment_min_area:
            continue
        score = _mask_score_for_mouth(mu8, img_rgb)
        scored.append((score, mu8))

    if not scored:
        raise RuntimeError("No usable SAM3 mouth candidate remained after filtering.")

    scored.sort(key=lambda x: x[0], reverse=True)

    best_score = scored[0][0]
    keep_masks: List[np.ndarray] = []
    for score, mask in scored:
        if score >= best_score - 0.75:
            keep_masks.append(mask)

    if not keep_masks:
        keep_masks = [scored[0][1]]

    sam_raw_mask = _merge_close_masks(keep_masks, dilate_px=20)
    sam_raw_mask = _remove_small_components(sam_raw_mask, params.fragment_min_area)

    if sam_raw_mask.sum() < params.min_roi_area:
        raise RuntimeError("Selected SAM3 mouth mask is too small.")

    roi_mask = sam_raw_mask.copy()
    roi_mask = _smooth_mask(
        roi_mask,
        params.roi_close_k,
        params.roi_open_k,
        params.fill_holes_flag,
    )

    roi_mask = _remove_top_bright_glove_regions(img_rgb, roi_mask)
    roi_mask = _remove_small_components(roi_mask, params.fragment_min_area)
    roi_mask = _smooth_mask(
        roi_mask,
        params.roi_close_k,
        params.roi_open_k,
        params.fill_holes_flag,
    )

    box = _bbox_from_mask(roi_mask)
    if box is None:
        raise RuntimeError("SAM3 ROI box is empty after cleanup.")

    roi_box = _expand_box(box, h, w, params.crop_pad)

    allowed_mask = roi_mask.copy()
    if params.allowed_expand_px > 0:
        ker = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (params.allowed_expand_px * 2 + 1, params.allowed_expand_px * 2 + 1),
        )
        allowed_mask = cv2.dilate(allowed_mask, ker, iterations=1)
        allowed_mask = (allowed_mask > 0).astype(np.uint8)

    exclusion_mask = (1 - roi_mask).astype(np.uint8)
    if params.exclusion_expand_px > 0:
        ker = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (params.exclusion_expand_px * 2 + 1, params.exclusion_expand_px * 2 + 1),
        )
        exclusion_mask = cv2.erode(exclusion_mask, ker, iterations=1)
        exclusion_mask = (exclusion_mask > 0).astype(np.uint8)

    exclusion_mask = _remove_small_components(exclusion_mask, params.min_exclusion_area)
    exclusion_soft_mask = exclusion_mask.copy()

    if params.subtract_exclusion_from_roi:
        roi_mask = ((roi_mask > 0) & (exclusion_soft_mask == 0)).astype(np.uint8)
        allowed_mask = ((allowed_mask > 0) & (exclusion_soft_mask == 0)).astype(np.uint8)

    return {
        "sam_raw_mask": sam_raw_mask.astype(np.uint8),
        "roi_mask": roi_mask.astype(np.uint8),
        "allowed_mask": allowed_mask.astype(np.uint8),
        "exclusion_mask": exclusion_mask.astype(np.uint8),
        "exclusion_soft_mask": exclusion_soft_mask.astype(np.uint8),
        "roi_box": roi_box,
    }


def crop_to_roi_region(img_rgb: np.ndarray, roi_box):
    x0, y0, x1, y1 = roi_box
    crop = img_rgb[y0:y1, x0:x1].copy()
    return crop, (x0, y0, x1, y1)


def paste_mask_back(crop_mask: np.ndarray, full_hw, crop_box):
    h, w = full_hw
    x0, y0, x1, y1 = crop_box
    out = np.zeros((h, w), dtype=crop_mask.dtype)
    out[y0:y1, x0:x1] = crop_mask[: y1 - y0, : x1 - x0]
    return out


def apply_roi_guidance(
    crop_rgb: np.ndarray,
    allowed_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    mode: str = "soft_keep_texture",
    view: str | None = None,
):
    out = crop_rgb.copy()
    keep = (allowed_mask > 0) & (exclusion_mask == 0)

    if mode != "soft_keep_texture":
        out[~keep] = 0
        return out

    ys, xs = np.where(allowed_mask > 0)
    if len(ys) == 0:
        return out

    h, w = allowed_mask.shape
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    roi_h = max(1, y1 - y0)

    if view == "front":
        band_top = y0 + int(0.04 * roi_h)
        band_bot = y0 + int(0.97 * roi_h)
    else:
        band_top = y0 + int(0.05 * roi_h)
        band_bot = y0 + int(0.97 * roi_h)

    yy = np.arange(h)[:, None]
    band = (yy >= band_top) & (yy <= band_bot)

    xx = np.arange(w)[None, :]
    edge_margin = max(28, int(0.20 * w))
    side_keep = (xx < edge_margin) | (xx >= (w - edge_margin))

    strong_keep = keep & (band | side_keep)
    weak_keep = keep & (~(band | side_keep))

    dim_outer = (crop_rgb * 0.25).astype(np.uint8)
    dim_weak = (crop_rgb * 0.92).astype(np.uint8)

    out[~keep] = dim_outer[~keep]
    out[weak_keep] = dim_weak[weak_keep]
    out[strong_keep] = crop_rgb[strong_keep]

    return out