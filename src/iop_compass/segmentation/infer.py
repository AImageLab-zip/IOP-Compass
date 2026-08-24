"""Segmentation inference over the benchmark grid, and the common prediction format.

:func:`infer_cells` produces the cells of the ROI x segmenter x post-processing grid
defined in :mod:`.grid` for one image.  A cell's post-processed half is a pure
function of its raw masks, so each ROI level costs one forward pass regardless of
how many of its cells were asked for.

Three design points, all recorded in every prediction file:

* the ROI is applied to **all five views**, including the two occlusal ones;
* every ROI level reaches every segmenter through the single :func:`run_with_roi`
  implementation, so the ablation isolates the ROI source instead of confounding it
  with a per-backend crop convention;
* the photometric ROI guidance stays on the SegmentAnyTooth path only, because it is
  a SegmentAnyTooth preprocessing trick rather than a property of the region; each
  prediction records ``roi_guidance`` so this is never inferred from the cell name.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..roi.base import RoiResult, full_image_result
from . import grid
from .postprocessing import PostProcessParams, postprocess_instances
from .segmentanytooth_adapter import SatPrediction, SegmentAnyToothRunner


@dataclass
class Prediction:
    """Canonical prediction record for one image and one variant."""

    image_id: str
    patient_id: str
    view_label: str
    variant: str
    width: int
    height: int
    instances: list[dict] = field(default_factory=list)
    roi: dict = field(default_factory=dict)
    postprocess: dict = field(default_factory=dict)
    runtime: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def save_prediction(
    prediction: Prediction, masks: list[np.ndarray], out_dir: Path
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raster = np.zeros((prediction.height, prediction.width), dtype=np.uint16)
    order = sorted(range(len(masks)), key=lambda i: -int(masks[i].sum()))
    for rank in order:
        raster[masks[rank]] = rank + 1
    for i, row in enumerate(prediction.instances):
        row["instance_id"] = i + 1
        row["raster_area"] = int((raster == i + 1).sum())
    png = out_dir / f"{prediction.image_id}_pred.png"
    js = out_dir / f"{prediction.image_id}_pred.json"
    if not cv2.imwrite(str(png), raster):
        raise IOError(f"failed to write {png}")
    js.write_text(json.dumps(prediction.to_dict(), indent=1))
    return png, js


def save_roi_masks(
    image_id: str,
    roi: RoiResult,
    eval_size: tuple[int, int],
    out_dir: Path,
) -> Path:
    """Store a raw cell's ROI and exclusion masks at the evaluation resolution.

    Post-processing clips against the ROI *mask*, which for the concept-prompted
    strategy is not the same thing as its bounding box.  Without this sidecar a
    post-processed cell could only ever be re-derived against the box, which silently
    changes the result for that strategy; with it, every ``post`` cell is an exact
    function of its ``no-post`` cell.

    Bit 0 is the ROI, bit 1 the exclusion region, so the two need one file and are
    allowed to overlap.
    """
    from ..data.imaging import resize_mask

    width, height = eval_size
    roi_small = resize_mask(roi.roi_mask > 0, width, height)
    exclusion_small = resize_mask(roi.exclusion_mask > 0, width, height)
    packed = roi_small.astype(np.uint8) | (exclusion_small.astype(np.uint8) << 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{image_id}_roi.png"
    if not cv2.imwrite(str(path), packed):
        raise IOError(f"failed to write {path}")
    return path


def load_roi_masks(json_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Read the ROI/exclusion sidecar of a prediction, or ``None`` if absent."""
    path = Path(str(json_path).replace("_pred.json", "_roi.png"))
    if not path.exists():
        return None
    packed = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if packed is None:
        return None
    return (packed & 1).astype(np.uint8), ((packed >> 1) & 1).astype(np.uint8)


def load_prediction(json_path: Path) -> tuple[dict, np.ndarray]:
    payload = json.loads(Path(json_path).read_text())
    png = Path(str(json_path).replace("_pred.json", "_pred.png"))
    raster = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if raster is None:
        raise IOError(f"cannot read {png}")
    return payload, raster.astype(np.uint16)


def _instances_payload(
    fdis: list[int | None], scores: list[float], masks: list[np.ndarray]
) -> list[dict]:
    rows = []
    for i, (fdi, score, mask) in enumerate(zip(fdis, scores, masks), start=1):
        ys, xs = np.nonzero(mask)
        bbox = (
            [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            if xs.size
            else [0, 0, 0, 0]
        )
        rows.append(
            {
                "instance_id": i,
                "fdi": fdi,
                "score": float(score),
                "bbox": bbox,
                "area": int(mask.sum()),
            }
        )
    return rows


def run_with_roi(
    predict_fn,
    image_bgr: np.ndarray,
    view_label: str,
    roi: RoiResult,
    guidance: bool = True,
    guidance_mode: str = "soft_keep_texture",
) -> tuple[list[np.ndarray], list[int | None], list[float], int, float]:
    """Crop to the ROI box, optionally guide the crop, segment it, paste back.

    Backend-agnostic on purpose: ``predict_fn(image_bgr, view_label)`` returns
    ``(masks, fdis, scores, n_detections)`` and is the *only* thing that differs
    between SegmentAnyTooth and Mask R-CNN.  Every ROI level therefore reaches every
    segmenter through one implementation, so the ROI axis cannot be confounded with
    a per-backend crop convention.

    Order and arguments follow the upstream pipeline (``Model_run.py:311-336``):
    crop the image *and* the masks to the box, guide the crop with the dilated
    allowed mask and the soft exclusion mask, then paste the predicted masks back
    into full-image coordinates.

    ``guidance`` is the one deliberate asymmetry: the photometric guidance is a
    SegmentAnyTooth preprocessing trick, not a property of the region, so it is off
    for Mask R-CNN.  Which way it ran is recorded in every prediction's ``meta``.
    """
    from ..roi._sam3_upstream import (
        apply_roi_guidance,
        crop_to_roi_region,
        paste_mask_back,
    )

    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb, crop_box = crop_to_roi_region(rgb, roi.box)
    if guidance:
        crop_rgb = apply_roi_guidance(
            crop_rgb,
            allowed_mask=roi.crop(roi.allowed_mask),
            exclusion_mask=roi.crop(roi.exclusion_mask),
            mode=guidance_mode,
            view=view_label,
        )
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

    start = time.perf_counter()
    masks, fdis, scores, n_detections = predict_fn(crop_bgr, view_label)
    elapsed = time.perf_counter() - start

    out_masks, out_fdis, out_scores = [], [], []
    for mask, fdi, score in zip(masks, fdis, scores):
        pasted = paste_mask_back(mask.astype(np.uint8), (h, w), crop_box) > 0
        if not pasted.any():
            continue
        out_masks.append(pasted)
        out_fdis.append(fdi)
        out_scores.append(score)
    return out_masks, out_fdis, out_scores, n_detections, elapsed


def sat_predict_fn(runner: SegmentAnyToothRunner):
    """Adapt SegmentAnyTooth to the ``predict_fn`` contract of :func:`run_with_roi`."""

    def predict(image_bgr: np.ndarray, view_label: str):
        pred = runner.predict(image_bgr, view_label)
        return pred.masks, pred.fdis, pred.scores, pred.n_detections

    return predict


def maskrcnn_predict_fn(predictor):
    """Adapt a :class:`MaskRcnnPredictor` to the ``predict_fn`` contract.

    Mask R-CNN has no separate detection stage, so ``n_detections`` is the number of
    instances that survived its score threshold.
    """

    def predict(image_bgr: np.ndarray, view_label: str):
        masks, fdis, scores = predictor.predict(image_bgr)
        return masks, fdis, scores, len(masks)

    return predict


def run_sat_with_roi(
    runner: SegmentAnyToothRunner,
    image_bgr: np.ndarray,
    view_label: str,
    roi: RoiResult,
    guidance_mode: str = "soft_keep_texture",
) -> tuple[SatPrediction, float]:
    """Backwards-compatible SAT wrapper around :func:`run_with_roi`."""
    from .segmentanytooth_adapter import SatInstance

    masks, fdis, scores, n_detections, elapsed = run_with_roi(
        sat_predict_fn(runner),
        image_bgr,
        view_label,
        roi,
        guidance=True,
        guidance_mode=guidance_mode,
    )
    full = SatPrediction(instances=[], fdi_raster=None, n_detections=n_detections)
    for mask, fdi, score in zip(masks, fdis, scores):
        ys, xs = np.nonzero(mask)
        bbox = (
            (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            if xs.size
            else (0.0, 0.0, 0.0, 0.0)
        )
        full.instances.append(SatInstance(mask=mask, fdi=fdi, score=score, bbox=bbox))
    return full, elapsed


def build_prediction(
    image_id: str,
    patient_id: str,
    view_label: str,
    variant: str,
    masks: list[np.ndarray],
    fdis: list[int | None],
    scores: list[float],
    eval_size: tuple[int, int],
    roi: RoiResult | None,
    postprocess_stats: dict | None,
    runtime: dict,
    meta: dict,
) -> tuple[Prediction, list[np.ndarray]]:
    """Downscale masks to the evaluation resolution and package the record."""
    width, height = eval_size
    small = [_resize_bool(mask, width, height) for mask in masks]
    keep = [i for i, m in enumerate(small) if m.any()]
    small = [small[i] for i in keep]
    fdis = [fdis[i] for i in keep]
    scores = [scores[i] for i in keep]

    prediction = Prediction(
        image_id=image_id,
        patient_id=patient_id,
        view_label=view_label,
        variant=variant,
        width=width,
        height=height,
        instances=_instances_payload(fdis, scores, small),
        roi=(
            {
                "strategy": roi.strategy,
                "box": list(roi.box),
                # the same box expressed at the evaluation resolution, so
                # downstream tools never have to reconstruct the scaling
                "box_eval": list(_scale_box(roi, width, height)),
                "fallback": bool(roi.fallback),
                "fallback_reason": roi.fallback_reason,
                "roi_area": int(roi.area),
                "runtime_s": roi.runtime_s,
            }
            if roi is not None
            else {
                "strategy": "R0_full",
                "box_eval": [0, 0, width, height],
                "fallback": False,
            }
        ),
        postprocess=postprocess_stats or {},
        runtime=runtime,
        meta=meta,
    )
    return prediction, small


def _resize_bool(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    from ..data.imaging import resize_mask

    return resize_mask(mask, width, height)


def _scale_box(roi: RoiResult, width: int, height: int) -> tuple[int, int, int, int]:
    """Express an ROI box at the evaluation resolution (exclusive ends)."""
    src_h, src_w = roi.roi_mask.shape[:2]
    sx, sy = width / max(src_w, 1), height / max(src_h, 1)
    x0, y0, x1, y1 = roi.box
    nx0 = max(0, min(int(round(x0 * sx)), width - 1))
    ny0 = max(0, min(int(round(y0 * sy)), height - 1))
    nx1 = max(nx0 + 1, min(int(round(x1 * sx)), width))
    ny1 = max(ny0 + 1, min(int(round(y1 * sy)), height))
    return nx0, ny0, nx1, ny1


def infer_cells(
    predict_fn,
    seg_level: str,
    image_bgr: np.ndarray,
    image_id: str,
    patient_id: str,
    view_label: str,
    cells: list[str],
    roi_strategies: dict[str, object],
    post_params: PostProcessParams,
    eval_size: tuple[int, int],
    peak_gpu_mb: float = 0.0,
    extra_meta: dict | None = None,
) -> tuple[dict[str, tuple[Prediction, list[np.ndarray]]], dict[str, RoiResult]]:
    """Produce every requested grid cell for one image, with one pass per ROI level.

    Returns ``(results, rois)``: the cell predictions, and the :class:`RoiResult` of
    each ``no-post`` cell so the caller can write the ROI sidecar that makes the
    matching ``post`` cell exactly re-derivable.

    ``cells`` are grid cell ids (see :mod:`.grid`) that all share ``seg_level``.  A
    cell's ``post`` half is a pure function of its ``no-post`` masks, so a ROI level
    whose two cells are both requested still costs a single forward pass.

    ``eval_size`` is ``(width, height)`` at the evaluation resolution and must be
    derived from the *native* image size, exactly as the reference rasters were.
    Recomputing it from the already-downscaled inference image rounds twice and can
    be one pixel off, which the evaluator would report as a shape mismatch and skip.
    """
    h, w = image_bgr.shape[:2]
    eval_w, eval_h = eval_size
    out: dict[str, tuple[Prediction, list[np.ndarray]]] = {}
    rois: dict[str, RoiResult] = {}
    guidance = grid.ROI_GUIDANCE[seg_level]

    wanted: dict[str, set[str]] = {}
    for cell in cells:
        roi_level, cell_seg, post = grid.parse_cell(cell)
        if cell_seg != seg_level:
            raise grid.GridError(
                f"cell {cell!r} is not a {seg_level!r} cell; group cells by segmenter"
            )
        wanted.setdefault(roi_level, set()).add(post)

    for roi_level in grid.ROI_LEVELS:
        posts = wanted.get(roi_level)
        if not posts:
            continue

        if roi_level == "no-roi":
            # No crop, no guidance: the segmenter sees the frame as it is.  This is
            # also what keeps the no-ROI cells bit-identical to the full-image
            # predictions already on disk.
            roi = full_image_result((h, w), grid.ROI_STRATEGY["no-roi"])
            start = time.perf_counter()
            masks, fdis, scores, n_detections = predict_fn(image_bgr, view_label)
            model_seconds = time.perf_counter() - start
            roi_seconds = 0.0
        else:
            strategy = roi_strategies.get(grid.ROI_STRATEGY[roi_level])
            if strategy is None:
                continue
            roi = strategy(image_bgr, view_label)
            masks, fdis, scores, n_detections, model_seconds = run_with_roi(
                predict_fn, image_bgr, view_label, roi, guidance=guidance
            )
            roi_seconds = roi.runtime_s

        base_runtime = {
            "roi_seconds": roi_seconds,
            "model_seconds": model_seconds,
            "total_seconds": roi_seconds + model_seconds,
        }
        base_meta = {
            "n_detections": n_detections,
            "peak_gpu_mb": peak_gpu_mb,
            "segmenter": seg_level,
            "roi_level": roi_level,
            "roi_guidance": guidance,
            **(extra_meta or {}),
        }

        if "no-post" in posts:
            cell = grid.cell_id(roi_level, seg_level, "no-post")
            # Handed back so the caller can store the ROI sidecar next to the raw
            # masks; that is what lets repostprocess.py reproduce the post cell
            # exactly for mask-based strategies as well as box-based ones.
            rois[cell] = roi
            out[cell] = build_prediction(
                image_id,
                patient_id,
                view_label,
                cell,
                masks,
                fdis,
                scores,
                (eval_w, eval_h),
                roi,
                None,
                dict(base_runtime),
                dict(base_meta),
            )

        if "post" in posts:
            cell = grid.cell_id(roi_level, seg_level, "post")
            pp_start = time.perf_counter()
            # The no-ROI level has no region to clip against, so the rules run
            # without one exactly as they did for the full-image variant.
            roi_mask = None if roi_level == "no-roi" else roi.roi_mask
            exclusion = None if roi_level == "no-roi" else roi.exclusion_mask
            post_masks, post_fdis, post_scores, stats = postprocess_instances(
                masks,
                fdis,
                scores,
                roi_mask,
                exclusion,
                post_params,
                view_label,
            )
            pp_seconds = time.perf_counter() - pp_start
            runtime = dict(base_runtime)
            runtime["post_seconds"] = pp_seconds
            runtime["total_seconds"] = base_runtime["total_seconds"] + pp_seconds
            out[cell] = build_prediction(
                image_id,
                patient_id,
                view_label,
                cell,
                post_masks,
                post_fdis,
                post_scores,
                (eval_w, eval_h),
                roi,
                stats.to_dict(),
                runtime,
                dict(base_meta),
            )

    return out, rois
