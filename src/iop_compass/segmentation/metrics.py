"""Shared segmentation evaluator used by every method.

Per image the evaluator reduces a prediction to a small set of additive
sufficient statistics (:class:`ImageEval`).  Every reported metric is then a pure
function of the sum of those statistics, which makes patient-level bootstrap
resampling exact and cheap: resample patients, sum, recompute.

Primary outcome
    FDI-aware instance F1 at mask IoU >= 0.50.  A true positive requires a
    one-to-one match, IoU >= 0.50 and the correct FDI code.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclasses_fields

import numpy as np

from .matching import MatchResult, match_instances

IOU_THRESHOLD = 0.50


@dataclass
class ImageEval:
    """Additive per-image statistics."""

    image_id: str
    patient_id: str
    view_label: str

    n_gt: int = 0
    n_pred: int = 0
    tp_inst: int = 0            # geometric matches at IoU >= threshold
    tp_fdi: int = 0             # matches that also carry the correct FDI
    fdi_correct: int = 0        # == tp_fdi, kept explicit for readability
    fdi_wrong: int = 0
    dice_sum: float = 0.0       # sum of Dice over matched pairs
    iou_sum: float = 0.0
    merges: int = 0
    splits: int = 0

    # mask-level (foreground) statistics
    gt_area: int = 0
    pred_area: int = 0
    inter_area: int = 0

    # boundary F-score accumulators (see boundary_f_score)
    boundary_tp_pred: int = 0
    boundary_n_pred: int = 0
    boundary_tp_gt: int = 0
    boundary_n_gt: int = 0

    complete_image: int = 0
    n_images: int = 1

    # AP support: per-match scores; empty when the method has no scores
    scores_tp: list[float] = field(default_factory=list)
    scores_fp: list[float] = field(default_factory=list)
    has_scores: bool = False

    runtime_s: float = 0.0
    peak_gpu_mb: float = 0.0

    def to_row(self) -> dict:
        row = asdict(self)
        row.pop("scores_tp")
        row.pop("scores_fp")
        return row

    @classmethod
    def from_row(cls, row: dict) -> "ImageEval":
        """Rebuild from a ``to_row`` record, e.g. a row of the per-image CSV.

        The per-image CSV holds every additive statistic, so any pooled metric or
        bootstrap can be recomputed from it without touching a mask again.  The
        per-match score lists are not stored and come back empty, which only affects
        average precision.
        """
        fields = {f.name: f for f in dataclasses_fields(cls)}
        kwargs: dict = {}
        for name, field_def in fields.items():
            if name in ("scores_tp", "scores_fp"):
                continue
            if name not in row or row[name] == "":
                continue
            value = row[name]
            if field_def.type in ("int", int):
                kwargs[name] = int(float(value))
            elif field_def.type in ("float", float):
                kwargs[name] = float(value)
            elif field_def.type in ("bool", bool):
                kwargs[name] = str(value).strip().lower() in ("1", "true", "yes")
            else:
                kwargs[name] = value
        return cls(**kwargs)


ADDITIVE_FIELDS = (
    "n_gt",
    "n_pred",
    "tp_inst",
    "tp_fdi",
    "fdi_correct",
    "fdi_wrong",
    "dice_sum",
    "iou_sum",
    "merges",
    "splits",
    "gt_area",
    "pred_area",
    "inter_area",
    "boundary_tp_pred",
    "boundary_n_pred",
    "boundary_tp_gt",
    "boundary_n_gt",
    "complete_image",
    "n_images",
    "runtime_s",
    "peak_gpu_mb",
)


def _f1(tp: float, fp: float, fn: float) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def _safe_div(num: float, den: float) -> float:
    return num / den if den else float("nan")


def boundary_f_score(
    gt_fg: np.ndarray, pred_fg: np.ndarray, tolerance_px: int
) -> tuple[int, int, int, int]:
    """Boundary agreement accumulators.

    Returns ``(tp_pred, n_pred, tp_gt, n_gt)`` boundary-pixel counts: how many
    predicted boundary pixels lie within ``tolerance_px`` of a reference boundary
    and vice versa.  The caller turns these into precision / recall / F.
    """
    import cv2

    def edges(mask: np.ndarray) -> np.ndarray:
        m = mask.astype(np.uint8)
        eroded = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
        return (m - eroded).astype(bool)

    gt_edge = edges(gt_fg)
    pred_edge = edges(pred_fg)
    if not gt_edge.any() and not pred_edge.any():
        return 0, 0, 0, 0

    k = max(1, int(tolerance_px))
    kernel = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    gt_dil = cv2.dilate(gt_edge.astype(np.uint8), kernel, iterations=1).astype(bool)
    pred_dil = cv2.dilate(pred_edge.astype(np.uint8), kernel, iterations=1).astype(bool)

    tp_pred = int((pred_edge & gt_dil).sum())
    tp_gt = int((gt_edge & pred_dil).sum())
    return tp_pred, int(pred_edge.sum()), tp_gt, int(gt_edge.sum())


def evaluate_image(
    image_id: str,
    patient_id: str,
    view_label: str,
    gt_masks: list[np.ndarray],
    gt_fdis: list[int | None],
    pred_masks: list[np.ndarray],
    pred_fdis: list[int | None],
    pred_scores: list[float] | None = None,
    iou_threshold: float = IOU_THRESHOLD,
    boundary_tolerance_frac: float = 0.002,
    compute_boundary: bool = True,
    runtime_s: float = 0.0,
    peak_gpu_mb: float = 0.0,
) -> tuple[ImageEval, MatchResult]:
    result = match_instances(
        gt_masks, gt_fdis, pred_masks, pred_fdis, iou_threshold=iou_threshold
    )

    ev = ImageEval(
        image_id=image_id,
        patient_id=patient_id,
        view_label=view_label,
        n_gt=result.n_gt,
        n_pred=result.n_pred,
        tp_inst=result.tp,
        merges=result.merges,
        splits=result.splits,
        runtime_s=runtime_s,
        peak_gpu_mb=peak_gpu_mb,
    )
    for m in result.matches:
        ev.dice_sum += m.dice
        ev.iou_sum += m.iou
        if m.fdi_correct:
            ev.tp_fdi += 1
    ev.fdi_correct = ev.tp_fdi
    ev.fdi_wrong = result.tp - ev.tp_fdi

    shape = None
    if gt_masks:
        shape = gt_masks[0].shape
    elif pred_masks:
        shape = pred_masks[0].shape

    if shape is not None:
        gt_fg = np.zeros(shape, dtype=bool)
        for m in gt_masks:
            gt_fg |= m
        pred_fg = np.zeros(shape, dtype=bool)
        for m in pred_masks:
            pred_fg |= m
        ev.gt_area = int(gt_fg.sum())
        ev.pred_area = int(pred_fg.sum())
        ev.inter_area = int((gt_fg & pred_fg).sum())
        if compute_boundary:
            diag = math.hypot(*shape)
            tol = max(1, int(round(boundary_tolerance_frac * diag)))
            bp, np_, bg, ng = boundary_f_score(gt_fg, pred_fg, tol)
            ev.boundary_tp_pred, ev.boundary_n_pred = bp, np_
            ev.boundary_tp_gt, ev.boundary_n_gt = bg, ng

    ev.complete_image = int(
        result.fn == 0
        and result.fp == 0
        and result.tp == result.n_gt
        and ev.fdi_wrong == 0
        and result.n_gt > 0
    )

    if pred_scores is not None and len(pred_scores) == result.n_pred:
        ev.has_scores = True
        matched_pred = {m.pred_index for m in result.matches if m.fdi_correct}
        for i, score in enumerate(pred_scores):
            (ev.scores_tp if i in matched_pred else ev.scores_fp).append(float(score))

    return ev, result


def aggregate(evals: list[ImageEval]) -> dict[str, float]:
    """Pool per-image statistics into the reported metric set."""
    totals = {name: 0.0 for name in ADDITIVE_FIELDS}
    for ev in evals:
        for name in ADDITIVE_FIELDS:
            totals[name] += getattr(ev, name)

    n_gt = totals["n_gt"]
    n_pred = totals["n_pred"]
    tp_inst = totals["tp_inst"]
    tp_fdi = totals["tp_fdi"]

    fp_inst = n_pred - tp_inst
    fn_inst = n_gt - tp_inst
    fp_fdi = n_pred - tp_fdi
    fn_fdi = n_gt - tp_fdi

    out: dict[str, float] = {
        # primary
        "fdi_instance_f1": _f1(tp_fdi, fp_fdi, fn_fdi),
        "fdi_instance_precision": _safe_div(tp_fdi, n_pred),
        "fdi_instance_recall": _safe_div(tp_fdi, n_gt),
        # geometric instance level
        "instance_f1": _f1(tp_inst, fp_inst, fn_inst),
        "instance_precision": _safe_div(tp_inst, n_pred),
        "instance_recall": _safe_div(tp_inst, n_gt),
        # overlap quality
        "matched_dice": _safe_div(totals["dice_sum"], tp_inst),
        "matched_iou": _safe_div(totals["iou_sum"], tp_inst),
        "unmatched_zero_dice": _safe_div(totals["dice_sum"], n_gt),
        "fdi_accuracy_matched": _safe_div(tp_fdi, tp_inst),
        # panoptic quality, class agnostic and FDI aware
        "panoptic_quality": _safe_div(
            totals["iou_sum"], tp_inst + 0.5 * fp_inst + 0.5 * fn_inst
        ),
        # foreground / area level
        "foreground_dice": _safe_div(
            2.0 * totals["inter_area"], totals["gt_area"] + totals["pred_area"]
        ),
        "area_precision": _safe_div(totals["inter_area"], totals["pred_area"]),
        "area_recall": _safe_div(totals["inter_area"], totals["gt_area"]),
        # counts
        "n_images": totals["n_images"],
        "n_gt_instances": n_gt,
        "n_pred_instances": n_pred,
        "n_true_positive": tp_inst,
        "n_false_positive": fp_inst,
        "n_false_negative": fn_inst,
        "n_fdi_errors": totals["fdi_wrong"],
        "n_merges": totals["merges"],
        "n_splits": totals["splits"],
        # completeness
        "complete_image_rate": _safe_div(totals["complete_image"], totals["n_images"]),
        "n_complete_images": totals["complete_image"],
        # runtime
        "mean_runtime_s": _safe_div(totals["runtime_s"], totals["n_images"]),
        "peak_gpu_mb": max((ev.peak_gpu_mb for ev in evals), default=0.0),
    }

    bp = _safe_div(totals["boundary_tp_pred"], totals["boundary_n_pred"])
    br = _safe_div(totals["boundary_tp_gt"], totals["boundary_n_gt"])
    out["boundary_precision"] = bp
    out["boundary_recall"] = br
    finite = not (math.isnan(bp) or math.isnan(br))
    out["boundary_f"] = (
        2 * bp * br / (bp + br) if finite and (bp + br) > 0 else float("nan")
    )

    # complete-patient correctness
    by_patient: dict[str, list[ImageEval]] = {}
    for ev in evals:
        by_patient.setdefault(ev.patient_id, []).append(ev)
    complete_patients = sum(
        1 for evs in by_patient.values() if all(e.complete_image for e in evs)
    )
    out["n_patients"] = float(len(by_patient))
    out["n_complete_patients"] = float(complete_patients)
    out["complete_patient_rate"] = _safe_div(complete_patients, len(by_patient))

    return out


def aggregate_by_view(evals: list[ImageEval]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[ImageEval]] = {}
    for ev in evals:
        groups.setdefault(ev.view_label, []).append(ev)
    return {view: aggregate(items) for view, items in sorted(groups.items())}


def average_precision(evals: list[ImageEval], n_gt: int | None = None) -> float:
    """COCO-style AP from per-instance scores.

    Only valid for methods that emit a confidence per instance; returns NaN
    otherwise, which is how it is reported in the paper.
    """
    if not evals or not all(ev.has_scores for ev in evals):
        return float("nan")
    entries: list[tuple[float, int]] = []
    total_gt = 0
    for ev in evals:
        total_gt += ev.n_gt
        entries += [(s, 1) for s in ev.scores_tp]
        entries += [(s, 0) for s in ev.scores_fp]
    if n_gt is not None:
        total_gt = n_gt
    if not entries or total_gt == 0:
        return float("nan")
    entries.sort(key=lambda t: -t[0])
    tp = fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for _, is_tp in entries:
        tp += is_tp
        fp += 1 - is_tp
        precisions.append(tp / (tp + fp))
        recalls.append(tp / total_gt)
    # 101-point interpolated AP
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        candidates = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(candidates) if candidates else 0.0
    return ap / 101.0
