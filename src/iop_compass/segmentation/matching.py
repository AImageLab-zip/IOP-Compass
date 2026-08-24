"""Instance matching between a prediction and the reference annotation.

One-to-one assignment is solved optimally with the Hungarian algorithm on the
mask-IoU matrix, then filtered at ``iou_threshold``.  Matching is
*class-agnostic*: FDI correctness is scored after the assignment, so a mask that
is geometrically right but labelled wrong is counted as an FDI error rather than
as a simultaneous false positive and false negative.

Merge / split events are read off the raw overlap matrix, before assignment, so
they are not hidden by the one-to-one constraint.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Match:
    gt_index: int
    pred_index: int
    iou: float
    dice: float
    gt_fdi: int | None
    pred_fdi: int | None

    @property
    def fdi_correct(self) -> bool:
        return (
            self.gt_fdi is not None
            and self.pred_fdi is not None
            and self.gt_fdi == self.pred_fdi
        )


@dataclass
class MatchResult:
    matches: list[Match]
    unmatched_gt: list[int]
    unmatched_pred: list[int]
    n_gt: int
    n_pred: int
    merges: int = 0
    splits: int = 0
    iou_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    @property
    def tp(self) -> int:
        return len(self.matches)

    @property
    def fp(self) -> int:
        return self.n_pred - self.tp

    @property
    def fn(self) -> int:
        return self.n_gt - self.tp

    @property
    def fdi_correct(self) -> int:
        return sum(1 for m in self.matches if m.fdi_correct)

    @property
    def fdi_wrong(self) -> int:
        return self.tp - self.fdi_correct


def _flatten(masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pack boolean masks into a 2-D array plus their areas."""
    if not masks:
        return np.zeros((0, 0), dtype=bool), np.zeros(0, dtype=np.int64)
    stack = np.stack([m.reshape(-1) for m in masks]).astype(bool)
    return stack, stack.sum(axis=1).astype(np.int64)


def overlap_matrices(
    gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(intersection, iou, dice)`` matrices of shape ``(n_gt, n_pred)``."""
    gt_flat, gt_area = _flatten(gt_masks)
    pred_flat, pred_area = _flatten(pred_masks)
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    if n_gt == 0 or n_pred == 0:
        empty = np.zeros((n_gt, n_pred), dtype=np.float64)
        return empty.copy(), empty.copy(), empty.copy()

    # float32 rather than uint8: a uint8 matmul silently wraps once the
    # intersection exceeds 255 pixels, which is every real tooth.  float32 is
    # exact for integers below 2**24, far above any mask size used here.
    inter = (gt_flat.astype(np.float32) @ pred_flat.astype(np.float32).T).astype(
        np.float64
    )
    union = gt_area[:, None] + pred_area[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
        total = gt_area[:, None] + pred_area[None, :]
        dice = np.where(total > 0, 2.0 * inter / total, 0.0)
    return inter, iou, dice


def count_merge_split(
    inter: np.ndarray,
    gt_areas: np.ndarray,
    pred_areas: np.ndarray,
    coverage: float = 0.5,
) -> tuple[int, int]:
    """Count merge and split events from the raw overlap matrix.

    * merge: one prediction covers at least ``coverage`` of two or more reference
      instances, i.e. two teeth were fused into one mask.
    * split: two or more predictions each have at least ``coverage`` of their own
      area inside a single reference instance, i.e. one tooth was fragmented.
    """
    if inter.size == 0:
        return 0, 0
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_of_gt = np.where(gt_areas[:, None] > 0, inter / gt_areas[:, None], 0.0)
        frac_of_pred = np.where(pred_areas[None, :] > 0, inter / pred_areas[None, :], 0.0)
    per_pred_gt_hits = (frac_of_gt >= coverage).sum(axis=0)
    per_gt_pred_hits = (frac_of_pred >= coverage).sum(axis=1)
    merges = int(np.clip(per_pred_gt_hits - 1, 0, None).sum())
    splits = int(np.clip(per_gt_pred_hits - 1, 0, None).sum())
    return merges, splits


def match_instances(
    gt_masks: list[np.ndarray],
    gt_fdis: list[int | None],
    pred_masks: list[np.ndarray],
    pred_fdis: list[int | None],
    iou_threshold: float = 0.50,
    merge_split_coverage: float = 0.5,
) -> MatchResult:
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    inter, iou, dice = overlap_matrices(gt_masks, pred_masks)

    _, gt_area = _flatten(gt_masks)
    _, pred_area = _flatten(pred_masks)
    merges, splits = count_merge_split(inter, gt_area, pred_area, merge_split_coverage)

    matches: list[Match] = []
    if n_gt and n_pred:
        rows, cols = linear_sum_assignment(-iou)
        for r, c in zip(rows, cols):
            if iou[r, c] >= iou_threshold:
                matches.append(
                    Match(
                        gt_index=int(r),
                        pred_index=int(c),
                        iou=float(iou[r, c]),
                        dice=float(dice[r, c]),
                        gt_fdi=gt_fdis[r],
                        pred_fdi=pred_fdis[c],
                    )
                )

    matched_gt = {m.gt_index for m in matches}
    matched_pred = {m.pred_index for m in matches}
    return MatchResult(
        matches=sorted(matches, key=lambda m: m.gt_index),
        unmatched_gt=[i for i in range(n_gt) if i not in matched_gt],
        unmatched_pred=[i for i in range(n_pred) if i not in matched_pred],
        n_gt=n_gt,
        n_pred=n_pred,
        merges=merges,
        splits=splits,
        iou_matrix=iou,
    )
