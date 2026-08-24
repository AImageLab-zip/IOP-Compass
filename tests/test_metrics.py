"""Known-answer tests for the shared evaluator.

Every case is constructed so the correct value of every metric can be computed by
hand, which is what makes the reported numbers auditable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from iop_compass.reporting.bootstrap import (
    bootstrap_metrics,
    is_inconclusive,
    paired_bootstrap,
)
from iop_compass.segmentation.matching import match_instances
from iop_compass.segmentation.metrics import aggregate, evaluate_image

SIZE = 64


def box(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def three_teeth():
    masks = [box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 50, 10)]
    fdis = [11, 12, 13]
    return masks, fdis


def test_perfect_prediction():
    masks, fdis = three_teeth()
    ev, result = evaluate_image("i", "p", "frontal", masks, fdis, masks, fdis, [1.0] * 3)
    assert result.tp == 3 and result.fp == 0 and result.fn == 0
    metrics = aggregate([ev])
    assert metrics["fdi_instance_f1"] == pytest.approx(1.0)
    assert metrics["instance_f1"] == pytest.approx(1.0)
    assert metrics["matched_dice"] == pytest.approx(1.0)
    assert metrics["complete_image_rate"] == pytest.approx(1.0)
    assert metrics["n_false_positive"] == 0
    assert metrics["n_false_negative"] == 0


def test_missed_tooth_is_a_false_negative():
    masks, fdis = three_teeth()
    ev, result = evaluate_image(
        "i", "p", "frontal", masks, fdis, masks[:2], fdis[:2], [1.0] * 2
    )
    assert (result.tp, result.fp, result.fn) == (2, 0, 1)
    metrics = aggregate([ev])
    # F1 = 2*2 / (2*2 + 0 + 1) = 0.8
    assert metrics["fdi_instance_f1"] == pytest.approx(0.8)
    assert metrics["complete_image_rate"] == 0.0
    # matched Dice is 1.0 for the two matches, unmatched-zero Dice is 2/3
    assert metrics["matched_dice"] == pytest.approx(1.0)
    assert metrics["unmatched_zero_dice"] == pytest.approx(2 / 3)


def test_extra_prediction_is_a_false_positive():
    masks, fdis = three_teeth()
    pred = masks + [box(0, 30, 10, 40)]
    ev, result = evaluate_image(
        "i", "p", "frontal", masks, fdis, pred, fdis + [14], [1.0] * 4
    )
    assert (result.tp, result.fp, result.fn) == (3, 1, 0)
    metrics = aggregate([ev])
    assert metrics["fdi_instance_f1"] == pytest.approx(2 * 3 / (2 * 3 + 1 + 0))
    assert metrics["complete_image_rate"] == 0.0


def test_wrong_fdi_is_not_a_double_error():
    masks, fdis = three_teeth()
    wrong = [11, 21, 13]  # middle tooth mislabelled
    ev, result = evaluate_image("i", "p", "frontal", masks, fdis, masks, wrong, [1.0] * 3)
    assert (result.tp, result.fp, result.fn) == (3, 0, 0)
    assert result.fdi_correct == 2
    metrics = aggregate([ev])
    assert metrics["instance_f1"] == pytest.approx(1.0)
    # FDI-aware: TP=2, FP=1, FN=1
    assert metrics["fdi_instance_f1"] == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))
    assert metrics["fdi_accuracy_matched"] == pytest.approx(2 / 3)
    assert metrics["n_fdi_errors"] == 1
    assert metrics["complete_image_rate"] == 0.0


def test_merge_is_detected():
    masks, fdis = three_teeth()
    merged = [box(0, 0, 30, 10)]  # one mask covering the first two teeth
    ev, result = evaluate_image("i", "p", "frontal", masks, fdis, merged, [11], [1.0])
    assert result.merges == 1
    metrics = aggregate([ev])
    assert metrics["n_merges"] == 1


def test_split_is_detected():
    gt = [box(0, 0, 20, 10)]
    pred = [box(0, 0, 10, 10), box(10, 0, 20, 10)]
    ev, result = evaluate_image("i", "p", "frontal", gt, [11], pred, [11, 11], [1.0, 1.0])
    assert result.splits == 1
    metrics = aggregate([ev])
    assert metrics["n_splits"] == 1


def test_iou_threshold_rejects_weak_overlap():
    gt = [box(0, 0, 20, 20)]
    # 10x20 inside a 20x20 reference: IoU = 200/400 = 0.50 -> accepted at >= 0.50
    good = [box(0, 0, 10, 20)]
    assert match_instances(gt, [11], good, [11], iou_threshold=0.50).tp == 1
    # 9x20: IoU = 180/400 = 0.45 -> rejected
    weak = [box(0, 0, 9, 20)]
    assert match_instances(gt, [11], weak, [11], iou_threshold=0.50).tp == 0


def test_empty_prediction():
    masks, fdis = three_teeth()
    ev, result = evaluate_image("i", "p", "frontal", masks, fdis, [], [], [])
    assert (result.tp, result.fp, result.fn) == (0, 0, 3)
    metrics = aggregate([ev])
    assert metrics["fdi_instance_f1"] == 0.0
    assert math.isnan(metrics["matched_dice"])
    assert metrics["unmatched_zero_dice"] == 0.0


def test_complete_patient_needs_every_view():
    masks, fdis = three_teeth()
    good, _ = evaluate_image("a", "p1", "frontal", masks, fdis, masks, fdis, [1.0] * 3)
    bad, _ = evaluate_image(
        "b", "p1", "left_buccal", masks, fdis, masks[:2], fdis[:2], [1.0] * 2
    )
    metrics = aggregate([good, bad])
    assert metrics["n_complete_images"] == 1
    assert metrics["n_complete_patients"] == 0
    metrics_ok = aggregate([good])
    assert metrics_ok["n_complete_patients"] == 1


def test_aggregate_is_additive_over_images():
    masks, fdis = three_teeth()
    a, _ = evaluate_image("a", "p1", "frontal", masks, fdis, masks, fdis, [1.0] * 3)
    b, _ = evaluate_image(
        "b", "p2", "frontal", masks, fdis, masks[:2], fdis[:2], [1.0] * 2
    )
    pooled = aggregate([a, b])
    assert pooled["n_gt_instances"] == 6
    assert pooled["n_true_positive"] == 5
    assert pooled["n_false_negative"] == 1
    assert pooled["fdi_instance_f1"] == pytest.approx(2 * 5 / (2 * 5 + 0 + 1))


def test_bootstrap_keeps_patients_together_and_brackets_the_point():
    masks, fdis = three_teeth()
    evals = []
    for i in range(12):
        pred = masks if i % 3 else masks[:2]
        pred_fdis = fdis if i % 3 else fdis[:2]
        for view in ("frontal", "left_buccal"):
            ev, _ = evaluate_image(
                f"img{i}{view}",
                f"p{i}",
                view,
                masks,
                fdis,
                pred,
                pred_fdis,
                [1.0] * len(pred),
            )
            evals.append(ev)
    ci = bootstrap_metrics(evals, ["fdi_instance_f1"], n_replicates=200, seed=1)
    interval = ci["fdi_instance_f1"]
    assert interval.low <= interval.point <= interval.high
    assert interval.n_replicates == 200


def test_paired_bootstrap_flags_no_difference_as_inconclusive():
    masks, fdis = three_teeth()
    evals = []
    for i in range(10):
        ev, _ = evaluate_image(
            f"img{i}", f"p{i}", "frontal", masks, fdis, masks, fdis, [1.0] * 3
        )
        evals.append(ev)
    paired = paired_bootstrap(evals, evals, ["fdi_instance_f1"], n_replicates=100, seed=3)
    assert paired["fdi_instance_f1"].point == pytest.approx(0.0)
    assert is_inconclusive(paired["fdi_instance_f1"])
