"""Hungarian instance matching and the post-processing rules."""

from __future__ import annotations

import numpy as np
import pytest

from iop_compass.segmentation.matching import (
    count_merge_split,
    match_instances,
    overlap_matrices,
)
from iop_compass.segmentation.postprocessing import (
    ABLATABLE_RULES,
    PostProcessParams,
    postprocess_instances,
    postprocess_raster,
)

SIZE = 64


def box(x0: int, y0: int, x1: int, y1: int, size: int = SIZE) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_overlap_matrices_are_exact():
    a = [box(0, 0, 10, 10)]
    b = [box(0, 0, 10, 10), box(5, 0, 15, 10)]
    inter, iou, dice = overlap_matrices(a, b)
    assert inter[0, 0] == 100
    assert iou[0, 0] == pytest.approx(1.0)
    assert inter[0, 1] == 50
    assert iou[0, 1] == pytest.approx(50 / 150)
    assert dice[0, 1] == pytest.approx(2 * 50 / 200)


def test_empty_inputs_give_empty_matrices():
    inter, iou, dice = overlap_matrices([], [box(0, 0, 5, 5)])
    assert inter.shape == (0, 1)
    result = match_instances([], [], [box(0, 0, 5, 5)], [11])
    assert (result.tp, result.fp, result.fn) == (0, 1, 0)


def test_matching_is_one_to_one():
    gt = [box(0, 0, 20, 20)]
    # two predictions overlap the same reference instance strongly
    pred = [box(0, 0, 20, 20), box(1, 1, 20, 20)]
    result = match_instances(gt, [11], pred, [11, 11])
    assert result.tp == 1
    assert result.fp == 1
    assert len(result.unmatched_pred) == 1


def test_hungarian_beats_greedy_on_a_crafted_case():
    # Greedy takes the globally best pair first and then has nothing left for the
    # second reference instance; the optimal assignment matches both.
    gt = [box(0, 0, 20, 20), box(18, 0, 38, 20)]
    pred = [box(18, 0, 38, 20), box(0, 0, 19, 20)]
    result = match_instances(gt, [11, 12], pred, [12, 11])
    assert result.tp == 2
    assert all(m.fdi_correct for m in result.matches)


def test_matching_is_class_agnostic_then_fdi_is_scored():
    gt = [box(0, 0, 10, 10)]
    pred = [box(0, 0, 10, 10)]
    result = match_instances(gt, [11], pred, [21])
    assert result.tp == 1
    assert result.fdi_correct == 0
    assert result.fdi_wrong == 1


def test_merge_and_split_counts():
    gt_areas = np.array([100, 100])
    pred_areas = np.array([200])
    inter = np.array([[100.0], [100.0]])
    merges, splits = count_merge_split(inter, gt_areas, pred_areas)
    assert merges == 1 and splits == 0

    inter = np.array([[100.0, 100.0]])
    merges, splits = count_merge_split(inter, np.array([200]), np.array([100, 100]))
    assert merges == 0 and splits == 1


def test_postprocessing_removes_speckle_and_keeps_teeth():
    tooth = box(20, 20, 40, 40)
    speckle = box(1, 1, 3, 3)
    params = PostProcessParams()
    masks, fdis, scores, stats = postprocess_instances(
        [tooth, speckle], [11, 12], [0.9, 0.2], None, None, params, "frontal"
    )
    assert len(masks) == 1
    assert fdis == [11]
    assert stats.n_in == 2 and stats.n_out == 1


def test_postprocessing_clips_to_the_roi():
    tooth = box(0, 0, 10, 10)
    roi = np.zeros((SIZE, SIZE), dtype=np.uint8)
    roi[20:60, 20:60] = 1
    masks, _, _, stats = postprocess_instances(
        [tooth], [11], [1.0], roi, 1 - roi, PostProcessParams(), "frontal"
    )
    assert masks == []
    assert stats.dropped_outside_roi == 1


def test_postprocessing_fills_a_hole():
    tooth = box(20, 20, 40, 40).copy()
    tooth[28:32, 28:32] = False
    masks, _, _, _ = postprocess_instances(
        [tooth], [11], [1.0], None, None, PostProcessParams(), "frontal"
    )
    assert len(masks) == 1
    assert masks[0][28:32, 28:32].all()


def test_duplicate_fdi_resolution_is_opt_in():
    big = box(20, 20, 40, 40)
    small = box(50, 50, 56, 56)
    default = PostProcessParams()
    masks, fdis, _, _ = postprocess_instances(
        [big, small], [11, 11], [1.0, 1.0], None, None, default, "frontal"
    )
    assert len(masks) == 2  # upstream behaviour: duplicates are kept

    strict = PostProcessParams.from_dict(
        {**default.to_dict(), "keep_only_best_component_per_label": True}
    )
    masks, fdis, _, stats = postprocess_instances(
        [big, small], [11, 11], [1.0, 1.0], None, None, strict, "frontal"
    )
    assert len(masks) == 1
    assert stats.dropped_duplicate_fdi == 1
    assert int(masks[0].sum()) == int(big.sum())


def test_far_side_rule_only_applies_to_buccal_views():
    # a narrow sliver at the right border, mostly inside the exclusion region
    sliver = np.zeros((SIZE, SIZE), dtype=bool)
    sliver[30:34, 61:63] = True
    roi = np.ones((SIZE, SIZE), dtype=np.uint8)
    exclusion = np.zeros((SIZE, SIZE), dtype=np.uint8)
    exclusion[:, 58:] = 1
    params = PostProcessParams.from_dict(
        {
            **PostProcessParams().to_dict(),
            "min_component_area": 1,
            "min_label_pixels": 1,
            "min_roi_overlap_pixels": 1,
            "max_exclusion_overlap_frac": 1.0,
            "side_far_max_roi_frac": 1.0,
        }
    )
    kept_frontal, _, _, _ = postprocess_instances(
        [sliver], [11], [1.0], roi, exclusion, params, "frontal"
    )
    kept_left, _, _, stats = postprocess_instances(
        [sliver], [11], [1.0], roi, exclusion, params, "left_buccal"
    )
    assert len(kept_frontal) == 1
    assert kept_left == []
    assert stats.dropped_far_side == 1


def test_every_ablatable_rule_is_switchable():
    # every rule must be switchable from a base that has it enabled, including the
    # duplicate-FDI resolution that the shipped config turns on
    base = PostProcessParams(keep_only_best_component_per_label=True)
    for rule in ABLATABLE_RULES:
        variant = base.without(rule)
        assert variant != base, rule
    stripped = base.without(*ABLATABLE_RULES)
    assert stripped.tooth_close_k == 0
    assert stripped.keep_only_best_component_per_label is False


def test_unknown_rule_name_is_rejected():
    with pytest.raises(KeyError):
        PostProcessParams().without("not_a_rule")


def test_raster_entry_point_matches_the_instance_entry_point():
    raw = np.zeros((SIZE, SIZE), dtype=np.uint8)
    raw[10:20, 10:20] = 11
    raw[30:40, 30:40] = 12
    roi = np.ones((SIZE, SIZE), dtype=np.uint8)
    exclusion = np.zeros((SIZE, SIZE), dtype=np.uint8)
    params = PostProcessParams()
    out = postprocess_raster(raw, roi, exclusion, params, "frontal")
    assert set(np.unique(out)) == {0, 11, 12}

    masks, fdis, _, _ = postprocess_instances(
        [raw == 11, raw == 12], [11, 12], None, roi, exclusion, params, "frontal"
    )
    rebuilt = np.zeros_like(out)
    for mask, fdi in zip(masks, fdis):
        rebuilt[mask] = fdi
    assert (rebuilt == out).all()
