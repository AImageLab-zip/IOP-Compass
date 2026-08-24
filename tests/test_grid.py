"""The benchmark grid is fully crossed and its ids round-trip."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.segmentation import grid  # noqa: E402


def test_grid_is_fully_crossed():
    cells = grid.all_cells()
    assert len(cells) == 16
    assert len(set(cells)) == 16
    expected = {
        (roi, seg, post)
        for roi in grid.ROI_LEVELS
        for seg in grid.SEG_LEVELS
        for post in grid.POST_LEVELS
    }
    assert {grid.parse_cell(c) for c in cells} == expected


def test_cell_ids_round_trip():
    for cell in grid.all_cells():
        roi, seg, post = grid.parse_cell(cell)
        assert grid.cell_id(roi, seg, post) == cell


def test_post_accepts_bool_for_convenience():
    assert grid.cell_id("no-roi", "sat", True) == grid.cell_id("no-roi", "sat", "post")
    assert grid.cell_id("no-roi", "sat", False) == grid.cell_id(
        "no-roi", "sat", "no-post"
    )


@pytest.mark.parametrize(
    "bad",
    [
        "no-roi+sat",
        "no-roi+sat+post+extra",
        "nope+sat+post",
        "no-roi+nope+post",
        "no-roi+sat+nope",
        "",
    ],
)
def test_malformed_cell_ids_are_rejected(bad):
    with pytest.raises(grid.GridError):
        grid.parse_cell(bad)


def test_only_raw_cells_need_a_forward_pass():
    raw = grid.raw_cells()
    assert len(raw) == 8
    assert all(grid.parse_cell(c)[2] == "no-post" for c in raw)


def test_post_pairs_cover_every_post_cell():
    pairs = grid.post_pairs()
    assert len(pairs) == 8
    assert {b for _, b in pairs} == {
        c for c in grid.all_cells() if grid.parse_cell(c)[2] == "post"
    }
    for source, target in pairs:
        # a pair may differ only on the post axis
        assert grid.parse_cell(source)[:2] == grid.parse_cell(target)[:2]
        assert grid.parse_cell(source)[2] == "no-post"


def test_every_cell_carries_the_replicate_suffix():
    """Two replicates measure different hold-outs and must not share a directory."""
    for cell in grid.all_cells():
        a = grid.prediction_dir_name(cell, 1)
        b = grid.prediction_dir_name(cell, 2)
        assert a != b
        assert grid.parse_result_name(a) == (cell, 1)
        assert grid.parse_result_name(b) == (cell, 2)


def test_replicate_is_required():
    with pytest.raises(grid.GridError):
        grid.prediction_dir_name("no-roi+sat+no-post", None)
    with pytest.raises(grid.GridError):
        grid.prediction_dir_name("no-roi+sat+no-post", -1)


def test_non_grid_result_names_are_not_grid_results():
    """So the evaluator can keep reading non-grid result files without confusion."""
    for name in ("index", "grid"):
        assert grid.parse_result_name(name) is None


def test_roi_strategy_mapping_covers_the_factory():
    from iop_compass.roi.factory import ALL_STRATEGIES

    assert set(grid.ROI_STRATEGY.values()) == set(ALL_STRATEGIES)
    assert set(grid.STRATEGY_ROI) == set(ALL_STRATEGIES)
    for level in grid.ROI_LEVELS:
        assert grid.STRATEGY_ROI[grid.ROI_STRATEGY[level]] == level


def test_strategies_for_returns_canonical_order_without_duplicates():
    cells = ["sam3+sat+post", "geometric+mask-rcnn+no-post", "geometric+sat+no-post"]
    assert grid.strategies_for(cells) == ["R1_geometric", "R3_sam3"]


def test_guidance_is_declared_for_every_segmenter():
    """The one deliberate asymmetry between backends must be explicit, not implied."""
    assert set(grid.ROI_GUIDANCE) == set(grid.SEG_LEVELS)
    assert grid.ROI_GUIDANCE["sat"] is True
    assert grid.ROI_GUIDANCE["mask-rcnn"] is False


def test_every_level_has_a_label():
    for level in grid.ROI_LEVELS + grid.SEG_LEVELS + grid.POST_LEVELS:
        assert grid.LABELS[level]
    for cell in grid.all_cells():
        assert grid.label(cell)


def test_result_names_identify_their_replicate():
    """The evaluator filters discovered directories on this.

    Without it, ``--all-variants --replicate 1`` would also pick up replicate 2's
    predictions and score them against replicate 1's split -- different patients,
    silently.
    """
    names = [grid.prediction_dir_name(c, r) for c in grid.all_cells() for r in (1, 2, 3)]
    assert len(set(names)) == len(names)
    for name in names:
        cell, rep = grid.parse_result_name(name)
        assert name == grid.prediction_dir_name(cell, rep)

    for rep in (1, 2, 3):
        mine = [n for n in names if grid.parse_result_name(n)[1] == rep]
        assert len(mine) == 16


def test_cell_ids_never_collide_with_the_replicate_marker():
    """No level may contain the separator or the replicate marker."""
    for level in grid.ROI_LEVELS + grid.SEG_LEVELS + grid.POST_LEVELS:
        assert grid.SEP not in level
    # a cell id must not itself parse as a result name
    for cell in grid.all_cells():
        assert grid.parse_result_name(cell) is None
