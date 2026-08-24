"""R3 degenerate-mask fallback.

The exception path was the only guard on the SAM 3 ROI, and the only exception the
upstream code raises is an empty box after cleanup.  The failure that actually
occurs is different: a mask that is large, non-empty and nearly disjoint from the
dentition, which raises nothing.  These tests pin the guard that catches it.
"""

from __future__ import annotations

import numpy as np
import pytest

from iop_compass.roi import _sam3_upstream
from iop_compass.roi.sam3_roi import Sam3Roi


def _upstream_output(roi_mask: np.ndarray, box: tuple[int, int, int, int]) -> dict:
    """Minimal stand-in for `make_sam3_roi_and_exclusion`."""
    return {
        "roi_mask": roi_mask,
        "allowed_mask": roi_mask,
        "exclusion_soft_mask": np.zeros_like(roi_mask),
        "exclusion_mask": np.zeros_like(roi_mask),
        "roi_box": box,
    }


def _healthy_mask(h: int = 100, w: int = 100) -> tuple[np.ndarray, tuple]:
    """A dentition-like mask: fills nearly all of its own bounding box."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:80, 10:90] = 1
    return mask, (10, 20, 90, 80)


def _degenerate_mask(h: int = 100, w: int = 100) -> tuple[np.ndarray, tuple]:
    """A soft-tissue-interior mask: a thin ring, so its box is mostly empty.

    This is the observed occlusal failure shape -- the box is right, the mask
    inside it is not.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:80, 10:90] = 1
    mask[26:74, 16:84] = 0  # hollow out the middle: ring of ~24% box fill
    return mask, (10, 20, 90, 80)


def _roi(monkeypatch, mask, box, **kwargs) -> object:
    strategy = Sam3Roi(params={"long_side": 1400}, device="cpu", **kwargs)
    monkeypatch.setattr(strategy, "_get_processor", lambda: object())
    monkeypatch.setattr(
        _sam3_upstream,
        "make_sam3_roi_and_exclusion",
        lambda *a, **k: _upstream_output(mask, box),
    )
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    return strategy(image, "upper_occlusal")


def test_healthy_mask_is_kept(monkeypatch):
    mask, box = _healthy_mask()
    result = _roi(monkeypatch, mask, box)
    assert result.fallback is False
    assert result.box == box
    assert result.extra["box_fill"] == pytest.approx(1.0)


def test_degenerate_mask_falls_back_to_full_image(monkeypatch):
    mask, box = _degenerate_mask()
    result = _roi(monkeypatch, mask, box)
    assert result.fallback is True
    assert "min_box_fill" in result.fallback_reason
    # the fallback must be the full frame, so no tooth can be clipped away
    assert result.box == (0, 0, mask.shape[1], mask.shape[0])
    assert result.roi_mask.all()
    assert result.extra["box_fill"] < Sam3Roi.DEFAULT_MIN_BOX_FILL


def test_box_fill_is_recorded_on_every_image(monkeypatch):
    """Logged even when the guard does not fire, so the threshold stays re-selectable."""
    mask, box = _healthy_mask()
    assert "box_fill" in _roi(monkeypatch, mask, box).extra


def test_guard_is_disabled_by_a_zero_threshold(monkeypatch):
    mask, box = _degenerate_mask()
    result = _roi(monkeypatch, mask, box, min_box_fill=0.0)
    assert result.fallback is False


def test_threshold_comes_from_the_config_block(monkeypatch):
    """`min_box_fill` travels inside `sam3_roi:` but must not reach the upstream dataclass."""
    strategy = Sam3Roi(params={"long_side": 1400, "min_box_fill": 0.42}, device="cpu")
    assert strategy.min_box_fill == pytest.approx(0.42)
    assert not hasattr(strategy.params, "min_box_fill")
    assert strategy.frozen_parameters()["min_box_fill"] == pytest.approx(0.42)
