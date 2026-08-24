"""R3 - SAM 3 concept-prompted intraoral ROI.

Thin wrapper over :mod:`iop_compass.roi._sam3_upstream`, the vendored upstream
extractor.  The wrapper adds:

* a frozen local checkpoint (no runtime Hugging Face download),
* the documented full-image fallback, on two conditions: the upstream call
  raising (its only raise is an empty box after cleanup) **and** a returned mask
  that is implausible for a dentition, see :meth:`Sam3Roi._degenerate_reason`,
* runtime accounting and the frozen-parameter record.

The second condition is the one that matters in practice: the concept prompts can
select the soft-tissue interior of the arch instead of the dentition, so the mask
is large but nearly disjoint from the teeth.  A guard that only catches empty
masks cannot see that, so the wrapper adds a mask-plausibility check that uses no
ground truth.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

from .base import RoiResult, clip_box, full_image_result


def resolve_sam3_checkpoint(explicit: str | None = None) -> Path | None:
    if explicit:
        return Path(explicit)
    env = os.environ.get("SAM3_CHECKPOINT")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[3]
    candidate = repo / "third_party" / "sam3.pt"
    return candidate if candidate.exists() else None


class Sam3Roi:
    """Callable ROI strategy using SAM 3 concept prompts."""

    name = "R3_sam3"

    #: A mask filling less than this fraction of its own bounding box is treated as
    #: degenerate.
    DEFAULT_MIN_BOX_FILL = 0.70

    def __init__(
        self,
        params: dict | None = None,
        device: str = "cuda",
        checkpoint_path: str | Path | None = None,
        allow_fallback: bool = True,
        min_box_fill: float | None = None,
    ):
        from ._sam3_upstream import Sam3RoiParams

        # `min_box_fill` is a wrapper-level guard, not an upstream parameter: it is
        # accepted inside the `sam3_roi` config block for provenance, but the
        # vendored dataclass must not see it.
        params = dict(params or {})
        from_config = params.pop("min_box_fill", None)
        self.min_box_fill = float(
            min_box_fill if min_box_fill is not None
            else from_config if from_config is not None
            else self.DEFAULT_MIN_BOX_FILL
        )
        self.params = Sam3RoiParams(**params)
        self.device = device
        self.checkpoint_path = resolve_sam3_checkpoint(
            str(checkpoint_path) if checkpoint_path else None
        )
        self.allow_fallback = allow_fallback
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from ._sam3_upstream import build_sam3

            self._processor = build_sam3(
                device=self.device,
                checkpoint_path=str(self.checkpoint_path) if self.checkpoint_path else None,
            )
        return self._processor

    def checkpoint_sha256(self) -> str | None:
        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            return None
        from ..data.adapter import sha256_file

        return sha256_file(Path(self.checkpoint_path))

    def frozen_parameters(self) -> dict:
        from dataclasses import asdict

        return {
            "params": asdict(self.params),
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256(),
            "allow_fallback": self.allow_fallback,
            "min_box_fill": self.min_box_fill,
        }

    def _box_fill(self, roi_mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
        """Fraction of the ROI box that the ROI mask actually fills.

        Ground-truth-free by construction.  The box is derived from the mask, so a
        healthy dentition mask fills nearly all of it; a mask that traces the
        soft-tissue interior instead leaves the arch itself outside, and the fill
        drops well below the healthy range.
        """
        x0, y0, x1, y1 = box
        box_area = (x1 - x0) * (y1 - y0)
        if box_area <= 0:
            return 0.0
        return float((roi_mask[y0:y1, x0:x1] > 0).sum()) / float(box_area)

    def _degenerate_reason(self, fill: float) -> str:
        if fill < self.min_box_fill:
            return f"roi mask fills {fill:.3f} of its box, below min_box_fill {self.min_box_fill:.2f}"
        return ""

    def __call__(self, image_bgr: np.ndarray, view_label: str) -> RoiResult:
        from ._sam3_upstream import make_sam3_roi_and_exclusion

        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        start = time.perf_counter()
        try:
            out = make_sam3_roi_and_exclusion(
                self._get_processor(), rgb, self.params, view=view_label
            )
        except Exception as exc:  # documented fallback path
            if not self.allow_fallback:
                raise
            result = full_image_result((h, w), self.name, True, str(exc)[:200])
            result.runtime_s = time.perf_counter() - start
            return result

        runtime = time.perf_counter() - start
        box = clip_box(out["roi_box"], (h, w))
        roi_mask = (np.asarray(out["roi_mask"]) > 0).astype(np.uint8)

        # The upstream call can succeed and still return a mask that is disjoint from
        # the dentition; that is the failure mode the exception path cannot see.
        fill = self._box_fill(roi_mask, box)
        reason = self._degenerate_reason(fill)
        if reason and self.allow_fallback:
            result = full_image_result((h, w), self.name, True, reason)
            result.runtime_s = runtime
            result.extra["box_fill"] = fill
            result.extra["rejected_box"] = box
            return result

        # Post-processing clips to `roi_mask` and uses `exclusion_soft_mask`, while
        # the photometric guidance uses the dilated `allowed_mask`; this mirrors
        # the upstream pipeline exactly (Model_run.py:305-341).
        return RoiResult(
            roi_mask=roi_mask,
            allowed_mask=(np.asarray(out["allowed_mask"]) > 0).astype(np.uint8),
            exclusion_mask=(np.asarray(out["exclusion_soft_mask"]) > 0).astype(np.uint8),
            box=box,
            strategy=self.name,
            runtime_s=runtime,
            extra={
                "sam_roi_area": int(roi_mask.sum()),
                "hard_exclusion_area": int(
                    (np.asarray(out["exclusion_mask"]) > 0).sum()
                ),
                # Recorded on every image, not only on rejections, so `min_box_fill`
                # can be re-selected on a future validation split from the logs.
                "box_fill": fill,
            },
        )
