"""R1 - view-specific geometric ROI prior.

For each clinical view the union of the reference tooth masks of the *training*
patients is reduced to a normalised bounding box, and the crop is the robust
percentile envelope of those boxes plus a fixed margin.  Robust percentiles are
used instead of the extrema so that a single mis-framed acquisition cannot widen
the prior to the whole image.

The five boxes are fitted once, written to a JSON file and frozen before any
test-set evaluation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .base import RoiResult, box_to_masks, clip_box, full_image_result


@dataclass
class GeometricPriorConfig:
    low_percentile: float = 2.0
    high_percentile: float = 98.0
    margin_frac: float = 0.05

    @classmethod
    def from_dict(cls, payload: dict | None) -> "GeometricPriorConfig":
        payload = dict(payload or {})
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class ViewPrior:
    """Normalised crop rectangle in ``[0, 1]`` image coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float
    n_images: int

    def to_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Denormalise to an exclusive-end pixel box."""
        return clip_box(
            (
                self.x0 * width,
                self.y0 * height,
                self.x1 * width,
                self.y1 * height,
            ),
            (height, width),
        )


def normalised_tooth_box(
    instance_raster: np.ndarray,
) -> tuple[float, float, float, float] | None:
    """Normalised bounding box of the tooth union in one reference raster."""
    ys, xs = np.nonzero(instance_raster)
    if xs.size == 0:
        return None
    h, w = instance_raster.shape
    return (
        float(xs.min()) / w,
        float(ys.min()) / h,
        float(xs.max() + 1) / w,
        float(ys.max() + 1) / h,
    )


def fit_priors(
    boxes_by_view: dict[str, list[tuple[float, float, float, float]]],
    config: GeometricPriorConfig,
) -> dict[str, ViewPrior]:
    priors: dict[str, ViewPrior] = {}
    for view, boxes in sorted(boxes_by_view.items()):
        if not boxes:
            continue
        arr = np.asarray(boxes, dtype=float)
        x0 = float(np.percentile(arr[:, 0], config.low_percentile))
        y0 = float(np.percentile(arr[:, 1], config.low_percentile))
        x1 = float(np.percentile(arr[:, 2], config.high_percentile))
        y1 = float(np.percentile(arr[:, 3], config.high_percentile))
        mx = config.margin_frac * max(x1 - x0, 1e-6)
        my = config.margin_frac * max(y1 - y0, 1e-6)
        priors[view] = ViewPrior(
            x0=max(0.0, x0 - mx),
            y0=max(0.0, y0 - my),
            x1=min(1.0, x1 + mx),
            y1=min(1.0, y1 + my),
            n_images=len(boxes),
        )
    return priors


def save_priors(
    priors: dict[str, ViewPrior],
    config: GeometricPriorConfig,
    path: Path,
    provenance: dict | None = None,
) -> None:
    """Write the prior together with the split it was fitted on.

    Without the provenance block a prior cannot be told apart from one fitted on a
    different cohort or a different replicate, and reusing such a prior puts
    validation or test patients into the fit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "provenance": provenance or {},
        "priors": {view: asdict(prior) for view, prior in sorted(priors.items())},
    }
    path.write_text(json.dumps(payload, indent=1))


def load_priors(
    path: Path,
) -> tuple[dict[str, ViewPrior], GeometricPriorConfig, dict]:
    payload = json.loads(Path(path).read_text())
    priors = {view: ViewPrior(**row) for view, row in payload["priors"].items()}
    return (
        priors,
        GeometricPriorConfig.from_dict(payload.get("config")),
        payload.get("provenance") or {},
    )


class GeometricRoi:
    """Callable ROI strategy backed by the frozen per-view priors."""

    name = "R1_geometric"

    def __init__(self, priors: dict[str, ViewPrior]):
        self.priors = priors

    def __call__(self, image_bgr: np.ndarray, view_label: str) -> RoiResult:
        h, w = image_bgr.shape[:2]
        prior = self.priors.get(view_label)
        if prior is None:
            return full_image_result((h, w), self.name, True, "no prior for view")
        box = prior.to_box(w, h)
        roi, exclusion = box_to_masks(box, (h, w))
        return RoiResult(
            roi_mask=roi,
            exclusion_mask=exclusion,
            box=box,
            strategy=self.name,
            extra={"prior_n_images": prior.n_images},
        )
