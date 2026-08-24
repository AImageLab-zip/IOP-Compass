"""ROI evaluation.

The ROI stage is judged on its own terms, before any downstream segmentation:
how much of the reference tooth signal survives the crop, and how much image area
is paid for it.  A strategy that improves downstream Dice while discarding tooth
pixels is a bad ROI, and this is where that shows up.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class RoiImageEval:
    image_id: str
    patient_id: str
    view_label: str
    strategy: str
    gt_tooth_pixels: int
    retained_tooth_pixels: int
    retained_tooth_pixels_box: int
    image_pixels: int
    roi_pixels: int
    box_pixels: int
    fallback: int
    runtime_s: float = 0.0
    peak_gpu_mb: float = 0.0
    # Strategy-reported mask plausibility (R3 only; see Sam3Roi._box_fill).  Logged on
    # every image so the R3 fallback threshold can be re-selected on validation
    # without re-running SAM 3.  Optional so shards written before it exist still load.
    box_fill: float = float("nan")

    @property
    def tooth_retention(self) -> float:
        return (
            self.retained_tooth_pixels / self.gt_tooth_pixels
            if self.gt_tooth_pixels
            else float("nan")
        )

    @property
    def tooth_retention_box(self) -> float:
        """Retention of the crop box alone, ignoring the mask.

        A strategy can be a good *crop* and a bad *clip*: the box may keep every
        tooth while the mask inside it discards some.  The full automatic pipeline
        clips predictions to the mask, so both numbers are needed to say which of
        the two costs recall.
        """
        return (
            self.retained_tooth_pixels_box / self.gt_tooth_pixels
            if self.gt_tooth_pixels
            else float("nan")
        )

    @property
    def area_retention(self) -> float:
        return self.roi_pixels / self.image_pixels if self.image_pixels else float("nan")

    @property
    def box_area_retention(self) -> float:
        return self.box_pixels / self.image_pixels if self.image_pixels else float("nan")

    def to_row(self) -> dict:
        row = asdict(self)
        row["tooth_retention"] = self.tooth_retention
        row["tooth_retention_box"] = self.tooth_retention_box
        row["area_retention"] = self.area_retention
        row["box_area_retention"] = self.box_area_retention
        return row


def evaluate_roi_image(
    image_id: str,
    patient_id: str,
    view_label: str,
    strategy: str,
    gt_raster: np.ndarray,
    roi_mask: np.ndarray,
    box: tuple[int, int, int, int],
    fallback: bool,
    runtime_s: float = 0.0,
    peak_gpu_mb: float = 0.0,
    box_fill: float = float("nan"),
) -> RoiImageEval:
    tooth = gt_raster > 0
    roi = np.asarray(roi_mask) > 0
    x0, y0, x1, y1 = box
    box_mask = np.zeros_like(tooth)
    box_mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return RoiImageEval(
        image_id=image_id,
        patient_id=patient_id,
        view_label=view_label,
        strategy=strategy,
        gt_tooth_pixels=int(tooth.sum()),
        retained_tooth_pixels=int((tooth & roi).sum()),
        retained_tooth_pixels_box=int((tooth & box_mask).sum()),
        image_pixels=int(tooth.size),
        roi_pixels=int(roi.sum()),
        box_pixels=int((x1 - x0 + 1) * (y1 - y0 + 1)),
        fallback=int(fallback),
        runtime_s=runtime_s,
        peak_gpu_mb=peak_gpu_mb,
        box_fill=float(box_fill),
    )


CATASTROPHIC_RETENTION = 0.90


def aggregate_roi(evals: list[RoiImageEval]) -> dict[str, float]:
    if not evals:
        return {}
    retention = np.asarray([e.tooth_retention for e in evals], dtype=float)
    retention_box = np.asarray([e.tooth_retention_box for e in evals], dtype=float)
    area = np.asarray([e.area_retention for e in evals], dtype=float)
    box_area = np.asarray([e.box_area_retention for e in evals], dtype=float)
    finite = retention[~np.isnan(retention)]
    total_tooth = sum(e.gt_tooth_pixels for e in evals)
    total_kept = sum(e.retained_tooth_pixels for e in evals)
    total_kept_box = sum(e.retained_tooth_pixels_box for e in evals)
    return {
        "n_images": float(len(evals)),
        "n_patients": float(len({e.patient_id for e in evals})),
        "tooth_pixel_retention_pooled": (total_kept / total_tooth) if total_tooth else float("nan"),
        "tooth_pixel_retention_mean": float(np.nanmean(retention)),
        "tooth_pixel_retention_min": float(np.nanmin(retention)) if finite.size else float("nan"),
        "tooth_pixel_retention_box_pooled": (
            (total_kept_box / total_tooth) if total_tooth else float("nan")
        ),
        "tooth_pixel_retention_box_mean": float(np.nanmean(retention_box)),
        "area_retention_mean": float(np.nanmean(area)),
        "box_area_retention_mean": float(np.nanmean(box_area)),
        "frac_images_retain_95": float((finite >= 0.95).mean()) if finite.size else float("nan"),
        "frac_images_retain_98": float((finite >= 0.98).mean()) if finite.size else float("nan"),
        "frac_images_retain_99": float((finite >= 0.99).mean()) if finite.size else float("nan"),
        "catastrophic_failure_rate": (
            float((finite < CATASTROPHIC_RETENTION).mean()) if finite.size else float("nan")
        ),
        "n_catastrophic": float((finite < CATASTROPHIC_RETENTION).sum()) if finite.size else 0.0,
        "fallback_rate": float(np.mean([e.fallback for e in evals])),
        "mean_runtime_s": float(np.mean([e.runtime_s for e in evals])),
        "peak_gpu_mb": float(max((e.peak_gpu_mb for e in evals), default=0.0)),
    }


def aggregate_roi_by_view(evals: list[RoiImageEval]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[RoiImageEval]] = {}
    for ev in evals:
        groups.setdefault(ev.view_label, []).append(ev)
    return {view: aggregate_roi(items) for view, items in sorted(groups.items())}
