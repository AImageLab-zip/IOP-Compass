"""Label-wise post-processing of tooth instance predictions.

Ported from the upstream ``PostProcessingVerification`` rules with three changes,
all of them auditable:

1. It operates on **instances**, not on a ``uint8`` FDI raster.  The original
   code keyed everything by FDI value, so two detections sharing an FDI code
   were silently merged before post-processing could see them.
2. Every constant is a config field.  ``side_far_border_max_width_frac``,
   ``side_far_border_max_roi_frac`` and ``keep_only_best_component_per_label``
   were declared but never read upstream (``PostProcessingVerification.py:28-31``),
   while the far-side artefact test used the bare literals ``0.07``, ``0.008``
   and ``0.16`` (``:125-128``).  Those literals are now the defaults of the
   corresponding fields, so the shipped defaults reproduce upstream behaviour.
3. ``keep_only_best_component_per_label`` is implemented (it was a no-op upstream).
   The dataclass default is ``False`` so the shipped defaults reproduce upstream
   behaviour exactly; ``configs/segmentation/postprocessing.yaml`` turns it on.

Rule rationale
--------------
``roi_clip``            teeth cannot lie outside the intraoral ROI.
``close``               closes the thin gap left by specular highlights on enamel.
``fill_holes``          a tooth crown is simply connected in a photograph.
``bridge_close``        reconnects one crown split by a bracket or wire, accepted
                        only while the result stays anchored to the original mask.
``min_component_area``  removes single-pixel speckle from the mask decoder.
``roi_overlap``         drops components that only touch the ROI incidentally.
``exclusion_overlap``   drops components mostly on lips, gloves or retractors.
``seed_anchor``         forbids post-processing from inventing mask area far away
                        from the original detection.
``far_side_artifact``   in a buccal view the contralateral image border shows the
                        opposite arch out of focus; slivers there are artefacts.
``duplicate_fdi``       one FDI code names one tooth, so when cleaning splits an
                        instance the largest surviving component is the tooth and the
                        rest are fragments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

# Views on which the far-side artefact rule applies (buccal views only).
SIDE_VIEWS = ("left_buccal", "right_buccal", "left", "right")


@dataclass(frozen=True)
class PostProcessParams:
    # morphology on the individual instance mask
    tooth_close_k: int = 3
    tooth_open_k: int = 0
    fill_holes: bool = True

    # component filtering
    min_component_area: int = 10
    min_label_pixels: int = 6

    # ROI agreement
    min_roi_overlap_pixels: int = 6
    min_roi_overlap_frac: float = 0.03
    max_exclusion_overlap_frac: float = 0.45

    # gap bridging within one instance
    bridge_close_k: int = 3
    min_seed_overlap_pixels: int = 8
    min_seed_overlap_frac: float = 0.05
    keep_only_components_touching_original: bool = True

    # far-side artefact rule (buccal views)
    far_side_enabled: bool = True
    side_far_border_frac: float = 0.10
    side_far_max_width_frac: float = 0.07
    side_far_max_roi_frac: float = 0.008
    side_far_min_exclusion_frac: float = 0.16

    # duplicate-FDI resolution; upstream declared this but never applied it
    keep_only_best_component_per_label: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "PostProcessParams":
        payload = dict(payload or {})
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})

    @classmethod
    def from_yaml(cls, path) -> "PostProcessParams":
        """Load the frozen parameters from a config file's ``postprocessing`` block.

        A missing path gives the dataclass defaults, which reproduce the reference
        implementation's behaviour.
        """
        import yaml

        path = Path(path)
        if not path.is_file():
            return cls()
        payload = yaml.safe_load(path.read_text()) or {}
        return cls.from_dict(payload.get("postprocessing"))

    def without(self, *rules: str) -> "PostProcessParams":
        """Return a copy with the named rule groups disabled (for ablations)."""
        changes: dict = {}
        for rule in rules:
            if rule == "close":
                changes["tooth_close_k"] = 0
            elif rule == "fill_holes":
                changes["fill_holes"] = False
            elif rule == "bridge_close":
                changes["bridge_close_k"] = 0
            elif rule == "min_area":
                changes["min_component_area"] = 0
                changes["min_label_pixels"] = 0
            elif rule == "roi_overlap":
                changes["min_roi_overlap_pixels"] = 0
                changes["min_roi_overlap_frac"] = 0.0
            elif rule == "exclusion_overlap":
                changes["max_exclusion_overlap_frac"] = 1.0
            elif rule == "seed_anchor":
                changes["keep_only_components_touching_original"] = False
            elif rule == "far_side":
                changes["far_side_enabled"] = False
            elif rule == "duplicate_fdi":
                changes["keep_only_best_component_per_label"] = False
            else:
                raise KeyError(f"unknown post-processing rule: {rule}")
        return replace(self, **changes)


# `duplicate_fdi` is ablatable so one ablation run covers the whole rule set: without
# it the "no post-processing" row would silently keep duplicate-component resolution
# switched on, and the rule that matters most would not be measurable.
ABLATABLE_RULES = (
    "close",
    "fill_holes",
    "bridge_close",
    "min_area",
    "roi_overlap",
    "exclusion_overlap",
    "seed_anchor",
    "far_side",
    "duplicate_fdi",
)


# --------------------------------------------------------------------------- #
# primitives (behaviour-identical to the upstream helpers)
# --------------------------------------------------------------------------- #


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    h, w = mask_u8.shape
    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    out = cv2.bitwise_or(mask_u8, holes)
    return (out > 0).astype(np.uint8)


def _clean_piece(mask: np.ndarray, params: PostProcessParams) -> np.ndarray:
    out = (mask > 0).astype(np.uint8)
    if params.tooth_close_k > 1:
        ker = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (params.tooth_close_k, params.tooth_close_k)
        )
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ker)
    if params.tooth_open_k > 1:
        ker = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (params.tooth_open_k, params.tooth_open_k)
        )
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, ker)
    if params.fill_holes:
        out = fill_holes(out)
    return (out > 0).astype(np.uint8)


def _seed_overlap_ok(
    piece: np.ndarray, seed: np.ndarray, params: PostProcessParams
) -> bool:
    overlap = int(((piece > 0) & (seed > 0)).sum())
    area = int((piece > 0).sum())
    if overlap < params.min_seed_overlap_pixels:
        return False
    return overlap / max(area, 1) >= params.min_seed_overlap_frac


def _is_far_side_artifact(
    piece: np.ndarray,
    roi: np.ndarray,
    exclusion_overlap: int,
    area: int,
    view_label: str | None,
    params: PostProcessParams,
) -> bool:
    if not params.far_side_enabled or view_label not in SIDE_VIEWS:
        return False
    h, w = piece.shape
    ys, xs = np.where(piece > 0)
    if len(xs) == 0:
        return True
    x_min, x_max = int(xs.min()), int(xs.max())
    width_frac = (x_max - x_min + 1) / float(max(w, 1))
    border_w = int(params.side_far_border_frac * w)
    is_left = view_label in ("left_buccal", "left")
    far_side = x_min > (w - border_w) if is_left else x_max < border_w
    roi_area = int((roi > 0).sum())
    roi_side_frac = area / max(roi_area, 1)
    exclusion_frac = exclusion_overlap / max(area, 1)
    return bool(
        far_side
        and width_frac < params.side_far_max_width_frac
        and roi_side_frac < params.side_far_max_roi_frac
        and exclusion_frac > params.side_far_min_exclusion_frac
    )


# --------------------------------------------------------------------------- #
# instance-level entry point
# --------------------------------------------------------------------------- #


@dataclass
class PostProcessStats:
    n_in: int = 0
    n_out: int = 0
    dropped_small_seed: int = 0
    dropped_outside_roi: int = 0
    dropped_small_component: int = 0
    dropped_roi_overlap: int = 0
    dropped_exclusion: int = 0
    dropped_seed_anchor: int = 0
    dropped_far_side: int = 0
    dropped_duplicate_fdi: int = 0
    split_into_components: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def postprocess_instances(
    masks: list[np.ndarray],
    fdis: list[int | None],
    scores: list[float] | None,
    roi_mask: np.ndarray | None,
    exclusion_mask: np.ndarray | None,
    params: PostProcessParams,
    view_label: str | None = None,
) -> tuple[list[np.ndarray], list[int | None], list[float], PostProcessStats]:
    """Clean and filter predicted instances.

    Each instance is processed independently.  When cleaning breaks an instance
    into several connected components every surviving component is emitted as its
    own instance, which is what the upstream raster-based code did implicitly.
    """
    stats = PostProcessStats(n_in=len(masks))
    if not masks:
        return [], [], [], stats

    shape = masks[0].shape
    roi = (
        np.ones(shape, dtype=np.uint8)
        if roi_mask is None
        else (np.asarray(roi_mask) > 0).astype(np.uint8)
    )
    exclusion = (
        np.zeros(shape, dtype=np.uint8)
        if exclusion_mask is None
        else (np.asarray(exclusion_mask) > 0).astype(np.uint8)
    )
    scores = list(scores) if scores is not None else [1.0] * len(masks)

    out_masks: list[np.ndarray] = []
    out_fdis: list[int | None] = []
    out_scores: list[float] = []

    for mask, fdi, score in zip(masks, fdis, scores):
        seed = (np.asarray(mask) > 0).astype(np.uint8)
        if int(seed.sum()) < params.min_label_pixels:
            stats.dropped_small_seed += 1
            continue

        piece0 = ((seed > 0) & (roi > 0)).astype(np.uint8)
        if piece0.sum() == 0:
            stats.dropped_outside_roi += 1
            continue

        cleaned = _clean_piece(piece0, params)

        if params.bridge_close_k > 1:
            ker = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (params.bridge_close_k, params.bridge_close_k)
            )
            bridged = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, ker)
            if not params.keep_only_components_touching_original or _seed_overlap_ok(
                bridged, seed, params
            ):
                cleaned = bridged

        n_cc, cc, cc_stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
        kept_here = 0
        for i in range(1, n_cc):
            component = (cc == i).astype(np.uint8)
            area = int(cc_stats[i, cv2.CC_STAT_AREA])

            if area < params.min_component_area or area < params.min_label_pixels:
                stats.dropped_small_component += 1
                continue

            roi_overlap = int(((component > 0) & (roi > 0)).sum())
            if roi_overlap < params.min_roi_overlap_pixels:
                stats.dropped_roi_overlap += 1
                continue
            if roi_overlap / max(area, 1) < params.min_roi_overlap_frac:
                stats.dropped_roi_overlap += 1
                continue

            exclusion_overlap = int(((component > 0) & (exclusion > 0)).sum())
            if exclusion_overlap / max(area, 1) > params.max_exclusion_overlap_frac:
                stats.dropped_exclusion += 1
                continue

            if params.keep_only_components_touching_original and not _seed_overlap_ok(
                component, seed, params
            ):
                stats.dropped_seed_anchor += 1
                continue

            if _is_far_side_artifact(
                component, roi, exclusion_overlap, area, view_label, params
            ):
                stats.dropped_far_side += 1
                continue

            out_masks.append(component.astype(bool))
            out_fdis.append(fdi)
            out_scores.append(float(score))
            kept_here += 1

        if kept_here > 1:
            stats.split_into_components += kept_here - 1

    if params.keep_only_best_component_per_label:
        best: dict[int | None, int] = {}
        for idx, (mask, fdi) in enumerate(zip(out_masks, out_fdis)):
            if fdi is None:
                continue
            current = best.get(fdi)
            if current is None or int(mask.sum()) > int(out_masks[current].sum()):
                best[fdi] = idx
        keep_idx = {i for i, fdi in enumerate(out_fdis) if fdi is None}
        keep_idx |= set(best.values())
        stats.dropped_duplicate_fdi = len(out_masks) - len(keep_idx)
        order = sorted(keep_idx)
        out_masks = [out_masks[i] for i in order]
        out_fdis = [out_fdis[i] for i in order]
        out_scores = [out_scores[i] for i in order]

    stats.n_out = len(out_masks)
    return out_masks, out_fdis, out_scores, stats


def postprocess_raster(
    raw_mask: np.ndarray,
    roi_mask: np.ndarray,
    exclusion_mask: np.ndarray,
    params: PostProcessParams,
    view_label: str | None = None,
) -> np.ndarray:
    """FDI-raster entry point, kept for parity with the upstream function."""
    raw = np.asarray(raw_mask).astype(np.uint8)
    labels = [int(v) for v in np.unique(raw) if int(v) != 0]
    masks = [(raw == lab) for lab in labels]
    out_masks, out_fdis, _, _ = postprocess_instances(
        masks, labels, None, roi_mask, exclusion_mask, params, view_label
    )
    final = np.zeros_like(raw)
    for mask, fdi in zip(out_masks, out_fdis):
        if fdi is not None:
            final[mask] = fdi
    return final
