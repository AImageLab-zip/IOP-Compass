"""Qualitative examples chosen by predefined criteria, not by eye.

The paper needs a figure showing the characteristic failure modes.  Picking the
prettiest overlay would be a form of cherry-picking, so the examples are selected
by an explicit rule per failure mode from the per-image metrics CSV, and the rule
is stored next to the figure.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

QUALITATIVE_CRITERIA = {
    "missed_tooth": "image with the largest number of unmatched reference instances",
    "false_positive": "image with the largest number of unmatched predicted instances",
    "wrong_fdi": "image with the largest number of FDI label errors",
    "merge_split": "image with the largest number of merge plus split events",
    "typical_success": "complete-image-correct case with the median matched Dice",
}


def select_examples(rows: list[dict]) -> dict[str, dict]:
    """Choose one image per failure mode from a per-image metrics CSV."""

    def as_float(row: dict, key: str) -> float:
        try:
            return float(row.get(key, 0) or 0)
        except ValueError:
            return 0.0

    def pick(key_fn, predicate=None) -> dict | None:
        candidates = [r for r in rows if predicate is None or predicate(r)]
        if not candidates:
            return None
        best = max(candidates, key=key_fn)
        return best if key_fn(best) > 0 else None

    selected: dict[str, dict] = {}

    missed = pick(lambda r: as_float(r, "n_gt") - as_float(r, "tp_inst"))
    if missed:
        selected["missed_tooth"] = {
            "image_id": missed["image_id"],
            "view_label": missed["view_label"],
            "value": as_float(missed, "n_gt") - as_float(missed, "tp_inst"),
        }

    extra = pick(lambda r: as_float(r, "n_pred") - as_float(r, "tp_inst"))
    if extra:
        selected["false_positive"] = {
            "image_id": extra["image_id"],
            "view_label": extra["view_label"],
            "value": as_float(extra, "n_pred") - as_float(extra, "tp_inst"),
        }

    wrong = pick(lambda r: as_float(r, "fdi_wrong"))
    if wrong:
        selected["wrong_fdi"] = {
            "image_id": wrong["image_id"],
            "view_label": wrong["view_label"],
            "value": as_float(wrong, "fdi_wrong"),
        }

    merged = pick(lambda r: as_float(r, "merges") + as_float(r, "splits"))
    if merged:
        selected["merge_split"] = {
            "image_id": merged["image_id"],
            "view_label": merged["view_label"],
            "value": as_float(merged, "merges") + as_float(merged, "splits"),
        }

    complete = [r for r in rows if as_float(r, "complete_image") >= 1.0]
    if complete:
        def dice(row: dict) -> float:
            matched = as_float(row, "tp_inst")
            return as_float(row, "dice_sum") / matched if matched else 0.0

        complete.sort(key=dice)
        median = complete[len(complete) // 2]
        selected["typical_success"] = {
            "image_id": median["image_id"],
            "view_label": median["view_label"],
            "value": dice(median),
        }
    return selected


def _colour_for(index: int) -> tuple[int, int, int]:
    # deterministic, readable instance colours
    palette = [
        (0, 114, 178),
        (213, 94, 0),
        (0, 158, 115),
        (204, 121, 167),
        (230, 159, 0),
        (86, 180, 233),
        (240, 228, 66),
        (120, 94, 240),
    ]
    return palette[index % len(palette)]


def overlay(image_bgr: np.ndarray, raster: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = image_bgr.copy()
    tint = np.zeros_like(out)
    for i, inst_id in enumerate(sorted(set(np.unique(raster).tolist()) - {0})):
        mask = raster == inst_id
        tint[mask] = _colour_for(i)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, _colour_for(i), 1)
    blended = cv2.addWeighted(out, 1 - alpha, tint, alpha, 0)
    return np.where(raster[..., None] > 0, blended, out)


FIVE_VIEW_ORDER = ["frontal", "left_buccal", "right_buccal", "upper_occlusal", "lower_occlusal"]


def five_view_showcase(cfg, out_dir: Path, patient_id: str = "P0001") -> list[Path]:
    """Render one patient's five views (raw photo + instance-mask overlay, no FDI
    text) as individual panels for the paper's Figure 1 grid, instead of a single
    hand-baked collage image."""
    from ..data.rasterize import load_instance_raster

    cache = cfg.derived_root / "cache" / f"img_long{cfg.eval_long_side}"
    raster_dir = cfg.derived_root / "gt_instances"
    panel_dir = out_dir / "five_views"
    panel_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for view in FIVE_VIEW_ORDER:
        image_id = f"{patient_id}_{view}"
        image_path = cache / f"{image_id}.png"
        raster_path = raster_dir / f"{image_id}_inst.png"
        if not (image_path.exists() and raster_path.exists()):
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        raster = load_instance_raster(raster_path)
        if image.shape[:2] != raster.shape[:2]:
            image = cv2.resize(
                image, (raster.shape[1], raster.shape[0]), interpolation=cv2.INTER_AREA
            )

        raw_path = panel_dir / f"{image_id}_raw.png"
        cv2.imwrite(str(raw_path), image)
        written.append(raw_path)

        blended = overlay(image, raster)
        overlay_path = panel_dir / f"{image_id}_overlay.png"
        cv2.imwrite(str(overlay_path), blended)
        written.append(overlay_path)
    return written


def render_panels(
    cfg,
    repo: Path,
    split: str,
    examples: dict[str, dict],
    baseline: str,
    variant: str,
    out_dir: Path,
    tile_height: int = 260,
) -> Path | None:
    """Render a reference / baseline / pipeline panel per selected example."""
    if not examples:
        return None
    from ..data.rasterize import load_instance_raster
    from ..segmentation.infer import load_prediction

    cache = cfg.derived_root / "cache" / f"img_long{cfg.eval_long_side}"
    raster_dir = cfg.derived_root / "gt_instances"
    pred_root = repo / "runs" / cfg.name / "predictions" / split

    rows: list[np.ndarray] = []
    labels: list[str] = []
    for mode, entry in examples.items():
        image_id = entry["image_id"]
        image_path = cache / f"{image_id}.png"
        gt_path = raster_dir / f"{image_id}_inst.png"
        if not image_path.exists() or not gt_path.exists():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        gt = load_instance_raster(gt_path)
        if image.shape[:2] != gt.shape[:2]:
            image = cv2.resize(image, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)

        tiles = [overlay(image, gt)]
        for name in (baseline, variant):
            pred_json = pred_root / name / f"{image_id}_pred.json"
            if pred_json.exists():
                _, raster = load_prediction(pred_json)
                if raster.shape != gt.shape:
                    raster = cv2.resize(
                        raster, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST
                    )
                tiles.append(overlay(image, raster))
            else:
                tiles.append(image.copy())

        scale = tile_height / tiles[0].shape[0]
        tiles = [
            cv2.resize(
                tile,
                (int(round(tile.shape[1] * scale)), tile_height),
                interpolation=cv2.INTER_AREA,
            )
            for tile in tiles
        ]
        rows.append(np.hstack(tiles))
        labels.append(f"{mode}: {image_id}")

    if not rows:
        return None
    width = max(r.shape[1] for r in rows)
    padded = [
        np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=255)
        for r in rows
    ]
    panel = np.vstack(padded)
    path = out_dir / "qualitative_panel.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel)
    (out_dir / "qualitative_panel_rows.txt").write_text("\n".join(labels) + "\n")
    return path
