#!/usr/bin/env python
"""Validation ablation of the individual post-processing rules.

    python scripts/ablate_postprocessing.py --config configs/dataset_final_1000.yaml \
        --split val
    python scripts/ablate_postprocessing.py --config ... --split val --replicate 1

Re-runs post-processing over the *stored* un-post-processed SegmentAnyTooth masks of
one grid cell with a single rule group switched off at a time, so each rule earns its
place from validation data alone.  No model inference is repeated, and the test set is
never touched.

``--replicate`` selects both the hold-out block and the stored predictions to read,
since each replicate has its own prediction directory and its own validation patients.

Scope: the ROI here is the recorded crop *box*, not the ROI mask, so this measures
the rules themselves under a benign region.  The separate effect of clipping to a
concept-prompted mask is the post axis of the grid, which keeps the two mechanisms
from being confounded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.data.adapter import load_dataset_config  # noqa: E402
from iop_compass.data.manifest import read_manifest  # noqa: E402
from iop_compass.data.rasterize import (  # noqa: E402
    load_instance_raster,
    load_instance_sidecar,
)
from iop_compass.data.splits import apply_seed_split, split_hash_for  # noqa: E402
from iop_compass.segmentation import grid  # noqa: E402
from iop_compass.segmentation.infer import load_prediction  # noqa: E402
from iop_compass.segmentation.metrics import aggregate, evaluate_image  # noqa: E402
from iop_compass.segmentation.postprocessing import (  # noqa: E402
    ABLATABLE_RULES,
    PostProcessParams,
    postprocess_instances,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument(
        "--source-cell",
        default="sam3+sat+no-post",
        help="grid cell holding the pre-post-processing masks",
    )
    ap.add_argument(
        "--source-variant",
        default=None,
        help="read this prediction directory verbatim instead of deriving it from "
        "--source-cell and --replicate",
    )
    ap.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="replicate whose split and stored predictions to ablate",
    )
    ap.add_argument("--post-config", default="configs/segmentation/postprocessing.yaml")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    base = PostProcessParams.from_yaml(args.post_config)

    rows = [
        r
        for r in apply_seed_split(
            read_manifest(cfg.derived_root / "manifest.csv"),
            cfg.derived_root,
            args.replicate,
        )
        if r["split"] == args.split and r["annotation_status"] == "annotated"
    ]
    if args.limit:
        rows = rows[: args.limit]

    pred_root = REPO / "runs" / cfg.name / "predictions" / args.split
    if args.source_variant:
        source = args.source_variant
    else:
        source = grid.prediction_dir_name(args.source_cell, args.replicate)
    pred_dir = pred_root / source
    raster_dir = cfg.derived_root / "gt_instances"
    if not pred_dir.exists():
        print(f"[ablate] missing {pred_dir}", file=sys.stderr)
        return 1

    configurations: dict[str, PostProcessParams] = {
        "none": base.without(*ABLATABLE_RULES),
        "full": base,
    }
    for rule in ABLATABLE_RULES:
        configurations[f"minus_{rule}"] = base.without(rule)
    # explicit both-ways row for the rule that carries the effect
    configurations["rules_only_no_duplicate_fdi"] = base.without("duplicate_fdi")

    per_config: dict[str, list] = {name: [] for name in configurations}
    n_images = 0
    for row in rows:
        pred_json = pred_dir / f"{row['image_id']}_pred.json"
        raster_path = raster_dir / row["mask_path"]
        sidecar_path = raster_dir / row["mask_path"].replace(".png", ".json")
        if not (pred_json.exists() and raster_path.exists() and sidecar_path.exists()):
            continue

        gt_raster = load_instance_raster(raster_path)
        sidecar = load_instance_sidecar(sidecar_path)
        gt_masks, gt_fdis = [], []
        for inst in sidecar["instances"]:
            mask = gt_raster == int(inst["instance_id"])
            if mask.any():
                gt_masks.append(mask)
                gt_fdis.append(inst.get("fdi"))

        prediction, raster = load_prediction(pred_json)
        masks, fdis, scores = [], [], []
        for inst in prediction["instances"]:
            mask = raster == int(inst["instance_id"])
            if mask.any():
                masks.append(mask)
                fdis.append(inst.get("fdi"))
                scores.append(float(inst.get("score", 1.0)))

        # Per-pixel ROI masks are not stored; the prediction record carries the ROI
        # box already expressed at the evaluation resolution, which is what the
        # clipping rules need.
        h, w = raster.shape
        roi = np.zeros((h, w), dtype=np.uint8)
        box = prediction.get("roi", {}).get("box_eval")
        if box:
            x0, y0, x1, y1 = (int(v) for v in box)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, max(x0 + 1, x1)), min(h, max(y0 + 1, y1))
            roi[y0:y1, x0:x1] = 1
        else:
            roi[:] = 1
        exclusion = (1 - roi).astype(np.uint8)

        for name, params in configurations.items():
            out_masks, out_fdis, out_scores, _ = postprocess_instances(
                masks, fdis, scores, roi, exclusion, params, row["view_label"]
            )
            ev, _ = evaluate_image(
                row["image_id"],
                row["patient_id"],
                row["view_label"],
                gt_masks,
                gt_fdis,
                out_masks,
                out_fdis,
                out_scores,
                compute_boundary=False,
            )
            per_config[name].append(ev)
        n_images += 1

    if not n_images:
        print("[ablate] no images evaluated", file=sys.stderr)
        return 1

    results = {name: aggregate(evs) for name, evs in per_config.items() if evs}
    full = results.get("full", {})
    out_dir = REPO / "results" / cfg.name / "segmentation" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_json = json.dumps(
        {
            "run_id": cfg.name,
            "split": args.split,
            "replicate": args.replicate,
            "split_hash": split_hash_for(cfg.derived_root, args.replicate),
            "source_variant": source,
            "n_images": n_images,
            "base_params": base.to_dict(),
            "results": results,
        },
        indent=1,
    )
    names = [f"postprocessing_ablation_r{args.replicate}.json"]
    if args.replicate == 1:
        # The unsuffixed name is the replicate-1 artefact the report and the
        # aggregation read.
        names.append("postprocessing_ablation.json")
    for name in names:
        (out_dir / name).write_text(payload_json)

    print(f"[ablate] {n_images} images, source={source}, replicate={args.replicate}")
    header = f"{'configuration':<34}{'FDI-F1':>9}{'dFDI-F1':>10}{'FP':>7}{'FN':>7}{'FDIerr':>8}"
    print(header)
    for name in ["none", "full"] + [
        n for n in results if n not in ("none", "full")
    ]:
        if name not in results:
            continue
        row = results[name]
        delta = row["fdi_instance_f1"] - full.get("fdi_instance_f1", float("nan"))
        print(
            f"{name:<34}{row['fdi_instance_f1']:>9.4f}{delta:>+10.4f}"
            f"{int(row['n_false_positive']):>7}{int(row['n_false_negative']):>7}"
            f"{int(row['n_fdi_errors']):>8}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
