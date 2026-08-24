#!/usr/bin/env python
"""Fit the R1 view-specific geometric ROI prior on training patients only.

The prior is written per run *and* per replicate, to
``runs/<run>/roi/geometric_prior_r<replicate>.json``, and records the split hash it was
fitted on.  A single shared path under ``configs/`` cannot be correct: whichever cohort
fitted it last would win, and a replicate rotates the hold-out, so another replicate's
prior was fitted on patients that are validation or test patients here.  The factory
refuses a prior whose recorded split does not match the one being evaluated.

    python scripts/fit_roi_priors.py --config configs/dataset_final_1000.yaml \
        --replicate 1

The prior is the robust percentile envelope of the per-image tooth-union boxes,
expanded by a documented margin.  Validation and test patients are never read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.data.adapter import load_dataset_config  # noqa: E402
from iop_compass.data.manifest import read_manifest  # noqa: E402
from iop_compass.data.rasterize import load_instance_raster  # noqa: E402
from iop_compass.data.splits import apply_seed_split, split_hash_for  # noqa: E402
from iop_compass.roi.factory import geometric_prior_path  # noqa: E402
from iop_compass.roi.geometric_crop import (  # noqa: E402
    GeometricPriorConfig,
    fit_priors,
    normalised_tooth_box,
    save_priors,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--roi-config", default=None, help="YAML with a geometric_prior block")
    ap.add_argument(
        "--out",
        default=None,
        help="default: runs/<run>/roi/geometric_prior_r<replicate>.json -- per run and "
        "per replicate, because the prior is fitted on that replicate's train split",
    )
    ap.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="replicate whose train split the prior is fitted on",
    )
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    prior_cfg = GeometricPriorConfig()
    if args.roi_config:
        payload = yaml.safe_load(Path(args.roi_config).read_text()) or {}
        prior_cfg = GeometricPriorConfig.from_dict(payload.get("geometric_prior"))

    # The prior must be fitted on this replicate's train patients: a replicate rotates
    # the hold-out, so another replicate's train set contains patients that are
    # validation or test patients here.
    rows = apply_seed_split(
        read_manifest(cfg.derived_root / "manifest.csv"),
        cfg.derived_root,
        args.replicate,
    )
    split_hash = split_hash_for(cfg.derived_root, args.replicate)
    raster_dir = cfg.derived_root / "gt_instances"

    boxes: dict[str, list[tuple[float, float, float, float]]] = {}
    skipped = 0
    for row in rows:
        if row["split"] != args.split or row["annotation_status"] != "annotated":
            continue
        path = raster_dir / row["mask_path"]
        if not path.exists():
            skipped += 1
            continue
        raster = load_instance_raster(path)
        box = normalised_tooth_box(raster)
        if box is None:
            skipped += 1
            continue
        boxes.setdefault(row["view_label"], []).append(box)

    priors = fit_priors(boxes, prior_cfg)
    out_path = (
        Path(args.out)
        if args.out
        else geometric_prior_path(cfg, args.replicate)
    )
    n_patients = len(
        {r["patient_id"] for r in rows if r["split"] == args.split}
    )
    save_priors(
        priors,
        prior_cfg,
        out_path,
        provenance={
            "run_id": cfg.name,
            "replicate": args.replicate,
            "fitted_on_split": args.split,
            "split_hash": split_hash,
            "n_patients": n_patients,
        },
    )

    for view, prior in sorted(priors.items()):
        print(
            f"[roi_prior] {view:16s} n={prior.n_images:4d} "
            f"x[{prior.x0:.3f},{prior.x1:.3f}] y[{prior.y0:.3f},{prior.y1:.3f}] "
            f"area={((prior.x1 - prior.x0) * (prior.y1 - prior.y0)):.3f}",
            flush=True,
        )
    print(
        f"[roi_prior] fitted on split={args.split} replicate={args.replicate} "
        f"patients={n_patients} split_hash={split_hash[:12]} skipped={skipped} "
        f"-> {out_path}"
    )
    return 0 if priors else 1


if __name__ == "__main__":
    raise SystemExit(main())
