#!/usr/bin/env python
"""Evaluate the ROI strategies on their own terms (Experiment B).

    python scripts/evaluate_roi.py --config configs/dataset_final_1000.yaml \
        --split val --shard 0 --num-shards 8
    python scripts/evaluate_roi.py --config ... --split val --reduce

Each shard writes per-image records; ``--reduce`` merges them into
``results/<dataset>/roi/<split>/summary_r<replicate>.json``.  This deliberately does
not look at downstream segmentation quality: a crop that improves Dice while
discarding tooth pixels is a bad ROI, and that only shows up here.

Every artefact carries the replicate it was produced under, because a replicate
rotates the hold-out *and* selects that replicate's ROI prior and detector, so two
replicates share neither the patients nor the strategies being measured.  Replicate 1
additionally writes the unsuffixed ``summary.json`` / ``per_image.csv``, which the
aggregation and the figures read.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party"))

from iop_compass.data.adapter import load_dataset_config  # noqa: E402
from iop_compass.data.imaging import load_image_bgr  # noqa: E402
from iop_compass.data.manifest import read_manifest  # noqa: E402
from iop_compass.data.rasterize import load_instance_raster  # noqa: E402
from iop_compass.data.splits import apply_seed_split, split_hash_for  # noqa: E402
from iop_compass.roi.evaluate import (  # noqa: E402
    RoiImageEval,
    aggregate_roi,
    aggregate_roi_by_view,
    evaluate_roi_image,
)
from iop_compass.roi.factory import build_strategies  # noqa: E402

STRATEGY_LABELS = {
    "R0_full": "R0 full image",
    "R1_geometric": "R1 view-specific geometric prior",
    "R2_learned": "R2 learned single-box ROI",
    "R3_sam3": "R3 SAM 3 concept-prompted ROI",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--strategies", default="R0_full,R1_geometric,R2_learned,R3_sam3")
    ap.add_argument("--roi-config", default="configs/roi/learned_roi.yaml")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reduce", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="replicate whose split and ROI artefacts to evaluate",
    )
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    out_dir = REPO / "results" / cfg.name / "roi" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    if args.reduce:
        return reduce_shards(cfg, out_dir, shard_dir, args.split, args.replicate)

    strategies = build_strategies(
        cfg,
        [s.strip() for s in args.strategies.split(",") if s.strip()],
        args.roi_config,
        args.device,
        replicate=args.replicate,
        split_hash=split_hash_for(cfg.derived_root, args.replicate),
    )

    rows = [
        r
        for r in apply_seed_split(
            read_manifest(cfg.derived_root / "manifest.csv"),
            cfg.derived_root,
            args.replicate,
        )
        if r["split"] == args.split and r["annotation_status"] == "annotated"
    ]
    patients = sorted({r["patient_id"] for r in rows})
    mine = {p for i, p in enumerate(patients) if i % args.num_shards == args.shard}
    rows = [r for r in rows if r["patient_id"] in mine]
    if args.limit:
        rows = rows[: args.limit]

    raster_dir = cfg.derived_root / "gt_instances"
    records: list[RoiImageEval] = []
    print(f"[roi_eval] split={args.split} images={len(rows)} strategies={list(strategies)}", flush=True)

    for row in rows:
        raster_path = raster_dir / row["mask_path"]
        if not raster_path.exists():
            continue
        gt = load_instance_raster(raster_path)
        image = load_image_bgr(cfg.root / row["image_path"], cfg.inference_long_side)
        h, w = image.shape[:2]
        # the reference raster defines the evaluation grid; recomputing it from the
        # downscaled image rounds twice and can be one pixel off
        eval_h, eval_w = gt.shape[:2]
        for name, strategy in strategies.items():
            roi = strategy(image, row["view_label"])
            # compare at the evaluation resolution, where the reference lives
            import cv2

            roi_small = (
                cv2.resize(
                    roi.roi_mask.astype("uint8"),
                    (eval_w, eval_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )
            sx, sy = eval_w / w, eval_h / h
            x0, y0, x1, y1 = roi.box
            box_small = (
                int(x0 * sx),
                int(y0 * sy),
                max(int(x0 * sx) + 1, int(x1 * sx)),
                max(int(y0 * sy) + 1, int(y1 * sy)),
            )
            records.append(
                evaluate_roi_image(
                    row["image_id"],
                    row["patient_id"],
                    row["view_label"],
                    name,
                    gt,
                    roi_small,
                    (box_small[0], box_small[1], box_small[2] - 1, box_small[3] - 1),
                    roi.fallback,
                    roi.runtime_s,
                    (
                        torch.cuda.max_memory_allocated() / 1e6
                        if torch.cuda.is_available()
                        else 0.0
                    ),
                    float(roi.extra.get("box_fill", float("nan"))),
                )
            )

    path = shard_dir / f"shard_{args.shard}_of_{args.num_shards}_r{args.replicate}.json"
    path.write_text(json.dumps([asdict(r) for r in records], indent=1))
    print(f"[roi_eval] wrote {len(records)} records -> {path}", flush=True)
    return 0


def reduce_shards(cfg, out_dir: Path, shard_dir: Path, split: str, replicate: int) -> int:
    records: list[RoiImageEval] = []
    # Only this replicate's shards: another replicate measured different strategies on
    # different patients, so pooling them would silently average across partitions.
    for path in sorted(shard_dir.glob(f"shard_*_r{replicate}.json")):
        for row in json.loads(path.read_text()):
            records.append(RoiImageEval(**row))
    if not records:
        print(
            f"[roi_eval] no shard records found for replicate {replicate}",
            file=sys.stderr,
        )
        return 1

    by_strategy: dict[str, list[RoiImageEval]] = {}
    for record in records:
        by_strategy.setdefault(record.strategy, []).append(record)

    summary = {
        "run_id": cfg.name,
        "split": split,
        "replicate": replicate,
        "split_hash": split_hash_for(cfg.derived_root, replicate),
        "eval_long_side": cfg.eval_long_side,
        "n_images": len({r.image_id for r in records}),
        "n_patients": len({r.patient_id for r in records}),
        "strategies": {},
    }
    for name, items in sorted(by_strategy.items()):
        summary["strategies"][name] = {
            "label": STRATEGY_LABELS.get(name, name),
            "overall": aggregate_roi(items),
            "by_view": aggregate_roi_by_view(items),
        }
        overall = summary["strategies"][name]["overall"]
        print(
            f"[roi_eval] {name:14s} tooth_retention={overall['tooth_pixel_retention_pooled']:.5f} "
            f"area={overall['area_retention_mean']:.3f} "
            f">=99%={overall['frac_images_retain_99']:.3f} "
            f"catastrophic={overall['catastrophic_failure_rate']:.4f} "
            f"fallback={overall['fallback_rate']:.4f} "
            f"t={overall['mean_runtime_s']:.2f}s",
            flush=True,
        )

    names = [f"summary_r{replicate}.json"]
    rows_names = [f"per_image_r{replicate}.csv"]
    if replicate == 1:
        # The unsuffixed names are the replicate-1 artefacts the aggregation and the
        # figures read.
        names.append("summary.json")
        rows_names.append("per_image.csv")

    payload = json.dumps(summary, indent=1)
    for name in names:
        (out_dir / name).write_text(payload)
    for name in rows_names:
        with (out_dir / name).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(records[0].to_row().keys()))
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_row())
    print(f"[roi_eval] wrote {out_dir / names[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
