#!/usr/bin/env python
"""Run benchmark grid cells over one split and store the predictions.

    python scripts/run_segmentation.py --config configs/dataset_final_1000.yaml \
        --split val --cells geometric+sat+no-post,fast-r+sat+no-post --shard 0 --num-shards 8
    python scripts/run_segmentation.py --config ... --split val \
        --cells sam3+mask-rcnn+no-post --replicate 1

Cells are the ids defined in ``iop_compass.segmentation.grid``; ``--cells all`` runs
the whole grid and ``--cells raw`` only the ``no-post`` half.  Cells belonging to
different segmenters are dispatched to different backends automatically, so a single
invocation can mix them.

A cell's ``post`` half costs no extra forward pass -- it is post-processing applied to
the same masks -- so both halves are produced here, at the *inference* resolution the
frozen post-processing parameters were selected at.  ``scripts/repostprocess.py`` can
re-derive a post cell from stored rasters when those parameters change, but it works at
the evaluation resolution and is therefore an approximation of this.

Predictions land in ``runs/<dataset>/predictions/<split>/<cell>+r<replicate>/`` as an
instance-ID PNG plus a JSON record.  Every cell carries the replicate suffix: a
replicate rotates the validation/test hold-out, so the same cell under two replicates
is two different measurements.  Work is sharded by patient and is resumable: an image whose
prediction JSON already exists is skipped unless ``--overwrite``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party"))

from iop_compass.data.adapter import load_dataset_config  # noqa: E402
from iop_compass.data.imaging import load_image_bgr, target_size  # noqa: E402
from iop_compass.data.manifest import read_manifest  # noqa: E402
from iop_compass.data.splits import apply_seed_split, split_hash_for  # noqa: E402
from iop_compass.roi.factory import build_strategies  # noqa: E402
from iop_compass.segmentation import grid  # noqa: E402
from iop_compass.segmentation.infer import (  # noqa: E402
    infer_cells,
    maskrcnn_predict_fn,
    sat_predict_fn,
    save_prediction,
    save_roi_masks,
)
from iop_compass.segmentation.postprocessing import PostProcessParams  # noqa: E402
from iop_compass.segmentation.segmentanytooth_adapter import (  # noqa: E402
    SegmentAnyToothRunner,
    resolve_code_dir,
    resolve_weight_dir,
)


def rows_for_shard(rows, split: str, shard: int, num_shards: int):
    selected = [
        r for r in rows if r["split"] == split and r["annotation_status"] == "annotated"
    ]
    patients = sorted({r["patient_id"] for r in selected})
    mine = {p for i, p in enumerate(patients) if i % num_shards == shard}
    return [r for r in selected if r["patient_id"] in mine]


def parse_cells(spec: str) -> list[str]:
    if spec.strip() == "all":
        # The whole grid: a post cell costs no extra forward pass, and producing it here
        # keeps post-processing at the resolution its parameters were selected at.
        return grid.all_cells()
    if spec.strip() == "raw":
        return grid.raw_cells()
    cells = [c.strip() for c in spec.split(",") if c.strip()]
    for cell in cells:
        grid.parse_cell(cell)  # raises GridError on anything malformed
    return cells


def peak_gpu_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0


def run_backend(
    args,
    cfg,
    rows,
    out_root: Path,
    seg_level: str,
    cells: list[str],
    post_params: PostProcessParams,
) -> int:
    """Run every requested cell of one segmenter over the shard's images."""
    device = args.device

    # The split hash lets the factory refuse a prior or detector fitted on a
    # different replicate, whose train patients are hold-out patients here.
    roi_strategies = build_strategies(
        cfg,
        grid.strategies_for(cells),
        args.roi_config,
        device,
        replicate=args.replicate,
        split_hash=split_hash_for(cfg.derived_root, args.replicate),
    )
    extra_meta: dict = {}

    if seg_level == "sat":
        runner = SegmentAnyToothRunner(
            resolve_weight_dir(), resolve_code_dir(), device=device
        )
        predict_fn = sat_predict_fn(runner)
        extra_meta["sat_weight_sha256"] = runner.weight_checksums()
        # SegmentAnyTooth sees the native image downscaled to the inference long side.
        image_for = lambda row: load_image_bgr(  # noqa: E731
            cfg.root / row["image_path"], cfg.inference_long_side
        )
    else:
        from iop_compass.segmentation.mask_rcnn import MaskRcnnPredictor

        ckpt = (
            Path(args.checkpoint)
            if args.checkpoint
            else REPO / "runs" / cfg.name / "maskrcnn" / f"seed{args.replicate}" / "best.pt"
        )
        if not ckpt.exists():
            print(f"[segment] missing Mask R-CNN checkpoint {ckpt}", file=sys.stderr)
            return 1
        predictor = MaskRcnnPredictor(ckpt, device=device)
        predict_fn = maskrcnn_predict_fn(predictor)
        extra_meta.update(
            {
                "checkpoint": str(ckpt),
                "checkpoint_sha256": predictor.checkpoint_sha256(),
                "seed": args.replicate,
                "score_threshold": predictor.config.score_threshold,
            }
        )
        # Mask R-CNN was trained on the evaluation-resolution cache and keeps reading
        # it; the ROI box is applied relatively, so both backends crop the same region
        # of the scene at their own working resolution.
        import cv2

        cache = cfg.derived_root / "cache" / f"img_long{cfg.eval_long_side}"

        def image_for(row):
            path = cache / f"{row['image_id']}.png"
            if not path.exists():
                raise FileNotFoundError(f"missing cached image {path}")
            return cv2.imread(str(path), cv2.IMREAD_COLOR)

    todo = []
    for row in rows:
        missing = [
            cell
            for cell in cells
            if args.overwrite
            or not (
                out_root
                / grid.prediction_dir_name(cell, args.replicate)
                / f"{row['image_id']}_pred.json"
            ).exists()
        ]
        if missing:
            todo.append((row, missing))
    print(f"[segment] {seg_level}: {len(todo)} images need work", flush=True)

    n_images = 0
    started = time.perf_counter()
    for row, missing in todo:
        try:
            image = image_for(row)
        except FileNotFoundError as exc:
            print(f"[segment] {exc}", file=sys.stderr)
            continue
        results, rois = infer_cells(
            predict_fn=predict_fn,
            seg_level=seg_level,
            image_bgr=image,
            image_id=row["image_id"],
            patient_id=row["patient_id"],
            view_label=row["view_label"],
            cells=missing,
            roi_strategies=roi_strategies,
            post_params=post_params,
            eval_size=target_size(
                int(row["width"]), int(row["height"]), cfg.eval_long_side
            ),
            peak_gpu_mb=peak_gpu_mb(),
            extra_meta=extra_meta,
        )
        for cell, (prediction, masks) in results.items():
            prediction.meta["inference_long_side"] = (
                cfg.inference_long_side if seg_level == "sat" else cfg.eval_long_side
            )
            prediction.meta["eval_long_side"] = cfg.eval_long_side
            prediction.meta["source_size"] = [
                int(row["width"] or 0),
                int(row["height"] or 0),
            ]
            prediction.meta["replicate"] = args.replicate
            cell_dir = out_root / grid.prediction_dir_name(cell, args.replicate)
            save_prediction(prediction, masks, cell_dir)
            if cell in rois:
                save_roi_masks(
                    row["image_id"],
                    rois[cell],
                    (prediction.width, prediction.height),
                    cell_dir,
                )
        n_images += 1
        if n_images % 10 == 0:
            rate = (time.perf_counter() - started) / n_images
            print(
                f"[segment] {seg_level}: {n_images}/{len(todo)} images, "
                f"{rate:.1f} s/image",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    print(
        f"[segment] {seg_level}: done, {n_images} images in {elapsed / 60:.1f} min "
        f"({elapsed / max(n_images, 1):.1f} s/image)",
        flush=True,
    )

    frozen = {
        "segmenter": seg_level,
        "cells": cells,
        "replicate": args.replicate,
        "post_params": post_params.to_dict(),
        "inference_long_side": cfg.inference_long_side,
        "eval_long_side": cfg.eval_long_side,
        **extra_meta,
    }
    if "R3_sam3" in roi_strategies:
        frozen["sam3"] = roi_strategies["R3_sam3"].frozen_parameters()
    if "R2_learned" in roi_strategies:
        frozen["learned_roi_checkpoint_sha256"] = roi_strategies[
            "R2_learned"
        ].checkpoint_sha256()
    meta_dir = out_root / "_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / f"inference_{seg_level}_{args.shard}_of_{args.num_shards}.json").write_text(
        json.dumps(frozen, indent=1)
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument(
        "--cells",
        default="all",
        help="comma-separated grid cell ids, or 'all' for every cell needing a "
        "forward pass",
    )
    ap.add_argument("--roi-config", default="configs/roi/learned_roi.yaml")
    ap.add_argument("--post-config", default="configs/segmentation/postprocessing.yaml")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="replicate index: selects the rotating hold-out block "
        "(splits_seed<N>.json) and the Mask R-CNN checkpoint trained for it",
    )
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="stop after N images (smoke)")
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    cells = parse_cells(args.cells)
    rows = apply_seed_split(
        read_manifest(cfg.derived_root / "manifest.csv"),
        cfg.derived_root,
        args.replicate,
    )
    mine = rows_for_shard(rows, args.split, args.shard, args.num_shards)
    if args.limit:
        mine = mine[: args.limit]
    out_root = REPO / "runs" / cfg.name / "predictions" / args.split

    print(
        f"[segment] split={args.split} replicate={args.replicate} "
        f"shard={args.shard}/{args.num_shards} images={len(mine)} cells={cells}",
        flush=True,
    )

    post_params = PostProcessParams.from_yaml(args.post_config)
    status = 0
    for seg_level in grid.SEG_LEVELS:
        group = [c for c in cells if grid.parse_cell(c)[1] == seg_level]
        if not group:
            continue
        status |= run_backend(
            args, cfg, mine, out_root, seg_level, group, post_params
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
