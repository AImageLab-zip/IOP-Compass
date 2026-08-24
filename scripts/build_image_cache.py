#!/usr/bin/env python
"""Pre-resize the source photographs into the caches used for training.

    python scripts/build_image_cache.py --config configs/dataset_final_1000.yaml \
        --kind classifier --shard 0 --num-shards 16
    python scripts/build_image_cache.py --config ... --kind detector

``classifier``
    square ``classifier_image_size`` x ``classifier_image_size`` PNG, matching the
    non-aspect-preserving resize used by the existing view classifier.
``detector``
    aspect-preserving PNG with the long side at ``eval_long_side``, used by the
    Mask R-CNN baseline and the learned ROI detector so their inputs line up
    pixel-for-pixel with the reference instance rasters.

Decoding the full-resolution PNGs (up to 6016x4016, ~18 MB each) costs seconds
per image, so this runs once as a CPU array job instead of inside every epoch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.data.adapter import discover_patients, load_dataset_config  # noqa: E402
from iop_compass.data.imaging import target_size  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--kind", choices=["classifier", "detector"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    patients = discover_patients(cfg, load_annotations=False)

    if args.kind == "classifier":
        out_dir = cfg.derived_root / "cache" / f"img_{cfg.classifier_image_size}"
    else:
        out_dir = cfg.derived_root / "cache" / f"img_long{cfg.eval_long_side}"
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [p for i, p in enumerate(patients) if i % args.num_shards == args.shard]
    written = skipped = failed = 0
    for patient in todo:
        for view in cfg.views_required:
            rec = patient.images.get(view)
            if rec is None or rec.image_path is None:
                continue
            dest = out_dir / f"{rec.image_id}.png"
            if dest.exists() and not args.overwrite:
                skipped += 1
                continue
            img = cv2.imread(str(rec.image_path), cv2.IMREAD_COLOR)
            if img is None:
                failed += 1
                print(f"[cache] unreadable: {rec.image_path}", flush=True)
                continue
            h, w = img.shape[:2]
            if args.kind == "classifier":
                size = cfg.classifier_image_size
                resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            else:
                tw, th = target_size(w, h, cfg.eval_long_side)
                resized = (
                    img
                    if (tw, th) == (w, h)
                    else cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
                )
            # cv2 picks the encoder from the extension, so the temporary file has
            # to keep a .png suffix.
            tmp = dest.with_name(dest.stem + ".tmp.png")
            if not cv2.imwrite(str(tmp), resized):
                failed += 1
                continue
            tmp.replace(dest)
            written += 1

    print(
        f"[cache] kind={args.kind} shard={args.shard}/{args.num_shards} "
        f"written={written} skipped={skipped} failed={failed} -> {out_dir}",
        flush=True,
    )
    # Unreadable images are recorded above, not fatal here: eligibility() excludes
    # the owning patient later (same convention as verify_decodable.py). Failing
    # this job would block every downstream SLURM stage over a data issue the
    # audit already handles.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
