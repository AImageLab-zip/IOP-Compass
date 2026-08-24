#!/usr/bin/env python
"""Build the canonical dataset manifest and the ground-truth instance rasters.

    python scripts/build_manifest.py --config configs/dataset_final_1000.yaml
    python scripts/build_manifest.py --config ... --shard 0 --num-shards 16 --rasters-only

Sharding exists so the raster pass can run as a SLURM CPU array; the manifest
itself is always written by the unsharded invocation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.data.adapter import discover_patients, load_dataset_config  # noqa: E402
from iop_compass.data.manifest import build_rows, write_manifest  # noqa: E402
from iop_compass.data.rasterize import rasterize_record  # noqa: E402
from iop_compass.data.splits import load_splits  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", default=None, help="splits.json to stamp into the manifest")
    ap.add_argument("--out", default=None, help="manifest path (default: derived_root/manifest.csv)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--rasters-only", action="store_true")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--no-checksums", action="store_true")
    ap.add_argument(
        "--reduce-geometry",
        action="store_true",
        help="merge the per-shard geometry reports into unusable_annotations.json",
    )
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    if args.reduce_geometry:
        shard_dir = cfg.derived_root / "geometry_shards"
        merged: dict[str, str] = {}
        for path in sorted(shard_dir.glob("shard_*.json")):
            merged.update(json.loads(path.read_text()))
        if not merged:
            # Fall back to scanning the reference sidecars, which record the same
            # fact; this also lets the check be re-derived from existing rasters.
            for sidecar in sorted((cfg.derived_root / "gt_instances").glob("*_inst.json")):
                payload = json.loads(sidecar.read_text())
                if payload.get("n_instances", 0) == 0:
                    merged[payload["image_id"]] = (
                        "no_polygon_geometry:"
                        f"{payload.get('n_dropped_degenerate', 0)}_records_without_contours"
                    )
        out = cfg.derived_root / "unusable_annotations.json"
        out.write_text(json.dumps(merged, indent=1, sort_keys=True))
        print(f"[build_manifest] {len(merged)} image(s) without usable geometry -> {out}")
        for image_id, reason in sorted(merged.items())[:50]:
            print(f"[build_manifest]   {image_id}: {reason}")
        return 0
    print(f"[build_manifest] root={cfg.root}", flush=True)
    patients = discover_patients(cfg)
    print(f"[build_manifest] {len(patients)} patient directories", flush=True)

    derived = cfg.derived_root
    raster_dir = derived / "gt_instances"

    if not args.manifest_only:
        todo = [p for i, p in enumerate(patients) if i % args.num_shards == args.shard]
        n_done = n_dropped = 0
        overlap_total = 0
        # Some annotation files carry FDI codes and bounding boxes but no polygon
        # at all (they are raw pipeline output that never went through the
        # correction interface).  Those images have no usable reference mask, so
        # they are recorded here and their patients are later excluded.
        unusable: dict[str, str] = {}
        for patient in todo:
            for view in cfg.views_required:
                rec = patient.images.get(view)
                if rec is None or not rec.is_complete or rec.width is None:
                    continue
                res = rasterize_record(
                    rec,
                    raster_dir,
                    drop_zero_area=bool(cfg.audit.get("drop_zero_area_instances", True)),
                    eval_long_side=cfg.eval_long_side,
                )
                n_done += 1
                n_dropped += res.n_dropped
                overlap_total += res.overlap_pixels
                if res.n_instances == 0:
                    unusable[rec.image_id] = (
                        f"no_polygon_geometry:{len(rec.instances)}_records_"
                        f"{res.n_dropped}_without_contours"
                    )
        shard_dir = derived / "geometry_shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        (shard_dir / f"shard_{args.shard}_of_{args.num_shards}.json").write_text(
            json.dumps(unusable, indent=1, sort_keys=True)
        )
        print(
            f"[build_manifest] shard {args.shard}/{args.num_shards}: "
            f"{n_done} rasters, {n_dropped} degenerate instances dropped, "
            f"{overlap_total} overlapping pixels, "
            f"{len(unusable)} images without usable geometry",
            flush=True,
        )

    if args.rasters_only:
        return 0

    splits = load_splits(Path(args.splits)) if args.splits else None
    checksums_needed = not args.no_checksums
    rows = build_rows(patients, cfg, checksums=checksums_needed, splits=splits)

    out = Path(args.out) if args.out else derived / "manifest.csv"
    digest = write_manifest(rows, out)
    meta = {
        "manifest_path": str(out),
        "manifest_hash": digest,
        "n_rows": len(rows),
        "config": str(cfg.config_path),
        "dataset_root": str(cfg.root),
    }
    (out.parent / "manifest_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[build_manifest] wrote {out} ({len(rows)} rows) sha256={digest[:16]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
