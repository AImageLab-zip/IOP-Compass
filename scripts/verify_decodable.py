#!/usr/bin/env python
"""Verify that every source image decodes, not merely that its header parses.

    python scripts/verify_decodable.py --config configs/dataset_final_1000.yaml \
        --shard 0 --num-shards 16
    python scripts/verify_decodable.py --config ... --reduce

A truncated PNG still has a valid IHDR, so a header-only check reports the right
dimensions for a file that cannot be read.  This pass attempts a full decode and
writes ``<derived>/undecodable_images.json``, which
``iop_compass.data.splits.eligibility`` consults so an unreadable image can never
silently enter a split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.data.adapter import discover_patients, load_dataset_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--reduce", action="store_true")
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    shard_dir = cfg.derived_root / "decode_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.derived_root / "undecodable_images.json"

    if args.reduce:
        bad: dict[str, str] = {}
        for path in sorted(shard_dir.glob("*.json")):
            bad.update(json.loads(path.read_text()))
        out_path.write_text(json.dumps(bad, indent=1, sort_keys=True))
        print(f"[verify] {len(bad)} undecodable image(s) -> {out_path}")
        for image_id, reason in sorted(bad.items()):
            print(f"[verify]   {image_id}: {reason}")
        return 0

    patients = discover_patients(cfg, load_annotations=False)
    todo = [p for i, p in enumerate(patients) if i % args.num_shards == args.shard]
    bad: dict[str, str] = {}
    checked = 0
    for patient in todo:
        for view in cfg.views_required:
            rec = patient.images.get(view)
            if rec is None or rec.image_path is None:
                continue
            checked += 1
            img = cv2.imread(str(rec.image_path), cv2.IMREAD_COLOR)
            if img is None:
                bad[rec.image_id] = f"decode_failed:{cfg.rel(rec.image_path)}"
                continue
            if rec.width and rec.height and img.shape[:2] != (rec.height, rec.width):
                bad[rec.image_id] = (
                    f"header_shape_mismatch:{img.shape[1]}x{img.shape[0]}"
                    f"!={rec.width}x{rec.height}"
                )
    path = shard_dir / f"shard_{args.shard}_of_{args.num_shards}.json"
    path.write_text(json.dumps(bad, indent=1, sort_keys=True))
    print(
        f"[verify] shard {args.shard}/{args.num_shards}: checked={checked} "
        f"undecodable={len(bad)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
