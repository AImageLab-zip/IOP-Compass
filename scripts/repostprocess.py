#!/usr/bin/env python
"""Derive the post-processed grid cells from the stored raw masks.

    python scripts/repostprocess.py --config configs/dataset_final_1000.yaml \
        --split val --source geometric+sat+no-post --target geometric+sat+post
    python scripts/repostprocess.py --config ... --split val --all-pairs

A ``no-post`` cell stores the raw segmenter output, so its ``post`` twin is a pure
function of that plus the frozen post-processing configuration.  Deriving it here
instead of re-running a model makes a change to that configuration cheap to apply.

**This is not how the benchmark grid produces its post cells, and it is not
equivalent.**  Stored predictions are at the *evaluation* resolution, so the rules run
there, while the frozen parameters were selected at the *inference* resolution -- a
3-pixel closing kernel covers twice the physical area at half the long side.  On this
cohort that is worth up to 0.003 FDI-F1.  ``run_segmentation.py`` therefore builds both
halves of a cell inline from one forward pass, at inference resolution, and this script
exists for re-deriving a post cell cheaply when its parameters change.  Treat its output
as an approximation of the inline result, not a substitute for it.

Post-processing clips against the ROI *mask*.  Raw cells written by
``run_segmentation.py`` carry that mask as a ``_roi.png`` sidecar, which is used when
present; otherwise the region falls back to the recorded crop box, which is only
equivalent for the box-shaped strategies.  Which of the two was used is recorded in
every derived prediction as ``roi_source``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.data.adapter import load_dataset_config  # noqa: E402
from iop_compass.data.manifest import read_manifest  # noqa: E402
from iop_compass.data.splits import apply_seed_split  # noqa: E402
from iop_compass.segmentation import grid  # noqa: E402
from iop_compass.segmentation.infer import (  # noqa: E402
    Prediction,
    load_prediction,
    load_roi_masks,
    save_prediction,
)
from iop_compass.segmentation.postprocessing import (  # noqa: E402
    PostProcessParams,
    postprocess_instances,
)


def region_for(record: dict, raster_shape, json_path: Path):
    """``(roi, exclusion, source)`` for one stored raw prediction."""
    sidecar = load_roi_masks(json_path)
    if sidecar is not None:
        return sidecar[0], sidecar[1], "roi_mask"

    h, w = raster_shape
    roi_level = (record.get("meta") or {}).get("roi_level")
    if roi_level == "no-roi":
        # No region to clip against; the rules run exactly as they do inline.
        return None, None, "none"
    roi = np.zeros((h, w), dtype=np.uint8)
    box = (record.get("roi") or {}).get("box_eval") or [0, 0, w, h]
    x0, y0, x1, y1 = (int(v) for v in box)
    roi[max(0, y0) : min(h, max(y0 + 1, y1)), max(0, x0) : min(w, max(x0 + 1, x1))] = 1
    return roi, (1 - roi).astype(np.uint8), "box_eval"


def repost_one(
    src_dir: Path, dst_dir: Path, rows: dict, params: PostProcessParams, target: str
) -> tuple[int, set[str]]:
    n = 0
    sources: set[str] = set()
    for js in sorted(src_dir.glob("*_pred.json")):
        record, raster = load_prediction(js)
        row = rows.get(record["image_id"])
        if row is None:
            continue
        masks, fdis, scores = [], [], []
        for inst in record["instances"]:
            mask = raster == int(inst["instance_id"])
            if mask.any():
                masks.append(mask)
                fdis.append(inst.get("fdi"))
                scores.append(float(inst.get("score", 1.0)))

        h, w = raster.shape
        roi, exclusion, roi_source = region_for(record, (h, w), js)
        sources.add(roi_source)

        out_masks, out_fdis, out_scores, stats = postprocess_instances(
            masks, fdis, scores, roi, exclusion, params, record["view_label"]
        )
        instances = [
            {
                "instance_id": i + 1,
                "fdi": fdi,
                "score": score,
                "bbox": [0, 0, 0, 0],
                "area": int(mask.sum()),
            }
            for i, (fdi, score, mask) in enumerate(zip(out_fdis, out_scores, out_masks))
        ]
        prediction = Prediction(
            image_id=record["image_id"],
            patient_id=record["patient_id"],
            view_label=record["view_label"],
            variant=target,
            width=w,
            height=h,
            instances=instances,
            roi=record.get("roi", {}),
            postprocess=stats.to_dict(),
            runtime=record.get("runtime", {}),
            meta={
                **record.get("meta", {}),
                "derived_from": src_dir.name,
                "roi_source": roi_source,
                "post_params": params.to_dict(),
            },
        )
        save_prediction(prediction, out_masks, dst_dir)
        n += 1
    return n, sources


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--source", default=None, help="cell holding the raw masks")
    ap.add_argument("--target", default=None, help="cell to write")
    ap.add_argument(
        "--all-pairs",
        action="store_true",
        help="derive every post cell of the grid from its no-post twin",
    )
    ap.add_argument("--post-config", default="configs/segmentation/postprocessing.yaml")
    ap.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="replicate index; must match the one the raw cells were run under",
    )
    ap.add_argument("--run", default=None)
    args = ap.parse_args()

    if not args.all_pairs and not (args.source and args.target):
        print("[repost] need --source and --target, or --all-pairs", file=sys.stderr)
        return 2

    cfg = load_dataset_config(args.config)
    params = PostProcessParams.from_yaml(args.post_config)

    run = args.run or cfg.name
    root = REPO / "runs" / run / "predictions" / args.split
    rows = {
        r["image_id"]: r
        for r in apply_seed_split(
            read_manifest(cfg.derived_root / "manifest.csv"),
            cfg.derived_root,
            args.replicate,
        )
        if r["split"] == args.split
    }

    if args.all_pairs:
        pairs = grid.post_pairs()
    else:
        pairs = [(args.source, args.target)]

    status = 0
    for source, target in pairs:
        src_dir = root / grid.prediction_dir_name(source, args.replicate)
        dst_dir = root / grid.prediction_dir_name(target, args.replicate)
        if not src_dir.exists():
            print(f"[repost] missing {src_dir}", file=sys.stderr)
            status = 1
            continue
        n, sources = repost_one(src_dir, dst_dir, rows, params, target)
        print(
            f"[repost] {src_dir.name} -> {dst_dir.name}: {n} images ({args.split}), "
            f"region from {sorted(sources) or ['-']}",
            flush=True,
        )
        if "box_eval" in sources and grid.parse_cell(source)[0] == "sam3":
            print(
                "[repost] WARNING: the concept-prompted cell was clipped against its "
                "box because no ROI sidecar was stored; re-run inference for this cell "
                "to make the post cell exact",
                file=sys.stderr,
            )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
