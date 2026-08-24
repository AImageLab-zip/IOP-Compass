#!/usr/bin/env python
"""Evaluate stored segmentation predictions against the reference annotations.

    python scripts/evaluate_segmentation.py --config configs/dataset_final_1000.yaml \
        --split val --all-variants
    python scripts/evaluate_segmentation.py --config ... --split val \
        --variants no-roi+sat+no-post+r1,sam3+sat+post+r1

One evaluator is used for every grid cell.  For each cell it writes
``results/<dataset>/segmentation/<split>/<cell>.json`` with the pooled metrics,
the per-view breakdown, patient-level bootstrap confidence intervals and the
paired comparison against the baseline, plus a per-image CSV so any number can be
audited.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.data.adapter import load_dataset_config  # noqa: E402
from iop_compass.data.manifest import manifest_hash, read_manifest  # noqa: E402
from iop_compass.data.rasterize import (  # noqa: E402
    load_instance_raster,
    load_instance_sidecar,
)
from iop_compass.data.splits import apply_seed_split, split_hash_for  # noqa: E402
from iop_compass.reporting.bootstrap import (  # noqa: E402
    bootstrap_metrics,
    is_inconclusive,
    paired_bootstrap,
)
from iop_compass.segmentation import grid  # noqa: E402
from iop_compass.segmentation.infer import load_prediction  # noqa: E402
from iop_compass.segmentation.metrics import (  # noqa: E402
    ImageEval,
    aggregate,
    aggregate_by_view,
    average_precision,
    evaluate_image,
)

BOOTSTRAP_METRICS = [
    "fdi_instance_f1",
    "instance_f1",
    "matched_dice",
    "unmatched_zero_dice",
    "fdi_accuracy_matched",
    "panoptic_quality",
    "complete_image_rate",
    "complete_patient_rate",
    "boundary_f",
]


def method_label(variant: str) -> str:
    """Human-readable label for a result file name."""
    parsed = grid.parse_result_name(variant)
    return grid.label(parsed[0]) if parsed is not None else variant


def load_reference(raster_dir: Path, row: dict):
    raster_path = raster_dir / row["mask_path"]
    sidecar_path = raster_dir / row["mask_path"].replace(".png", ".json")
    if not raster_path.exists() or not sidecar_path.exists():
        return None, None
    raster = load_instance_raster(raster_path)
    sidecar = load_instance_sidecar(sidecar_path)
    masks, fdis = [], []
    for inst in sidecar["instances"]:
        mask = raster == int(inst["instance_id"])
        if mask.any():
            masks.append(mask)
            fdis.append(inst.get("fdi"))
    return masks, fdis


def evaluate_variant(
    cfg, rows, variant_dir: Path, variant: str, bootstrap: int, seed: int
) -> tuple[list[ImageEval], list[dict]]:
    raster_dir = cfg.derived_root / "gt_instances"
    evals: list[ImageEval] = []
    missing: list[dict] = []
    for row in rows:
        pred_json = variant_dir / f"{row['image_id']}_pred.json"
        if not pred_json.exists():
            missing.append({"image_id": row["image_id"], "reason": "missing_prediction"})
            continue
        gt_masks, gt_fdis = load_reference(raster_dir, row)
        if gt_masks is None:
            missing.append({"image_id": row["image_id"], "reason": "missing_reference"})
            continue
        payload, raster = load_prediction(pred_json)
        pred_masks, pred_fdis, pred_scores = [], [], []
        for inst in payload["instances"]:
            mask = raster == int(inst["instance_id"])
            if not mask.any():
                continue
            pred_masks.append(mask)
            pred_fdis.append(inst.get("fdi"))
            pred_scores.append(float(inst.get("score", 1.0)))
        if gt_masks and pred_masks and gt_masks[0].shape != pred_masks[0].shape:
            missing.append({"image_id": row["image_id"], "reason": "shape_mismatch"})
            continue
        runtime = float(payload.get("runtime", {}).get("total_seconds", 0.0))
        peak = float(payload.get("meta", {}).get("peak_gpu_mb", 0.0))
        ev, _ = evaluate_image(
            row["image_id"],
            row["patient_id"],
            row["view_label"],
            gt_masks,
            gt_fdis,
            pred_masks,
            pred_fdis,
            pred_scores,
            runtime_s=runtime,
            peak_gpu_mb=peak,
        )
        evals.append(ev)
    return evals, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--variants", default=None)
    ap.add_argument("--all-variants", action="store_true")
    ap.add_argument(
        "--paired-baseline",
        default="no-roi+sat+no-post",
        help="cell every other cell is compared against, without the replicate "
        "suffix (default: the raw full-image SegmentAnyTooth cell)",
    )
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=12345)
    ap.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="replicate index; selects the rotating hold-out block to evaluate against",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    rows = [
        r
        for r in apply_seed_split(
            read_manifest(cfg.derived_root / "manifest.csv"),
            cfg.derived_root,
            args.replicate,
        )
        if r["split"] == args.split and r["annotation_status"] == "annotated"
    ]
    pred_root = REPO / "runs" / cfg.name / "predictions" / args.split
    if not pred_root.exists():
        print(f"[evaluate] no predictions at {pred_root}", file=sys.stderr)
        return 1

    if args.all_variants:
        # Only this replicate's cells: a cell carries the replicate it was produced
        # under, and scoring another replicate's predictions against this replicate's
        # split would silently evaluate the wrong patients.
        variants = []
        for d in sorted(pred_root.iterdir()):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            parsed = grid.parse_result_name(d.name)
            if parsed is None or parsed[1] != args.replicate:
                continue
            variants.append(d.name)
    else:
        variants = [v.strip() for v in (args.variants or "").split(",") if v.strip()]
    if not variants:
        print("[evaluate] no variants selected", file=sys.stderr)
        return 2

    out_dir = (
        Path(args.out)
        if args.out
        else REPO / "results" / cfg.name / "segmentation" / args.split
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    split_hash = split_hash_for(cfg.derived_root, args.replicate)
    m_hash = manifest_hash(cfg.derived_root / "manifest.csv")

    all_evals: dict[str, list[ImageEval]] = {}
    summary: dict[str, dict] = {}
    for variant in variants:
        evals, missing = evaluate_variant(
            cfg, rows, pred_root / variant, variant, args.bootstrap, args.bootstrap_seed
        )
        if not evals:
            print(f"[evaluate] {variant}: no evaluable images", file=sys.stderr)
            continue
        all_evals[variant] = evals
        overall = aggregate(evals)
        by_view = aggregate_by_view(evals)
        ci = bootstrap_metrics(
            evals, BOOTSTRAP_METRICS, args.bootstrap, args.bootstrap_seed
        )
        payload = {
            "run_id": cfg.name,
            "dataset_manifest_hash": m_hash,
            "split_hash": split_hash,
            "split": args.split,
            "replicate": args.replicate,
            "method": variant,
            "method_label": method_label(variant),
            "n_images": len(evals),
            "n_missing": len(missing),
            "missing": missing[:50],
            "overall": overall,
            "by_view": by_view,
            "bootstrap": {k: v.to_dict() for k, v in ci.items()},
            "mask_ap50": average_precision(evals),
            "eval_long_side": cfg.eval_long_side,
            "iou_threshold": 0.50,
        }
        (out_dir / f"{variant}.json").write_text(json.dumps(payload, indent=1))
        summary[variant] = payload

        with (out_dir / f"{variant}_per_image.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(evals[0].to_row().keys()))
            writer.writeheader()
            for ev in evals:
                writer.writerow(ev.to_row())

        print(
            f"[evaluate] {variant:32s} n={len(evals):4d} "
            f"FDI-F1={overall['fdi_instance_f1']:.4f} "
            f"inst-F1={overall['instance_f1']:.4f} "
            f"mDice={overall['matched_dice']:.4f} "
            f"complete-img={overall['complete_image_rate']:.4f} "
            f"FP={int(overall['n_false_positive'])} FN={int(overall['n_false_negative'])} "
            f"FDIerr={int(overall['n_fdi_errors'])}",
            flush=True,
        )

    # The baseline is named as a bare cell but stored per replicate.
    baseline = args.paired_baseline
    if baseline not in all_evals:
        suffixed = f"{baseline}{grid.SEP}r{args.replicate}"
        if suffixed in all_evals:
            baseline = suffixed
    if baseline in all_evals:
        for variant, evals in all_evals.items():
            if variant == baseline:
                continue
            paired = paired_bootstrap(
                all_evals[baseline],
                evals,
                BOOTSTRAP_METRICS,
                args.bootstrap,
                args.bootstrap_seed,
            )
            payload = summary[variant]
            payload["paired_vs_baseline"] = {
                "baseline": baseline,
                "metrics": {
                    k: {**v.to_dict(), "inconclusive": is_inconclusive(v)}
                    for k, v in paired.items()
                },
            }
            (out_dir / f"{variant}.json").write_text(json.dumps(payload, indent=1))
            delta = paired.get("fdi_instance_f1")
            if delta:
                verdict = "inconclusive" if is_inconclusive(delta) else "conclusive"
                print(
                    f"[evaluate] {variant:32s} vs {baseline}: "
                    f"dFDI-F1={delta.point:+.4f} "
                    f"[{delta.low:+.4f}, {delta.high:+.4f}] {verdict}",
                    flush=True,
                )

    (out_dir / "index.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "variants": sorted(summary),
                "baseline": baseline,
                "bootstrap_replicates": args.bootstrap,
                "bootstrap_seed": args.bootstrap_seed,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
