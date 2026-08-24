#!/usr/bin/env python
"""Evaluate trained view classifiers on a split, with bootstrap intervals.

    python scripts/evaluate_classifier.py --config configs/dataset_final_1000.yaml \
        --run final_1000 --split val
    python scripts/evaluate_classifier.py --config ... --run final_1000 --split test \
        --only-selected

For each variant/seed it writes per-image predictions and metrics, then a summary
with the mean and standard deviation over seeds and patient-level bootstrap
confidence intervals for the selected model.

``--replicates`` names the rotating hold-out blocks the campaign trained against.
Checkpoint ``<variant>_seed<k>`` is evaluated against replicate ``k``'s patients and
no others, because a replicate's validation patients are training patients for the
other replicates.  The default ``1`` reproduces the single-partition behaviour.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.classification.dataset import (  # noqa: E402
    VIEW_CLASSES,
    AugmentationConfig,
    ViewDataset,
    samples_from_manifest,
)
from iop_compass.classification.evaluate import (  # noqa: E402
    bootstrap_classification,
    evaluate_predictions,
)
from iop_compass.classification.models import (  # noqa: E402
    build_resnet18,
    load_checkpoint,
)
from iop_compass.classification.train import evaluate_loader  # noqa: E402
from iop_compass.data.adapter import load_dataset_config  # noqa: E402
from iop_compass.data.manifest import read_manifest  # noqa: E402
from iop_compass.data.splits import apply_seed_split, split_hash_for  # noqa: E402

PRINCIPAL_VARIANT = "C1"
SELECTION_TOLERANCE = 1e-6


def select_principal(run_dir: Path) -> tuple[str, int] | None:
    """Principal classifier, selected on validation only.

    Rule, fixed before the test set was touched:

    1. highest validation macro F1;
    2. ties broken in favour of the pre-registered principal configuration
       ``C1`` (ImageNet-pretrained with the clinically plausible augmentation);
    3. remaining ties broken by the lowest seed.

    A tie-break is needed because the view task saturates: several variants reach
    the same validation macro F1, and without a pre-specified rule the choice
    would fall to dictionary ordering.
    """
    candidates: list[tuple[float, float, str, int]] = []
    for path in sorted((run_dir / "classification").glob("*/metrics.json")):
        payload = json.loads(path.read_text())
        if payload.get("split") != "val":
            continue
        overall = payload.get("overall", {})
        candidates.append(
            (
                float(overall.get("macro_f1", 0.0)),
                float(overall.get("patient_set_exact_constrained", 0.0)),
                str(payload["method"]),
                int(payload["seed"]),
            )
        )
    if not candidates:
        return None
    best_f1 = max(c[0] for c in candidates)
    shortlist = [c for c in candidates if best_f1 - c[0] <= SELECTION_TOLERANCE]
    best_set = max(c[1] for c in shortlist)
    shortlist = [c for c in shortlist if best_set - c[1] <= SELECTION_TOLERANCE]
    shortlist.sort(key=lambda c: (c[2] != PRINCIPAL_VARIANT, c[2], c[3]))
    return shortlist[0][2], shortlist[0][3]


def replicate_of(checkpoint_name: str, replicates: tuple[int, ...]) -> int:
    """Which rotating hold-out block ``<variant>_seed<k>`` was trained against.

    A single-replicate campaign trains every seed on the same partition, so all
    checkpoints belong to that one replicate.  A multi-replicate campaign is
    submitted as ``--seed k --replicate k`` (the convention ``stage_maskrcnn`` already
    uses), so there the seed *is* the replicate.
    """
    if len(replicates) == 1:
        return replicates[0]
    seed = int(checkpoint_name.split("_seed")[1]) if "_seed" in checkpoint_name else 0
    if seed not in replicates:
        raise SystemExit(
            f"checkpoint {checkpoint_name} has seed {seed}, which is not one of the "
            f"campaign replicates {list(replicates)}; a multi-replicate campaign must "
            f"be trained with --seed k --replicate k"
        )
    return seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=12345)
    ap.add_argument("--only-selected", action="store_true")
    ap.add_argument(
        "--replicates",
        default="1",
        help="comma-separated rotating hold-out blocks the campaign trained against; "
        "checkpoint <variant>_seed<k> is scored against replicate k only",
    )
    args = ap.parse_args()
    replicates = tuple(
        int(part) for part in str(args.replicates).split(",") if part.strip()
    )

    cfg = load_dataset_config(args.config)
    run_dir = REPO / "runs" / args.run
    cache_dir = cfg.derived_root / "cache" / f"img_{cfg.classifier_image_size}"

    def samples_for(replicate: int):
        rows = apply_seed_split(
            read_manifest(cfg.derived_root / "manifest.csv"),
            cfg.derived_root,
            replicate,
        )
        return samples_from_manifest(rows, args.split, cache_dir)

    principal = select_principal(run_dir)
    checkpoints = sorted((run_dir / "classification").glob("*/best.pt"))
    if args.only_selected and principal:
        variant, seed = principal
        checkpoints = [run_dir / "classification" / f"{variant}_seed{seed}" / "best.pt"]

    out_dir = REPO / "results" / cfg.name / "classification" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Built lazily and reused: three replicates means three different sets of
    # validation patients, so one loader cannot serve every checkpoint.
    loaders: dict[int, DataLoader] = {}
    n_images: dict[int, int] = {}

    def loader_for(replicate: int) -> DataLoader | None:
        if replicate not in loaders:
            samples = samples_for(replicate)
            if not samples:
                return None
            n_images[replicate] = len(samples)
            loaders[replicate] = DataLoader(
                ViewDataset(
                    samples, cfg.classifier_image_size, AugmentationConfig(enabled=False)
                ),
                batch_size=32,
                shuffle=False,
                num_workers=2,
            )
        return loaders[replicate]

    if loader_for(replicates[0]) is None:
        print(f"[classify-eval] no samples for split={args.split}", file=sys.stderr)
        return 1

    per_model: dict[str, dict] = {}
    for ckpt in checkpoints:
        name = ckpt.parent.name
        replicate = replicate_of(name, replicates)
        loader = loader_for(replicate)
        if loader is None:
            print(
                f"[classify-eval] no {args.split} samples for replicate {replicate} "
                f"({name}), skipped",
                file=sys.stderr,
            )
            continue
        model = build_resnet18(len(VIEW_CLASSES), pretrained=False)
        load_checkpoint(model, ckpt)
        model.to(args.device)
        out = evaluate_loader(model, loader, args.device, torch.nn.CrossEntropyLoss())
        metrics = evaluate_predictions(out.probs, out.labels, list(out.patient_ids))
        np.savez_compressed(
            out_dir / f"{name}_predictions.npz",
            probs=out.probs,
            labels=out.labels,
            image_ids=np.asarray(out.image_ids),
            patient_ids=np.asarray(out.patient_ids),
            classes=np.asarray(VIEW_CLASSES),
        )
        payload = {
            "run_id": cfg.name,
            "method": name.split("_seed")[0],
            "seed": int(name.split("_seed")[1]) if "_seed" in name else 0,
            "replicate": replicate,
            "split": args.split,
            "split_hash": split_hash_for(cfg.derived_root, replicate),
            "n_images": n_images[replicate],
            "checkpoint": str(ckpt),
            "overall": metrics.to_dict(),
        }
        is_principal = principal is not None and name == f"{principal[0]}_seed{principal[1]}"
        if is_principal:
            payload["bootstrap"] = bootstrap_classification(
                out.probs,
                out.labels,
                list(out.patient_ids),
                n_replicates=args.bootstrap,
                seed=args.bootstrap_seed,
            )
            payload["is_principal"] = True
        (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=1))
        per_model[name] = payload
        print(
            f"[classify-eval] {name:12s} acc={metrics.accuracy:.4f} "
            f"macroF1={metrics.macro_f1:.4f} "
            f"patient_all_correct={metrics.patient_all_correct_independent:.4f} "
            f"patient_set_exact={metrics.patient_set_exact_constrained:.4f} "
            f"ECE={metrics.ece:.4f}"
            + ("  [principal]" if is_principal else ""),
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # mean / sd across seeds, per variant.  With more than one replicate each seed is a
    # different partition, so this spread is over partitions rather than over weight
    # initialisations; `spread_over` records which.
    by_variant: dict[str, list[dict]] = {}
    for name, payload in per_model.items():
        by_variant.setdefault(payload["method"], []).append(payload["overall"])
    summary = {
        "run_id": cfg.name,
        "split": args.split,
        "principal": (
            {"variant": principal[0], "seed": principal[1]} if principal else None
        ),
        "replicates": list(replicates),
        "spread_over": "replicates" if len(replicates) > 1 else "model_seeds",
        "n_images": n_images.get(replicates[0], 0),
        "n_images_by_replicate": {str(k): v for k, v in sorted(n_images.items())},
        "variants": {},
    }
    metric_names = [
        "accuracy",
        "macro_f1",
        "accuracy_constrained",
        "macro_f1_constrained",
        "patient_all_correct_independent",
        "patient_set_exact_constrained",
        "ece",
    ]
    for variant, records in sorted(by_variant.items()):
        entry: dict[str, object] = {"n_seeds": len(records)}
        for metric in metric_names:
            values = np.asarray([float(r[metric]) for r in records], dtype=float)
            entry[metric] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=0)),
                "values": [float(v) for v in values],
            }
        entry["per_view_recall_mean"] = {
            view: float(
                np.nanmean([float(r["per_view_recall"][view]) for r in records])
            )
            for view in VIEW_CLASSES
        }
        summary["variants"][variant] = entry

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[classify-eval] wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
