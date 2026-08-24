#!/usr/bin/env python
"""Train and evaluate one view-classification variant / seed.

    python scripts/train_classifier.py --config configs/dataset_final_1000.yaml \
        --variant-config configs/classification/C1.yaml --seed 1
    python scripts/train_classifier.py --config ... --variant-config ... \
        --seed 1 --replicate 1

Writes into ``runs/<dataset>/classification/<variant>_seed<seed>/``:
``best.pt``, ``history.json``, ``val_predictions.npz``, ``metrics.json``.

``--replicate`` selects the rotating hold-out block to train against, so a campaign
run as ``--seed k --replicate k`` measures each variant on a *different* set of
validation patients and the spread over seeds becomes a spread over partitions
rather than over weight initialisations.  It defaults to 0, which is the canonical
partition, so single-replicate runs are unaffected.

The test split is not touched; ``scripts/run_final_test.py`` does that once, after
the configuration is frozen.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.classification.dataset import (  # noqa: E402
    VIEW_CLASSES,
    AugmentationConfig,
    samples_from_manifest,
)
from iop_compass.classification.evaluate import evaluate_predictions  # noqa: E402
from iop_compass.classification.train import (  # noqa: E402
    TrainConfig,
    evaluate_loader,
    train_classifier,
)
from iop_compass.data.adapter import load_dataset_config  # noqa: E402
from iop_compass.data.manifest import manifest_hash, read_manifest  # noqa: E402
from iop_compass.data.splits import apply_seed_split, split_hash_for  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant-config", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="rotating hold-out block to train against (default 1, the canonical "
        "partition)",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="1 epoch, tiny subset")
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    variant = yaml.safe_load(Path(args.variant_config).read_text())
    name = variant["variant"]

    manifest_path = cfg.derived_root / "manifest.csv"
    # The manifest carries whichever partition it was built against, so a rotated
    # hold-out has to re-derive membership or it would train on its own val patients.
    rows = apply_seed_split(
        read_manifest(manifest_path), cfg.derived_root, args.replicate
    )
    cache_dir = cfg.derived_root / "cache" / f"img_{cfg.classifier_image_size}"

    train_samples = samples_from_manifest(rows, "train", cache_dir)
    val_samples = samples_from_manifest(rows, "val", cache_dir)
    if args.smoke:
        train_samples = train_samples[:40]
        val_samples = val_samples[:20]

    missing = [s.image_path for s in train_samples + val_samples if not s.image_path.exists()]
    if missing:
        print(
            f"[classify] {len(missing)} cached images missing, e.g. {missing[0]}",
            file=sys.stderr,
        )
        return 1

    train_cfg = TrainConfig.from_dict(variant.get("train"))
    if args.smoke:
        train_cfg = TrainConfig.from_dict(
            {**variant.get("train", {}), "epochs": 1, "num_workers": 0}
        )
    aug_cfg = AugmentationConfig.from_dict(variant.get("augmentation"))
    pretrained = bool(variant.get("model", {}).get("pretrained", True))

    out_dir = (
        Path(args.out)
        if args.out
        else REPO / "runs" / cfg.name / "classification" / f"{name}_seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[classify] variant={name} seed={args.seed} replicate={args.replicate} "
        f"train_images={len(train_samples)} val_images={len(val_samples)} "
        f"pretrained={pretrained} augment={aug_cfg.enabled}",
        flush=True,
    )

    result = train_classifier(
        train_samples=train_samples,
        val_samples=val_samples,
        train_cfg=train_cfg,
        aug_cfg=aug_cfg,
        image_size=cfg.classifier_image_size,
        pretrained=pretrained,
        seed=args.seed,
        out_dir=out_dir,
        device=args.device,
    )

    # reload the selected checkpoint and dump validation predictions
    from torch.utils.data import DataLoader

    from iop_compass.classification.dataset import ViewDataset
    from iop_compass.classification.models import build_resnet18, load_checkpoint

    model = build_resnet18(len(VIEW_CLASSES), pretrained=False)
    load_checkpoint(model, result.checkpoint_path)
    model.to(args.device)
    val_loader = DataLoader(
        ViewDataset(val_samples, cfg.classifier_image_size, AugmentationConfig(enabled=False)),
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )
    val_out = evaluate_loader(model, val_loader, args.device, torch.nn.CrossEntropyLoss())
    np.savez_compressed(
        out_dir / "val_predictions.npz",
        probs=val_out.probs,
        labels=val_out.labels,
        image_ids=np.asarray(val_out.image_ids),
        patient_ids=np.asarray(val_out.patient_ids),
        classes=np.asarray(VIEW_CLASSES),
    )

    metrics = evaluate_predictions(
        val_out.probs, val_out.labels, list(val_out.patient_ids)
    )
    payload = {
        "run_id": cfg.name,
        "method": name,
        "seed": args.seed,
        "replicate": args.replicate,
        "split": "val",
        "dataset_manifest_hash": manifest_hash(manifest_path),
        "split_hash": split_hash_for(cfg.derived_root, args.replicate),
        "checkpoint": str(result.checkpoint_path),
        "checkpoint_hash": _sha256(result.checkpoint_path),
        "best_epoch": result.best_epoch,
        "best_val_macro_f1": result.best_val_macro_f1,
        "overall": metrics.to_dict(),
        "runtime": {
            "train_seconds": result.total_seconds,
            "peak_gpu_mb": result.peak_gpu_mb,
            "epochs_run": len(result.history),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "hostname": platform.node(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "config": {"dataset": str(args.config), "variant": str(args.variant_config)},
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=1))
    print(
        f"[classify] {name} seed={args.seed} val macroF1={metrics.macro_f1:.4f} "
        f"acc={metrics.accuracy:.4f} "
        f"patient_set_exact={metrics.patient_set_exact_constrained:.4f}",
        flush=True,
    )
    return 0


def _sha256(path: Path | None) -> str | None:
    if path is None or not Path(path).exists():
        return None
    from iop_compass.data.adapter import sha256_file

    return sha256_file(Path(path))


if __name__ == "__main__":
    raise SystemExit(main())
