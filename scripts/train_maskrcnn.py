#!/usr/bin/env python
"""Train the Mask R-CNN baseline on training patients only.

    python scripts/train_maskrcnn.py --config configs/dataset_final_1000.yaml \
        --model-config configs/segmentation/maskrcnn.yaml --seed 1

Checkpoint selection and the score-threshold operating point are both chosen on
validation FDI-aware instance F1.  The test split is never read.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.data.adapter import load_dataset_config, sha256_file  # noqa: E402
from iop_compass.data.manifest import read_manifest  # noqa: E402
from iop_compass.data.rasterize import (  # noqa: E402
    load_instance_raster,
    load_instance_sidecar,
)
from iop_compass.data.splits import apply_seed_split, split_hash_for  # noqa: E402
from iop_compass.segmentation.mask_rcnn import (  # noqa: E402
    MaskRcnnConfig,
    ToothInstanceDataset,
    build_maskrcnn,
    collate,
    mirror_fdi,
)
from iop_compass.segmentation.metrics import aggregate, evaluate_image  # noqa: E402


def collect(cfg, rows, split: str):
    cache = cfg.derived_root / "cache" / f"img_long{cfg.eval_long_side}"
    raster_dir = cfg.derived_root / "gt_instances"
    items = []
    for row in rows:
        if row["split"] != split or row["annotation_status"] != "annotated":
            continue
        image = cache / f"{row['image_id']}.png"
        raster = raster_dir / row["mask_path"]
        sidecar = raster_dir / row["mask_path"].replace(".png", ".json")
        if image.exists() and raster.exists() and sidecar.exists():
            items.append((image, raster, sidecar, row))
    return sorted(items, key=lambda t: t[3]["image_id"])


@torch.no_grad()
def validate(model, items, fdi_labels, cfg_model, device, score_thresholds):
    """Evaluate validation FDI-aware F1 at several score thresholds."""
    index_to_fdi = {i + 1: fdi for i, fdi in enumerate(fdi_labels)}
    model.eval()
    per_threshold: dict[float, list] = {t: [] for t in score_thresholds}
    for image_path, raster_path, sidecar_path, row in items:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        raster = load_instance_raster(raster_path)
        sidecar = load_instance_sidecar(sidecar_path)
        if img.shape[:2] != raster.shape[:2]:
            img = cv2.resize(
                img, (raster.shape[1], raster.shape[0]), interpolation=cv2.INTER_AREA
            )
        gt_masks, gt_fdis = [], []
        for inst in sidecar["instances"]:
            mask = raster == int(inst["instance_id"])
            if mask.any():
                gt_masks.append(mask)
                gt_fdis.append(inst.get("fdi"))

        tensor = (
            torch.from_numpy(img[:, :, ::-1].copy())
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
            .to(device)
        )
        out = model([tensor])[0]
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        masks = out["masks"].cpu().numpy()[:, 0]
        for threshold in score_thresholds:
            keep = scores >= threshold
            pred_masks, pred_fdis, pred_scores = [], [], []
            for mask, label, score in zip(masks[keep], labels[keep], scores[keep]):
                binary = mask >= cfg_model.mask_threshold
                if not binary.any():
                    continue
                pred_masks.append(binary)
                pred_fdis.append(index_to_fdi.get(int(label)))
                pred_scores.append(float(score))
            ev, _ = evaluate_image(
                row["image_id"],
                row["patient_id"],
                row["view_label"],
                gt_masks,
                gt_fdis,
                pred_masks,
                pred_fdis,
                pred_scores,
                compute_boundary=False,
            )
            per_threshold[threshold].append(ev)
    return {t: aggregate(evs) for t, evs in per_threshold.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-config", default="configs/segmentation/maskrcnn.yaml")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--replicate",
        type=int,
        default=None,
        help="rotating hold-out block to train against; defaults to --seed, which is "
        "the pairing the benchmark grid uses",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    payload = yaml.safe_load(Path(args.model_config).read_text()) or {}
    model_cfg = MaskRcnnConfig.from_dict(payload.get("maskrcnn"))
    thresholds = [
        float(t) for t in payload.get("maskrcnn", {}).get("select_score_thresholds", [0.05])
    ]
    if args.smoke:
        model_cfg = MaskRcnnConfig.from_dict(
            {**(payload.get("maskrcnn") or {}), "epochs": 1, "num_workers": 0}
        )

    # A replicate rotates the hold-out, so the training split has to be re-derived
    # from that replicate's split file rather than taken from the manifest column.
    replicate = args.seed if args.replicate is None else args.replicate
    rows = apply_seed_split(
        read_manifest(cfg.derived_root / "manifest.csv"), cfg.derived_root, replicate
    )
    train_items = collect(cfg, rows, "train")
    val_items = collect(cfg, rows, "val")
    if args.smoke:
        train_items, val_items = train_items[:8], val_items[:4]
    if not train_items:
        print("[maskrcnn] no training data", file=sys.stderr)
        return 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    fdi_labels = cfg.fdi_labels
    fdi_to_index = {fdi: i + 1 for i, fdi in enumerate(fdi_labels)}
    flip_map = {fdi: mirror_fdi(fdi) for fdi in fdi_labels}

    out_dir = (
        Path(args.out)
        if args.out
        else REPO / "runs" / cfg.name / "maskrcnn" / f"seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ToothInstanceDataset(
        [(a, b, c) for a, b, c, _ in train_items],
        fdi_to_index,
        hflip_prob=model_cfg.hflip_prob,
        flip_fdi=flip_map,
    )
    loader = DataLoader(
        dataset,
        batch_size=model_cfg.batch_size,
        shuffle=True,
        num_workers=model_cfg.num_workers,
        collate_fn=collate,
    )

    model = build_maskrcnn(
        fdi_labels,
        pretrained=True,
        trainable_backbone_layers=model_cfg.trainable_backbone_layers,
        max_detections=model_cfg.max_detections,
    ).to(args.device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=model_cfg.lr,
        momentum=model_cfg.momentum,
        weight_decay=model_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, model_cfg.epochs)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=model_cfg.amp and args.device == "cuda")

    best_f1 = -1.0
    best_epoch = -1
    best_threshold = thresholds[0]
    history = []
    started = time.perf_counter()
    stale = 0

    for epoch in range(1, model_cfg.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for images, targets in loader:
            images = [i.to(args.device) for i in images]
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
            if all(t["boxes"].numel() == 0 for t in targets):
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                losses = model(images, targets)
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss)
            n += 1
        scheduler.step()

        per_threshold = validate(
            model, val_items, fdi_labels, model_cfg, args.device, thresholds
        )
        best_here = max(
            per_threshold.items(), key=lambda kv: kv[1].get("fdi_instance_f1", 0.0)
        )
        epoch_f1 = best_here[1]["fdi_instance_f1"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / max(n, 1),
                "val_fdi_instance_f1": epoch_f1,
                "best_score_threshold": best_here[0],
                "val_instance_f1": best_here[1]["instance_f1"],
                "val_matched_dice": best_here[1]["matched_dice"],
            }
        )
        print(
            f"[maskrcnn] epoch {epoch:3d} loss={total / max(n, 1):.4f} "
            f"val_fdi_F1={epoch_f1:.4f} @score>={best_here[0]:.2f}",
            flush=True,
        )

        if epoch_f1 > best_f1:
            best_f1, best_epoch, best_threshold = epoch_f1, epoch, best_here[0]
            saved_cfg = dict(model_cfg.__dict__)
            saved_cfg["score_threshold"] = best_threshold
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_fdi_instance_f1": epoch_f1,
                    "fdi_labels": list(fdi_labels),
                    "config": saved_cfg,
                    "seed": args.seed,
                },
                out_dir / "best.pt",
            )
            stale = 0
        else:
            stale += 1
            if model_cfg.early_stopping_patience and stale >= model_cfg.early_stopping_patience:
                print(f"[maskrcnn] early stopping at epoch {epoch}", flush=True)
                break

    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": cfg.name,
                "method": "mask-rcnn",
                "seed": args.seed,
                "replicate": replicate,
                "split_hash": split_hash_for(cfg.derived_root, replicate),
                "best_epoch": best_epoch,
                "best_val_fdi_instance_f1": best_f1,
                "selected_score_threshold": best_threshold,
                "history": history,
                "n_train_images": len(train_items),
                "n_val_images": len(val_items),
                "n_classes": len(fdi_labels) + 1,
                "checkpoint_hash": sha256_file(out_dir / "best.pt"),
                "runtime": {
                    "train_seconds": time.perf_counter() - started,
                    "peak_gpu_mb": (
                        torch.cuda.max_memory_allocated() / 1e6
                        if torch.cuda.is_available()
                        else 0.0
                    ),
                },
                "config": model_cfg.__dict__,
            },
            indent=1,
        )
    )
    print(
        f"[maskrcnn] best val FDI-aware F1={best_f1:.4f} @epoch {best_epoch} "
        f"score>={best_threshold:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
