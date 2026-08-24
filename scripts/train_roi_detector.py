#!/usr/bin/env python
"""Train the R2 learned single-box ROI detector on training patients only.

    python scripts/train_roi_detector.py --config configs/dataset_final_1000.yaml \
        --roi-config configs/roi/learned_roi.yaml

Selection metric: mean intersection-over-union between the predicted box and the
expanded tooth-union box on the validation split.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.data.adapter import load_dataset_config  # noqa: E402
from iop_compass.data.manifest import read_manifest  # noqa: E402
from iop_compass.data.rasterize import load_instance_raster  # noqa: E402
from iop_compass.data.splits import apply_seed_split, split_hash_for  # noqa: E402
from iop_compass.roi.factory import learned_roi_checkpoint  # noqa: E402
from iop_compass.roi.learned_roi import (  # noqa: E402
    LearnedRoiConfig,
    build_detector,
    target_box_from_raster,
)


class RoiBoxDataset(Dataset):
    def __init__(self, items: list[tuple[Path, Path]], margin_frac: float):
        self.items = items
        self.margin_frac = margin_frac

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_path, raster_path = self.items[index]
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        raster = load_instance_raster(raster_path)
        if img is None:
            raise IOError(f"cannot read {image_path}")
        if img.shape[:2] != raster.shape[:2]:
            img = cv2.resize(
                img, (raster.shape[1], raster.shape[0]), interpolation=cv2.INTER_AREA
            )
        box = target_box_from_raster(raster, self.margin_frac)
        tensor = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float().div_(255.0)
        if box is None:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            x0, y0, x1, y1 = box
            boxes = torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32)
            labels = torch.ones((1,), dtype=torch.int64)
        return tensor, {"boxes": boxes, "labels": labels}


def collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def box_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def collect(cfg, rows, split: str) -> list[tuple[Path, Path]]:
    cache = cfg.derived_root / "cache" / f"img_long{cfg.eval_long_side}"
    raster_dir = cfg.derived_root / "gt_instances"
    out = []
    for row in rows:
        if row["split"] != split or row["annotation_status"] != "annotated":
            continue
        image = cache / f"{row['image_id']}.png"
        raster = raster_dir / row["mask_path"]
        if image.exists() and raster.exists():
            out.append((image, raster))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--roi-config", default="configs/roi/learned_roi.yaml")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--replicate",
        type=int,
        default=None,
        help="replicate whose train split to fit on; defaults to --seed",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    payload = yaml.safe_load(Path(args.roi_config).read_text()) or {}
    roi_cfg = LearnedRoiConfig.from_dict(payload.get("learned_roi"))
    if args.smoke:
        roi_cfg = LearnedRoiConfig.from_dict(
            {**(payload.get("learned_roi") or {}), "epochs": 1, "num_workers": 0}
        )

    # Fit on this replicate's train patients only; another replicate's train set holds
    # patients that are validation or test patients here.
    replicate = args.seed if args.replicate is None else args.replicate
    rows = apply_seed_split(
        read_manifest(cfg.derived_root / "manifest.csv"), cfg.derived_root, replicate
    )
    split_hash = split_hash_for(cfg.derived_root, replicate)
    train_items = collect(cfg, rows, "train")
    val_items = collect(cfg, rows, "val")
    if args.smoke:
        train_items, val_items = train_items[:16], val_items[:8]
    if not train_items:
        print("[roi_train] no training images found", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = (
        Path(args.out)
        if args.out
        else learned_roi_checkpoint(cfg, replicate).parent
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        RoiBoxDataset(train_items, roi_cfg.target_margin_frac),
        batch_size=roi_cfg.batch_size,
        shuffle=True,
        num_workers=roi_cfg.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        RoiBoxDataset(val_items, roi_cfg.target_margin_frac),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    model = build_detector(pretrained=True).to(args.device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=roi_cfg.lr, momentum=roi_cfg.momentum, weight_decay=roi_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, roi_cfg.epochs)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=roi_cfg.amp and args.device == "cuda")

    best_iou = -1.0
    best_epoch = -1
    history = []
    started = time.perf_counter()
    for epoch in range(1, roi_cfg.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for images, targets in train_loader:
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

        model.eval()
        ious = []
        with torch.no_grad():
            for images, targets in val_loader:
                out = model([i.to(args.device) for i in images])[0]
                gt = targets[0]["boxes"]
                if gt.numel() == 0:
                    continue
                gt_box = gt[0].tolist()
                scores = out["scores"].cpu().numpy()
                if scores.size == 0:
                    ious.append(0.0)
                    continue
                pred = out["boxes"].cpu().numpy()[int(np.argmax(scores))].tolist()
                ious.append(box_iou(pred, gt_box))
        mean_iou = float(np.mean(ious)) if ious else 0.0
        history.append(
            {"epoch": epoch, "train_loss": total / max(n, 1), "val_box_iou": mean_iou}
        )
        print(
            f"[roi_train] epoch {epoch:3d} loss={total / max(n, 1):.4f} "
            f"val_box_iou={mean_iou:.4f}",
            flush=True,
        )
        if mean_iou > best_iou:
            best_iou, best_epoch = mean_iou, epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_box_iou": mean_iou,
                    "config": roi_cfg.__dict__,
                    "seed": args.seed,
                    # Recorded so roi.factory can refuse this checkpoint if it is ever
                    # used against a split it was not fitted on.
                    "replicate": replicate,
                    "split_hash": split_hash,
                },
                out_dir / "best.pt",
            )

    from iop_compass.data.adapter import sha256_file

    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": cfg.name,
                "method": "R2_learned",
                "seed": args.seed,
                "replicate": replicate,
                "split_hash": split_hash,
                "best_epoch": best_epoch,
                "best_val_box_iou": best_iou,
                "history": history,
                "n_train_images": len(train_items),
                "n_val_images": len(val_items),
                "checkpoint_hash": sha256_file(out_dir / "best.pt"),
                "runtime": {"train_seconds": time.perf_counter() - started},
                "config": roi_cfg.__dict__,
            },
            indent=1,
        )
    )
    print(f"[roi_train] best val_box_iou={best_iou:.4f} @ epoch {best_epoch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
