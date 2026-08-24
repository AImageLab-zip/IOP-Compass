"""View-classifier training loop.

Model selection uses validation macro F1 only; the test split is never read here.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import (
    VIEW_CLASSES,
    AugmentationConfig,
    Sample,
    ViewDataset,
)
from .models import build_resnet18


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    amp: bool = True
    early_stopping_patience: int = 8
    num_workers: int = 4
    grad_clip: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict | None) -> "TrainConfig":
        payload = dict(payload or {})
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    lr: float
    seconds: float


@dataclass
class TrainResult:
    best_epoch: int
    best_val_macro_f1: float
    best_val_accuracy: float
    history: list[EpochLog] = field(default_factory=list)
    checkpoint_path: Path | None = None
    total_seconds: float = 0.0
    peak_gpu_mb: float = 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def macro_f1(confusion: np.ndarray) -> float:
    f1s = []
    for k in range(confusion.shape[0]):
        tp = confusion[k, k]
        fp = confusion[:, k].sum() - tp
        fn = confusion[k, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


@dataclass
class EvalOutputs:
    loss: float
    confusion: np.ndarray
    probs: np.ndarray
    labels: np.ndarray
    image_ids: list[str]
    patient_ids: list[str]

    @property
    def accuracy(self) -> float:
        return float(np.trace(self.confusion) / max(self.confusion.sum(), 1))

    @property
    def macro_f1(self) -> float:
        return macro_f1(self.confusion)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module, loader: DataLoader, device: str, criterion: nn.Module
) -> EvalOutputs:
    model.eval()
    n_classes = len(VIEW_CLASSES)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    total_loss = 0.0
    n = 0
    probs_all: list[np.ndarray] = []
    labels_all: list[int] = []
    image_ids: list[str] = []
    patient_ids: list[str] = []
    for images, labels, ids, pids in loader:
        images = images.to(device, non_blocking=True)
        target = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, target)
        total_loss += float(loss) * images.size(0)
        n += images.size(0)
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        for t, p in zip(labels.numpy(), preds):
            confusion[int(t), int(p)] += 1
        probs_all.append(probs)
        labels_all.extend(int(v) for v in labels.numpy())
        image_ids.extend(ids)
        patient_ids.extend(pids)
    probs_arr = np.concatenate(probs_all) if probs_all else np.zeros((0, n_classes))
    return EvalOutputs(
        loss=total_loss / max(n, 1),
        confusion=confusion,
        probs=probs_arr,
        labels=np.asarray(labels_all, dtype=np.int64),
        image_ids=image_ids,
        patient_ids=patient_ids,
    )


def train_classifier(
    train_samples: list[Sample],
    val_samples: list[Sample],
    train_cfg: TrainConfig,
    aug_cfg: AugmentationConfig,
    image_size: int,
    pretrained: bool,
    seed: int,
    out_dir: Path,
    device: str = "cuda",
) -> TrainResult:
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ViewDataset(train_samples, image_size, aug_cfg, seed=seed)
    val_ds = ViewDataset(val_samples, image_size, AugmentationConfig(enabled=False))
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=train_cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        persistent_workers=train_cfg.num_workers > 0,
    )

    model = build_resnet18(len(VIEW_CLASSES), pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    if train_cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )
    elif train_cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=train_cfg.lr,
            momentum=0.9,
            weight_decay=train_cfg.weight_decay,
        )

    if train_cfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, train_cfg.epochs)
        )
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.amp and device == "cuda")

    result = TrainResult(best_epoch=-1, best_val_macro_f1=-1.0, best_val_accuracy=0.0)
    checkpoint_path = out_dir / "best.pt"
    started = time.perf_counter()
    epochs_without_improvement = 0

    for epoch in range(1, train_cfg.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running = 0.0
        seen = 0
        for images, labels, _, _ in train_loader:
            images = images.to(device, non_blocking=True)
            target = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(images)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            if train_cfg.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss) * images.size(0)
            seen += images.size(0)
        scheduler.step()

        val_out = evaluate_loader(model, val_loader, device, criterion)
        val_loss = val_out.loss
        accuracy = val_out.accuracy
        f1 = val_out.macro_f1
        log = EpochLog(
            epoch=epoch,
            train_loss=running / max(seen, 1),
            val_loss=val_loss,
            val_accuracy=accuracy,
            val_macro_f1=f1,
            lr=float(optimizer.param_groups[0]["lr"]),
            seconds=time.perf_counter() - epoch_start,
        )
        result.history.append(log)
        print(
            f"[classify] epoch {epoch:3d} train_loss={log.train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={accuracy:.4f} val_macroF1={f1:.4f}",
            flush=True,
        )

        if f1 > result.best_val_macro_f1:
            result.best_val_macro_f1 = f1
            result.best_val_accuracy = accuracy
            result.best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": f1,
                    "val_accuracy": accuracy,
                    "classes": list(VIEW_CLASSES),
                    "seed": seed,
                    "image_size": image_size,
                },
                checkpoint_path,
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                train_cfg.early_stopping_patience
                and epochs_without_improvement >= train_cfg.early_stopping_patience
            ):
                print(
                    f"[classify] early stopping at epoch {epoch} "
                    f"(best {result.best_val_macro_f1:.4f} @ {result.best_epoch})",
                    flush=True,
                )
                break

    result.checkpoint_path = checkpoint_path
    result.total_seconds = time.perf_counter() - started
    if device == "cuda":
        result.peak_gpu_mb = torch.cuda.max_memory_allocated() / 1e6
    (out_dir / "history.json").write_text(
        json.dumps([log.__dict__ for log in result.history], indent=1)
    )
    return result
