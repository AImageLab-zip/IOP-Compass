"""Reproducible figure generation.

Every figure writes the data it was drawn from next to it as CSV or JSON, so a
plot can be re-checked without re-running the pipeline.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..classification.dataset import VIEW_CLASSES  # noqa: E402
from .labels import ROI_ORDER, ROI_SHORT, VIEW_SHORT  # noqa: E402

# Colour-blind-safe qualitative palette (Okabe-Ito).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def _dump(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def confusion_figure(agg, out_dir: Path, split: str = "test") -> Path | None:
    detail = (agg.classification.get(split) or {}).get("principal_detail")
    if not detail:
        return None
    matrix = np.asarray(
        detail["overall"]["confusion_constrained"], dtype=float
    )
    raw = np.asarray(detail["overall"]["confusion"], dtype=float)
    normalised = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.4))
    for ax, data, title in (
        (axes[0], raw / np.maximum(raw.sum(axis=1, keepdims=True), 1), "Independent argmax"),
        (axes[1], normalised, "Constrained assignment"),
    ):
        im = ax.imshow(data, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(VIEW_CLASSES)))
        ax.set_yticks(range(len(VIEW_CLASSES)))
        labels = [VIEW_SHORT.get(v, v).replace(" ", "\n") for v in VIEW_CLASSES]
        ax.set_xticklabels(labels, fontsize=5, rotation=45, ha="right")
        ax.set_yticklabels(labels, fontsize=5)
        ax.set_title(title, fontsize=7)
        ax.grid(False)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i, j] > 0.005:
                    ax.text(
                        j,
                        i,
                        f"{data[i, j]:.2f}".lstrip("0"),
                        ha="center",
                        va="center",
                        fontsize=4.5,
                        color="white" if data[i, j] > 0.5 else "black",
                    )
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    path = out_dir / "view_confusion.pdf"
    fig.savefig(path)
    plt.close(fig)
    _dump(
        out_dir / "view_confusion_data.csv",
        [
            {
                "true_view": VIEW_CLASSES[i],
                "pred_view": VIEW_CLASSES[j],
                "count_independent": int(raw[i, j]),
                "count_constrained": int(matrix[i, j]),
            }
            for i in range(len(VIEW_CLASSES))
            for j in range(len(VIEW_CLASSES))
        ],
    )
    return path


def roi_figure(agg, out_dir: Path, split: str = "test") -> Path | None:
    roi = (agg.roi.get(split) or {}).get("strategies") or {}
    if not roi:
        return None
    names = [n for n in ROI_ORDER if n in roi]
    retention = [
        100 * roi[n]["overall"]["tooth_pixel_retention_pooled"] for n in names
    ]
    area = [100 * roi[n]["overall"]["area_retention_mean"] for n in names]

    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    ax.scatter(area, retention, c=PALETTE[: len(names)], s=26, zorder=3)
    for x, y, name in zip(area, retention, names):
        ax.annotate(
            ROI_SHORT.get(name, name).split(" ")[0],
            (x, y),
            textcoords="offset points",
            xytext=(4, -2),
            fontsize=6,
        )
    ax.set_xlabel("image area retained (%)")
    ax.set_ylabel("reference tooth pixels retained (%)")
    ax.set_ylim(min(retention) - 0.5, 100.3)
    path = out_dir / "roi_tradeoff.pdf"
    fig.savefig(path)
    plt.close(fig)
    _dump(
        out_dir / "roi_tradeoff_data.csv",
        [
            {
                "strategy": n,
                "tooth_pixel_retention_pct": 100 * roi[n]["overall"]["tooth_pixel_retention_pooled"],
                "area_retention_pct": 100 * roi[n]["overall"]["area_retention_mean"],
                "catastrophic_rate": roi[n]["overall"]["catastrophic_failure_rate"],
                "fallback_rate": roi[n]["overall"]["fallback_rate"],
                "mean_runtime_s": roi[n]["overall"]["mean_runtime_s"],
            }
            for n in names
        ],
    )
    return path


def calibration_figure(agg, out_dir: Path, split: str = "test", n_bins: int = 10) -> Path | None:
    cls = agg.classification.get(split) or {}
    principal = cls.get("principal") or {}
    if not principal:
        return None
    # the per-image predictions live next to the metrics written by
    # scripts/evaluate_classifier.py
    repo = Path(__file__).resolve().parents[3]
    npz_path = (
        repo
        / "results"
        / agg.run_id
        / "classification"
        / split
        / f"{principal['variant']}_seed{principal['seed']}_predictions.npz"
    )
    if not npz_path.is_file():
        return None
    data = np.load(npz_path, allow_pickle=True)
    probs, labels = data["probs"], data["labels"]
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (confidence > lo) & (confidence <= hi)
        if not sel.any():
            continue
        xs.append(confidence[sel].mean())
        ys.append(correct[sel].mean())
        counts.append(int(sel.sum()))

    fig, ax = plt.subplots(figsize=(2.6, 2.2))
    ax.plot([0, 1], [0, 1], ls="--", lw=0.6, color="grey")
    ax.plot(xs, ys, "o-", color=PALETTE[0], markersize=3, lw=1.0)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    path = out_dir / "calibration.pdf"
    fig.savefig(path)
    plt.close(fig)
    _dump(
        out_dir / "calibration_data.csv",
        [
            {"bin_confidence": x, "bin_accuracy": y, "n": n}
            for x, y, n in zip(xs, ys, counts)
        ],
    )
    return path


def generate_all(agg, out_dir: Path, split: str = "test") -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    produced = []
    for fn in (
        confusion_figure,
        roi_figure,
        calibration_figure,
    ):
        path = fn(agg, out_dir, split)
        if path:
            produced.append(path)
            marker = path.with_suffix(".placeholder")
            if marker.exists():
                marker.unlink()
    (out_dir / "figure_index.json").write_text(
        json.dumps([p.name for p in produced], indent=1)
    )
    return produced
