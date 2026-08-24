#!/usr/bin/env python
"""Document every trained weight of a run inside its results directory.

    python scripts/report_checkpoints.py --run final_1000 \
        --config configs/dataset_final_1000.yaml

Writes ``results/<dataset>/checkpoints.json``, ``results/<dataset>/checkpoints.md``
and a ``results/<dataset>/weights/`` directory of symlinks.

The weights themselves stay under ``runs/<run>/``: that is where
``scripts/freeze_experiment.py`` hashes them from, where ``run_segmentation.py``
resolves the Mask R-CNN checkpoint, and where ``roi/factory.py`` resolves the learned
detector, so moving them would break the final-test lock.  This script makes them
*findable* from the results side, with the hash of each file recorded next to the
metric it was selected on, so a reader of a results directory never has to guess which
file produced a number.  Pass ``--copy`` to place real copies in ``weights/`` instead
of symlinks.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.data.adapter import load_dataset_config, sha256_file  # noqa: E402


def _metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _entry(
    role: str,
    path: Path,
    run_dir: Path,
    *,
    variant: str | None = None,
    replicate: int | None = None,
    seed: int | None = None,
    metrics_path: Path | None = None,
    selected_on: dict | None = None,
) -> dict:
    return {
        "role": role,
        "variant": variant,
        # Only ever what the training run recorded.  Runs predating the rotation wrote
        # no replicate, and inferring one from the seed would claim a partition that
        # campaign never used.
        "replicate": replicate,
        "seed": seed,
        "path": str(path.relative_to(REPO)),
        "absolute_path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "metrics": (
            str(metrics_path.relative_to(REPO))
            if metrics_path and metrics_path.exists()
            else None
        ),
        "selected_on": selected_on or {},
        # Stable, collision-free name for the symlink farm.
        "link_name": "__".join(
            part
            for part in (
                role,
                variant,
                None if replicate is None else f"r{replicate}",
                None if seed is None or variant else f"seed{seed}",
            )
            if part
        )
        + path.suffix,
    }


def collect(run_dir: Path) -> list[dict]:
    """Every trained artefact of the run, with the replicate it belongs to."""
    entries: list[dict] = []

    for ckpt in sorted((run_dir / "maskrcnn").glob("seed*/best.pt")):
        payload = _metrics(ckpt.parent / "metrics.json")
        entries.append(
            _entry(
                "maskrcnn",
                ckpt,
                run_dir,
                replicate=payload.get("replicate"),
                seed=payload.get("seed"),
                metrics_path=ckpt.parent / "metrics.json",
                selected_on={
                    "best_val_fdi_instance_f1": payload.get(
                        "best_val_fdi_instance_f1"
                    ),
                    "selected_score_threshold": payload.get(
                        "selected_score_threshold"
                    ),
                    "best_epoch": payload.get("best_epoch"),
                    "split_hash": payload.get("split_hash"),
                },
            )
        )

    for ckpt in sorted((run_dir / "roi_detector").glob("r*/best.pt")):
        payload = _metrics(ckpt.parent / "metrics.json")
        entries.append(
            _entry(
                "roi_detector",
                ckpt,
                run_dir,
                replicate=payload.get("replicate", int(ckpt.parent.name.lstrip("r") or 0)),
                seed=payload.get("seed"),
                metrics_path=ckpt.parent / "metrics.json",
                selected_on={
                    "best_val_box_iou": payload.get("best_val_box_iou"),
                    "best_epoch": payload.get("best_epoch"),
                    "split_hash": payload.get("split_hash"),
                },
            )
        )

    for prior in sorted((run_dir / "roi").glob("geometric_prior_r*.json")):
        replicate = prior.stem.rsplit("_r", 1)[-1]
        entries.append(
            _entry(
                "geometric_prior",
                prior,
                run_dir,
                replicate=int(replicate) if replicate.isdigit() else None,
            )
        )

    for ckpt in sorted((run_dir / "classification").glob("*_seed*/best.pt")):
        variant, _, seed = ckpt.parent.name.partition("_seed")
        payload = _metrics(ckpt.parent / "metrics.json")
        entries.append(
            _entry(
                "classifier",
                ckpt,
                run_dir,
                variant=f"{variant}_seed{seed}",
                replicate=payload.get("replicate"),
                seed=payload.get("seed"),
                metrics_path=ckpt.parent / "metrics.json",
                selected_on={
                    "best_val_macro_f1": payload.get("best_val_macro_f1"),
                    "best_epoch": payload.get("best_epoch"),
                    "split_hash": payload.get("split_hash"),
                },
            )
        )

    return entries


def link_weights(entries: list[dict], weights_dir: Path, copy: bool) -> None:
    weights_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        link = weights_dir / entry["link_name"]
        target = Path(entry["absolute_path"])
        if link.is_symlink() or link.exists():
            link.unlink()
        if copy:
            shutil.copy2(target, link)
        else:
            link.symlink_to(target)
        entry["results_link"] = str(link.relative_to(REPO))


def markdown(run_name: str, dataset: str, entries: list[dict], copy: bool) -> str:
    kind = "copies" if copy else "symlinks"
    lines = [
        f"# Trained weights — run `{run_name}`",
        "",
        f"Dataset `{dataset}`. Generated {datetime.now(timezone.utc).isoformat()} by "
        "`scripts/report_checkpoints.py`.",
        "",
        f"The files live under `runs/{run_name}/` — that is the path every consumer "
        "resolves (`freeze_experiment.py` hashes them from there, `run_segmentation.py` "
        "and `roi/factory.py` load them from there). `weights/` in this directory holds "
        f"{kind} of the same files for convenience; the `sha256` column is the "
        "authoritative identity.",
        "",
        "| role | variant | replicate | seed | path | bytes | sha256 | selected on |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        selected = ", ".join(
            f"{k}={v}"
            for k, v in entry["selected_on"].items()
            if v is not None and k != "split_hash"
        )
        lines.append(
            "| {role} | {variant} | {replicate} | {seed} | `{path}` | {bytes} | `{sha}` | {sel} |".format(
                role=entry["role"],
                variant=entry["variant"] or "-",
                replicate="-" if entry["replicate"] is None else entry["replicate"],
                seed="-" if entry["seed"] is None else entry["seed"],
                path=entry["path"],
                bytes=entry["bytes"],
                sha=(entry["sha256"] or "")[:16],
                sel=selected or "-",
            )
        )
    if not entries:
        lines.append("| _no checkpoints found_ | | | | | | | |")
    lines += [
        "",
        "Per-replicate split hashes are recorded in `checkpoints.json`, so a checkpoint "
        "can always be tied back to the exact partition it was trained on.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument(
        "--copy",
        action="store_true",
        help="place real copies in results/<dataset>/weights/ instead of symlinks",
    )
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    run_dir = REPO / "runs" / args.run
    if not run_dir.exists():
        print(f"[checkpoints] missing {run_dir}", file=sys.stderr)
        return 1

    out_dir = REPO / "results" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = collect(run_dir)
    link_weights(entries, out_dir / "weights", args.copy)

    (out_dir / "checkpoints.json").write_text(
        json.dumps(
            {
                "run_id": cfg.name,
                "run_name": args.run,
                "run_dir": str(run_dir.relative_to(REPO)),
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "materialised_as": "copy" if args.copy else "symlink",
                "n_checkpoints": len(entries),
                "checkpoints": entries,
            },
            indent=1,
        )
    )
    (out_dir / "checkpoints.md").write_text(
        markdown(args.run, cfg.name, entries, args.copy)
    )
    print(
        f"[checkpoints] {len(entries)} artefacts -> {out_dir / 'checkpoints.md'} "
        f"and {out_dir / 'weights'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
