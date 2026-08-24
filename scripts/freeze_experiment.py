#!/usr/bin/env python
"""Freeze the experimental configuration before the test set is evaluated.

    python scripts/freeze_experiment.py --config configs/dataset_final_1000.yaml \
        --run final_1000

Collects everything that determines a test-set number - dataset config, ROI prior,
post-processing parameters, model configs, selected checkpoints and their hashes,
the manifest hash and the split hash - hashes it, and writes
``runs/<run>/frozen_config.json`` plus ``frozen_config.sha256``.

``scripts/run_final_test.py`` refuses to run unless it is given this hash, which is
what makes the final-test lock auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.data.adapter import load_dataset_config, sha256_file  # noqa: E402
from iop_compass.data.manifest import manifest_hash  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def collect_checkpoints(run_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(run_dir.rglob("best.pt")):
        rel = str(path.relative_to(run_dir))
        out[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return out


def selected_classifier(run_dir: Path) -> dict:
    """Principal classifier, using the same validation-only rule as the evaluator."""
    sys.path.insert(0, str(REPO / "scripts"))
    from evaluate_classifier import select_principal

    picked = select_principal(run_dir)
    if picked is None:
        return {}
    variant, seed = picked
    path = run_dir / "classification" / f"{variant}_seed{seed}" / "metrics.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {
        "variant": variant,
        "seed": seed,
        "metrics_path": str(path.relative_to(run_dir)),
        "val_macro_f1": payload.get("overall", {}).get("macro_f1"),
        "val_patient_set_exact": payload.get("overall", {}).get(
            "patient_set_exact_constrained"
        ),
        "checkpoint": payload.get("checkpoint"),
        "checkpoint_hash": payload.get("checkpoint_hash"),
        "selection_rule": "highest val macro F1; ties to C1; then lowest seed",
    }


def selected_maskrcnn(run_dir: Path) -> dict:
    best = None
    for path in sorted((run_dir / "maskrcnn").glob("seed*/metrics.json")):
        payload = json.loads(path.read_text())
        score = payload.get("best_val_fdi_instance_f1", 0.0)
        if best is None or score > best[0]:
            best = (score, payload, path)
    if best is None:
        return {}
    score, payload, path = best
    return {
        "seed": payload.get("seed"),
        "val_fdi_instance_f1": score,
        "selected_score_threshold": payload.get("selected_score_threshold"),
        "metrics_path": str(path.relative_to(run_dir)),
        "checkpoint_hash": payload.get("checkpoint_hash"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--roi-config", default="configs/roi/learned_roi.yaml")
    ap.add_argument("--post-config", default="configs/segmentation/postprocessing.yaml")
    ap.add_argument("--maskrcnn-config", default="configs/segmentation/maskrcnn.yaml")
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    run_dir = REPO / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    def read(path: str) -> str | None:
        p = Path(path)
        return p.read_text() if p.exists() else None

    payload = {
        "run": args.run,
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "dataset": {
            "config_path": args.config,
            "config_sha256": hashlib.sha256(
                Path(args.config).read_bytes()
            ).hexdigest(),
            "manifest_hash": manifest_hash(cfg.derived_root / "manifest.csv"),
            "split_hash": json.loads(
                (cfg.derived_root / "splits.json").read_text()
            )["split_hash"],
            "eval_long_side": cfg.eval_long_side,
            "inference_long_side": cfg.inference_long_side,
            "fdi_labels": list(cfg.fdi_labels),
        },
        "configs": {
            "roi": read(args.roi_config),
            "postprocessing": read(args.post_config),
            "maskrcnn": read(args.maskrcnn_config),
            "geometric_prior": {
                p.name: read(str(p.relative_to(REPO)))
                for p in sorted((run_dir / "roi").glob("geometric_prior_r*.json"))
            },
        },
        "checkpoints": collect_checkpoints(run_dir),
        "selection": {
            "classifier": selected_classifier(run_dir),
            "maskrcnn": selected_maskrcnn(run_dir),
        },
    }

    blob = json.dumps(payload, sort_keys=True, indent=1)
    digest = hashlib.sha256(blob.encode()).hexdigest()
    (run_dir / "frozen_config.json").write_text(blob)
    (run_dir / "frozen_config.sha256").write_text(digest + "\n")

    print(f"[freeze] wrote {run_dir / 'frozen_config.json'}")
    print(f"[freeze] config hash: {digest}")
    print("[freeze] selection:")
    print(json.dumps(payload["selection"], indent=1))
    print(
        "\n[freeze] evaluate the test set exactly once with:\n"
        f"  python internal/scripts/run_final_test.py --config {args.config} "
        f"--run {args.run} --final-test --config-hash {digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
