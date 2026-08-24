"""Collect every result file into one aggregate document.

The aggregate is the only thing the LaTeX generator reads, and every macro it
emits carries a provenance row (result file, JSON key, run, split, method, seed
aggregation, commit) so a reader can trace any number in the paper back to a file
on disk.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Provenance:
    macro: str
    value: str
    result_file: str
    json_key: str
    run_id: str
    split: str
    method: str
    seed_aggregation: str
    code_commit: str

    def to_row(self) -> dict[str, str]:
        return self.__dict__.copy()


@dataclass
class Aggregate:
    run_id: str
    commit: str
    dataset: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)
    classification: dict = field(default_factory=dict)
    roi: dict = field(default_factory=dict)
    segmentation: dict = field(default_factory=dict)
    postprocessing_ablation: dict = field(default_factory=dict)
    test_access: list = field(default_factory=list)
    provenance: list[Provenance] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = {
            "run_id": self.run_id,
            "code_commit": self.commit,
            "dataset": self.dataset,
            "audit": self.audit,
            "classification": self.classification,
            "roi": self.roi,
            "segmentation": self.segmentation,
            "postprocessing_ablation": self.postprocessing_ablation,
            "test_access": self.test_access,
        }
        return payload


def git_commit(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _roi_over_replicates(roi_dir: Path) -> dict | None:
    """Mean and standard deviation of every ROI metric across the replicates.

    Each replicate fits its own geometric prior and learned detector on its own
    training patients, so this spread is the ROI counterpart of the grid's
    ``std_over_replicates``: it says how much the ROI stage moves when the partition
    moves, which a single partition cannot show.  Returns ``None`` below two
    replicates, where a standard deviation would be meaningless.
    """
    if not roi_dir.exists():
        return None
    per_replicate: dict[int, dict] = {}
    for path in sorted(roi_dir.glob("summary_r*.json")):
        payload = _load(path)
        if payload is None:
            continue
        per_replicate[int(payload.get("replicate", 0))] = payload
    if len(per_replicate) < 2:
        return None

    replicates = sorted(per_replicate)
    strategies: dict[str, dict] = {}
    for name in sorted(
        {
            strategy
            for payload in per_replicate.values()
            for strategy in (payload.get("strategies") or {})
        }
    ):
        overalls = [
            per_replicate[r]["strategies"][name]["overall"]
            for r in replicates
            if name in (per_replicate[r].get("strategies") or {})
        ]
        metrics: dict[str, dict] = {}
        for metric in sorted({k for o in overalls for k in o}):
            values = [o[metric] for o in overalls if isinstance(o.get(metric), (int, float))]
            if len(values) < 2:
                continue
            mean = sum(values) / len(values)
            # population sd, matching scripts/report_grid.py's statistics.pstdev
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            metrics[metric] = {
                "mean": mean,
                "std": variance**0.5,
                "values": values,
            }
        if metrics:
            strategies[name] = metrics
    if not strategies:
        return None
    return {
        "replicates": replicates,
        "n_replicates": len(replicates),
        "spread": "std_over_replicates",
        "split_hashes": {
            str(r): per_replicate[r].get("split_hash") for r in replicates
        },
        "strategies": strategies,
    }


def build_aggregate(repo: Path, cfg, run_name: str) -> Aggregate:
    results = repo / "results" / cfg.name
    run_dir = repo / "runs" / run_name
    agg = Aggregate(run_id=cfg.name, commit=git_commit(repo))

    splits = _load(cfg.derived_root / "splits.json") or {}
    manifest_meta = _load(cfg.derived_root / "manifest_meta.json") or {}
    audit = _load(results / "audit" / "audit_report.json") or {}
    agg.audit = audit
    stats = audit.get("stats", {})

    agg.dataset = {
        "name": cfg.name,
        "n_patient_directories": stats.get("n_patient_directories"),
        "n_complete_patients": stats.get("n_complete_patients"),
        "n_eligible_patients": splits.get("n_eligible"),
        "n_excluded_patients": splits.get("n_excluded"),
        "n_train_patients": splits.get("n_train"),
        "n_val_patients": splits.get("n_val"),
        "n_test_patients": splits.get("n_test"),
        "n_images": stats.get("n_images"),
        "n_instances": stats.get("n_instances"),
        "n_distinct_fdi": stats.get("n_distinct_fdi"),
        "n_views": len(cfg.views_required),
        "instances_per_view": stats.get("instances_per_view", {}),
        "split_hash": splits.get("split_hash"),
        "manifest_hash": manifest_meta.get("manifest_hash"),
        "eval_long_side": cfg.eval_long_side,
        "inference_long_side": cfg.inference_long_side,
        "release_patients": 1000,
        "release_images": 5000,
    }

    for split in ("val", "test"):
        summary = _load(results / "classification" / split / "summary.json")
        if summary:
            agg.classification[split] = summary
            principal = summary.get("principal") or {}
            if principal:
                name = f"{principal['variant']}_seed{principal['seed']}"
                detail = _load(results / "classification" / split / f"{name}.json")
                if detail:
                    agg.classification[split]["principal_detail"] = detail

        roi_summary = _load(results / "roi" / split / "summary.json")
        if roi_summary:
            agg.roi[split] = roi_summary
        roi_replicated = _roi_over_replicates(results / "roi" / split)
        if roi_replicated:
            agg.roi.setdefault(split, {})
            agg.roi[split]["over_replicates"] = roi_replicated

        seg_dir = results / "segmentation" / split
        if seg_dir.exists():
            variants = {}
            for path in sorted(seg_dir.glob("*.json")):
                if path.name in ("index.json", "grid.json") or path.name.startswith(
                    "postprocessing_ablation"
                ):
                    continue
                payload = _load(path)
                if payload:
                    variants[path.stem] = payload
            if variants:
                agg.segmentation[split] = variants
            ablation = _load(seg_dir / "postprocessing_ablation.json")
            if ablation:
                agg.postprocessing_ablation[split] = ablation

    log = results / "test_access_log.jsonl"
    if log.exists():
        agg.test_access = [
            json.loads(ln) for ln in log.read_text().splitlines() if ln.strip()
        ]

    frozen = _load(run_dir / "frozen_config.json")
    if frozen:
        agg.dataset["frozen_config_hash"] = (
            (run_dir / "frozen_config.sha256").read_text().strip()
            if (run_dir / "frozen_config.sha256").exists()
            else None
        )
        agg.dataset["selection"] = frozen.get("selection")

    return agg
