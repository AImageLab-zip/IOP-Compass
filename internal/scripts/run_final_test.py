#!/usr/bin/env python
"""Evaluate the held-out test set once, under the final-test lock.

    python internal/scripts/run_final_test.py --config configs/dataset_final_1000.yaml \
        --run final_1000 --final-test --config-hash <sha256 from freeze_experiment>

Refuses to run without ``--final-test`` and the frozen configuration hash, appends
an entry to ``results/<dataset>/test_access_log.jsonl`` recording date, git commit,
config hash, checkpoint hashes, user and host, and then submits the test-set
inference and evaluation jobs.

Every previous access is printed first, so repeated test-set use is impossible to
hide.
"""

from __future__ import annotations

import argparse
import getpass
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from slurm_util import JobSpec, record_jobs, submit  # noqa: E402

from iop_compass.data.adapter import load_dataset_config  # noqa: E402
from iop_compass.segmentation import grid  # noqa: E402

N_SHARDS = 8

# Must match the replicates the validation split was measured under; passed with
# --replicates so the locked split is never measured under a different set by accident.
DEFAULT_REPLICATES = (1, 2, 3)
# Both post halves are produced inline from the same forward pass, so post-processing
# runs at the inference resolution its frozen parameters were selected at.
SAT_CELLS = ",".join(c for c in grid.all_cells() if grid.parse_cell(c)[1] == "sat")
MRCNN_CELLS = ",".join(
    c for c in grid.all_cells() if grid.parse_cell(c)[1] == "mask-rcnn"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--config-hash", default=None)
    ap.add_argument("--reason", default="pre-registered final test evaluation")
    ap.add_argument(
        "--replicates",
        default=",".join(str(r) for r in DEFAULT_REPLICATES),
        help="comma-separated replicate indices; must match the validation campaign",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--evaluate-only", action="store_true")
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    replicates = tuple(int(v) for v in args.replicates.split(",") if v.strip() != "")
    if not replicates:
        print("[final-test] no replicates requested", file=sys.stderr)
        return 2
    run_dir = REPO / "runs" / args.run
    frozen_path = run_dir / "frozen_config.json"
    hash_path = run_dir / "frozen_config.sha256"

    if not frozen_path.exists() or not hash_path.exists():
        print(
            "[final-test] configuration is not frozen; run scripts/freeze_experiment.py",
            file=sys.stderr,
        )
        return 2
    expected = hash_path.read_text().strip()

    log_path = REPO / "results" / cfg.name / "test_access_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    previous = []
    if log_path.exists():
        previous = [json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()]
    if previous:
        print(f"[final-test] {len(previous)} previous test-set access(es):")
        for entry in previous:
            print(
                f"  {entry['timestamp_utc']} commit={entry['git_commit'][:8]} "
                f"hash={entry['config_hash'][:12]} reason={entry.get('reason')}"
            )

    if not args.final_test:
        print(
            "[final-test] refusing: pass --final-test to evaluate the held-out test set",
            file=sys.stderr,
        )
        return 3
    if args.config_hash != expected:
        print(
            "[final-test] refusing: --config-hash does not match the frozen configuration\n"
            f"  expected {expected}\n  given    {args.config_hash}",
            file=sys.stderr,
        )
        return 4

    frozen = json.loads(frozen_path.read_text())
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": cfg.name,
        "split": args.split,
        "config_hash": expected,
        "git_commit": frozen.get("git_commit"),
        "manifest_hash": frozen.get("dataset", {}).get("manifest_hash"),
        "split_hash": frozen.get("dataset", {}).get("split_hash"),
        "checkpoints": frozen.get("checkpoints"),
        "selection": frozen.get("selection"),
        "user": getpass.getuser(),
        "hostname": platform.node(),
        "slurm_job_id": None,
        "reason": args.reason,
        "replicates": list(replicates),
        "access_index": len(previous) + 1,
    }

    specs: list[JobSpec] = []
    if not args.evaluate_only:
        specs.append(
            JobSpec(
                name=f"segment_{args.split}",
                commands=[
                    f"python scripts/run_segmentation.py --config {args.config} "
                    f"--split {args.split} --cells {SAT_CELLS} --replicate {r} "
                    f"--shard {i} --num-shards {N_SHARDS}"
                    for r in replicates
                    for i in range(N_SHARDS)
                ],
                gpu=True,
                cpus=4,
                mem="48G",
                time="12:00:00",
            )
        )
        specs.append(
            JobSpec(
                name=f"maskrcnn_infer_{args.split}",
                commands=[
                    f"python scripts/run_segmentation.py --config {args.config} "
                    f"--split {args.split} --cells {MRCNN_CELLS} --replicate {r} "
                    f"--shard {i} --num-shards {N_SHARDS}"
                    for r in replicates
                    for i in range(N_SHARDS)
                ],
                gpu=True,
                cpus=4,
                mem="48G",
                time="04:00:00",
            )
        )

    submitted: dict[str, dict] = {}
    deps: list[str] = []
    for spec in specs:
        job_id = submit(spec, run_dir, dry_run=args.dry_run)
        deps.append(job_id)
        submitted[spec.name] = {
            "job_id": job_id,
            "name": spec.name,
            "n_tasks": len(spec.commands),
            "sbatch": {
                "cpus": spec.cpus,
                "mem": spec.mem,
                "time": spec.time,
                "partition": spec.resolved_partition(),
            },
        }

    evaluate = " && ".join(
        f"python scripts/evaluate_segmentation.py --config {args.config} "
        f"--split {args.split} --all-variants --replicate {r}"
        for r in replicates
    )
    eval_cmd = (
        f"python scripts/evaluate_roi.py --config {args.config} --split {args.split} "
        f"--reduce ; "
        f"{evaluate} && "
        f"python scripts/report_grid.py --config {args.config} --run {args.run} "
        f"--split {args.split} && "
        f"python scripts/evaluate_classifier.py --config {args.config} "
        f"--run {args.run} --split {args.split}"
    )
    eval_spec = JobSpec(
        name=f"evaluate_{args.split}",
        commands=[eval_cmd],
        gpu=False,
        cpus=8,
        mem="60G",
        time="12:00:00",
        depends_on=[d for d in deps if not d.startswith("dry_")],
    )
    job_id = submit(eval_spec, run_dir, dry_run=args.dry_run)
    submitted[eval_spec.name] = {
        "job_id": job_id,
        "name": eval_spec.name,
        "n_tasks": 1,
        "sbatch": {
            "cpus": eval_spec.cpus,
            "mem": eval_spec.mem,
            "time": eval_spec.time,
            "partition": eval_spec.resolved_partition(),
        },
    }
    entry["slurm_job_id"] = job_id

    if not args.dry_run:
        record_jobs(run_dir, submitted)
        with log_path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
        print(f"[final-test] recorded access #{entry['access_index']} in {log_path}")
    print(json.dumps({k: v["job_id"] for k, v in submitted.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
