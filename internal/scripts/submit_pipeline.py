#!/usr/bin/env python
"""Submit the IOP-Compass experimental campaign to SLURM.

    python internal/scripts/submit_pipeline.py --config configs/dataset_final_1000.yaml \
        --run-name final_1000 --stages prepare
    python internal/scripts/submit_pipeline.py --config ... --run-name final_1000 --stages all

Stages, in dependency order:

``prepare``   ground-truth rasters + the two image caches (CPU array)
``audit``     dataset audit with checksums; gates everything downstream
``classify``  view-classification variants x seeds (GPU array)
``roi``       ROI prior fitting, learned ROI training, ROI evaluation on val
``segment``   SegmentAnyTooth over every ROI level on val (GPU array, sharded);
              both post halves come from the same forward pass
``maskrcnn``  Mask R-CNN training, one per grid replicate (GPU array)
``evaluate``  evaluation + post-processing ablation + the grid report
``aggregate`` result aggregation, figures, LaTeX macros

The final test set is deliberately *not* part of this pipeline; it is evaluated
only by ``scripts/run_final_test.py`` after the configuration is frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from slurm_util import JobSpec, job_state, load_jobs, record_jobs, submit  # noqa: E402

from iop_compass.data.adapter import load_dataset_config  # noqa: E402
from iop_compass.data.splits import SplitError, check_seed_capacity  # noqa: E402
from iop_compass.segmentation import grid  # noqa: E402

PY = "python"
N_PREPARE_SHARDS = 16
N_SEGMENT_SHARDS = 8
SEEDS = (1, 2, 3)
CLASSIFIER_VARIANTS = ("C0", "C1", "C2")

# Default replicates of the benchmark grid, overridable with --replicates.  A
# replicate rotates the validation/test hold-out *and* selects the Mask R-CNN
# checkpoint trained for it, so it is one complete re-measurement rather than a
# re-seeded model on a fixed split.  The rotation must fit without reusing a hold-out
# patient; scripts/create_splits.py refuses the campaign up front when it cannot.
DEFAULT_REPLICATES = (1, 2, 3)


def stage_prepare(cfg_path: str, n_shards: int = N_PREPARE_SHARDS) -> list[JobSpec]:
    raster = [
        f"{PY} scripts/build_manifest.py --config {cfg_path} --rasters-only "
        f"--shard {i} --num-shards {n_shards}"
        for i in range(n_shards)
    ]
    cache = [
        f"{PY} scripts/build_image_cache.py --config {cfg_path} --kind classifier "
        f"--shard {i} --num-shards {n_shards} && "
        f"{PY} scripts/build_image_cache.py --config {cfg_path} --kind detector "
        f"--shard {i} --num-shards {n_shards}"
        for i in range(n_shards)
    ]
    verify = [
        f"{PY} scripts/verify_decodable.py --config {cfg_path} "
        f"--shard {i} --num-shards {n_shards}"
        for i in range(n_shards)
    ]
    return [
        JobSpec(
            name="prepare_rasters",
            commands=raster,
            gpu=False,
            cpus=2,
            mem="8G",
            time="02:00:00",
            max_parallel=8,
        ),
        JobSpec(
            name="verify_decode",
            commands=verify,
            gpu=False,
            cpus=2,
            mem="8G",
            time="03:00:00",
            max_parallel=8,
        ),
        JobSpec(
            name="prepare_cache",
            commands=cache,
            gpu=False,
            cpus=2,
            mem="8G",
            time="02:00:00",
            max_parallel=8,
        ),
    ]


def stage_audit(
    cfg_path: str, deps: list[str], replicates: tuple[int, ...] = DEFAULT_REPLICATES
) -> list[JobSpec]:
    derived = load_dataset_config(cfg_path).derived_root
    # One split file per replicate, written here so an infeasible rotation fails at
    # the audit gate instead of after a day of GPU time.
    cmd = (
        f"{PY} scripts/verify_decodable.py --config {cfg_path} --reduce && "
        f"{PY} scripts/create_splits.py --config {cfg_path} "
        f"--seeds {len(replicates)} && "
        f"{PY} scripts/build_manifest.py --config {cfg_path} --manifest-only "
        f"--splits {derived}/splits.json && "
        f"{PY} scripts/audit_dataset.py --config {cfg_path} "
        f"--splits {derived}/splits.json --checksums --fail-on-fatal && "
        f"{PY} scripts/report_exclusions.py --config {cfg_path}"
    )
    return [
        JobSpec(
            name="audit",
            commands=[cmd],
            gpu=False,
            cpus=4,
            mem="32G",
            time="06:00:00",
            depends_on=deps,
        )
    ]


def stage_classify(
    cfg_path: str, deps: list[str], replicates: tuple[int, ...] = DEFAULT_REPLICATES
) -> list[JobSpec]:
    """One classifier per variant per repetition.

    With a single replicate the repetitions are re-seeded models on the canonical
    partition, which is what this stage has always done.  With several replicates the
    seed *is* the replicate -- the convention ``stage_maskrcnn`` uses -- so each
    variant is measured on a different set of validation patients and the spread over
    seeds becomes a spread over partitions.  Either way the job count is
    ``len(CLASSIFIER_VARIANTS) * max(len(replicates), len(SEEDS))``.
    """
    seeds = replicates if len(replicates) > 1 else SEEDS
    commands = [
        f"{PY} scripts/train_classifier.py --config {cfg_path} "
        f"--variant-config configs/classification/{variant}.yaml --seed {seed} "
        f"--replicate {seed if len(replicates) > 1 else replicates[0]}"
        for variant in CLASSIFIER_VARIANTS
        for seed in seeds
    ]
    return [
        JobSpec(
            name="classify",
            commands=commands,
            gpu=True,
            cpus=4,
            mem="24G",
            time="04:00:00",
            depends_on=deps,
        )
    ]


def stage_classify_eval(
    cfg_path: str,
    run_name: str,
    deps: list[str],
    split: str = "val",
    replicates: tuple[int, ...] = DEFAULT_REPLICATES,
) -> list[JobSpec]:
    reps = ",".join(str(r) for r in replicates)
    return [
        JobSpec(
            name=f"classify_eval_{split}",
            commands=[
                f"{PY} scripts/evaluate_classifier.py --config {cfg_path} "
                f"--run {run_name} --split {split} --replicates {reps}"
            ],
            gpu=True,
            cpus=4,
            mem="24G",
            time="04:00:00",
            depends_on=deps,
        )
    ]


def stage_roi(
    cfg_path: str, deps: list[str], replicates: tuple[int, ...] = DEFAULT_REPLICATES
) -> list[JobSpec]:
    """One geometric prior and one learned detector per replicate.

    Both are fitted on *train* patients, and a replicate rotates the hold-out, so
    sharing them across replicates would fit on patients that are validation or test
    patients for the others.
    """
    fit = [
        f"{PY} scripts/fit_roi_priors.py --config {cfg_path} --replicate {r}"
        for r in replicates
    ]
    train_learned = [
        f"{PY} scripts/train_roi_detector.py --config {cfg_path} "
        f"--roi-config configs/roi/learned_roi.yaml --replicate {r}"
        for r in replicates
    ]
    return [
        JobSpec(
            name="roi_fit",
            commands=fit,
            gpu=False,
            cpus=4,
            mem="32G",
            time="06:00:00",
            depends_on=deps,
        ),
        JobSpec(
            name="roi_train",
            commands=train_learned,
            gpu=True,
            cpus=4,
            mem="32G",
            time="06:00:00",
            depends_on=deps,
        ),
    ]


def stage_roi_eval(
    cfg_path: str,
    deps: list[str],
    split: str = "val",
    replicates: tuple[int, ...] = DEFAULT_REPLICATES,
) -> list[JobSpec]:
    """ROI quality per replicate.

    ``stage_roi`` fits a separate geometric prior and learned detector per replicate,
    so evaluating only replicate 1 would report one replicate's ROI as if it were the
    campaign's.  The shards are reduced in ``stage_aggregate``, which is the first
    point where every shard of every replicate has landed.
    """
    commands = [
        f"{PY} scripts/evaluate_roi.py --config {cfg_path} --split {split} "
        f"--replicate {r} --shard {i} --num-shards {N_SEGMENT_SHARDS}"
        for r in replicates
        for i in range(N_SEGMENT_SHARDS)
    ]
    return [
        JobSpec(
            name=f"roi_eval_{split}",
            commands=commands,
            gpu=True,
            cpus=4,
            mem="32G",
            time="06:00:00",
            depends_on=deps,
        )
    ]


def cells_for(seg: str) -> str:
    """Every cell of one segmenter, both post halves.

    Both halves are produced inline from the same forward pass, which costs nothing
    extra and keeps post-processing at the *inference* resolution that the frozen
    parameters in configs/segmentation/postprocessing.yaml were selected at.
    Deriving the post half from a stored raw prediction instead would apply those
    parameters at the evaluation resolution, where a 3-pixel closing kernel covers
    twice the physical area -- worth up to 0.003 FDI-F1 on this cohort.
    """
    return ",".join(c for c in grid.all_cells() if grid.parse_cell(c)[1] == seg)


def stage_segment(
    cfg_path: str,
    deps: list[str],
    split: str = "val",
    replicates: tuple[int, ...] = DEFAULT_REPLICATES,
) -> list[JobSpec]:
    """SegmentAnyTooth over every ROI level; the post cells are derived later."""
    commands = [
        f"{PY} scripts/run_segmentation.py --config {cfg_path} --split {split} "
        f"--cells {cells_for('sat')} --replicate {r} "
        f"--shard {i} --num-shards {N_SEGMENT_SHARDS}"
        for r in replicates
        for i in range(N_SEGMENT_SHARDS)
    ]
    return [
        JobSpec(
            name=f"segment_{split}",
            commands=commands,
            gpu=True,
            cpus=4,
            mem="48G",
            time="20:00:00",
            depends_on=deps,
        )
    ]


def stage_repost(
    cfg_path: str,
    deps: list[str],
    split: str = "val",
    replicates: tuple[int, ...] = DEFAULT_REPLICATES,
) -> list[JobSpec]:
    """Derive every post-processed cell from the stored raw masks.

    A post cell is a pure function of its no-post twin plus the post-processing
    configuration, so a validation-selected change to that configuration is applied
    here instead of by re-running any model.
    """
    commands = [
        f"{PY} scripts/repostprocess.py --config {cfg_path} --split {split} "
        f"--all-pairs --replicate {r}"
        for r in replicates
    ]
    return [
        JobSpec(
            name=f"repost_{split}",
            commands=commands,
            gpu=False,
            cpus=4,
            mem="24G",
            time="04:00:00",
            depends_on=deps,
        )
    ]


def stage_maskrcnn(
    cfg_path: str, deps: list[str], replicates: tuple[int, ...] = DEFAULT_REPLICATES
) -> list[JobSpec]:
    commands = [
        f"{PY} scripts/train_maskrcnn.py --config {cfg_path} "
        f"--model-config configs/segmentation/maskrcnn.yaml --seed {r} --replicate {r}"
        for r in replicates
    ]
    return [
        JobSpec(
            name="maskrcnn",
            commands=commands,
            gpu=True,
            cpus=4,
            mem="48G",
            time="24:00:00",
            depends_on=deps,
        )
    ]


def stage_maskrcnn_infer(
    cfg_path: str,
    deps: list[str],
    split: str = "val",
    replicates: tuple[int, ...] = DEFAULT_REPLICATES,
) -> list[JobSpec]:
    commands = [
        f"{PY} scripts/run_segmentation.py --config {cfg_path} --split {split} "
        f"--cells {cells_for('mask-rcnn')} --replicate {r} "
        f"--shard {i} --num-shards {N_SEGMENT_SHARDS}"
        for r in replicates
        for i in range(N_SEGMENT_SHARDS)
    ]
    return [
        JobSpec(
            name=f"maskrcnn_infer_{split}",
            commands=commands,
            gpu=True,
            cpus=4,
            mem="48G",
            time="04:00:00",
            depends_on=deps,
        )
    ]


def stage_evaluate(
    cfg_path: str,
    run_name: str,
    deps: list[str],
    split: str = "val",
    replicates: tuple[int, ...] = DEFAULT_REPLICATES,
) -> list[JobSpec]:
    """One array task per replicate, then the grid report once they all land.

    Scoring a replicate touches only that replicate's cells and its own patients, and the
    paired bootstrap compares cells *within* a replicate, so the replicate is the
    coarsest unit that is still independent -- and the finest one that keeps `index.json`
    and the baseline comparison consistent.  Running them concurrently turns a sum into a
    max: the old single chained command paid 3x the per-replicate cost end to end.
    """
    commands = [
        f"{PY} scripts/evaluate_segmentation.py --config {cfg_path} --split {split} "
        f"--all-variants --replicate {r} && "
        f"{PY} scripts/ablate_postprocessing.py --config {cfg_path} --split {split} "
        f"--replicate {r}"
        for r in replicates
    ]
    return [
        JobSpec(
            name=f"evaluate_{split}",
            commands=commands,
            gpu=False,
            # Parallelism here is one task per replicate, not threads inside the task;
            # the extra cores serve the raster/prediction I/O and whatever the array
            # libraries use implicitly.
            cpus=16,
            mem="60G",
            time="12:00:00",
            depends_on=deps,
        )
    ]


def stage_report_grid(
    cfg_path: str, run_name: str, deps: list[str], split: str = "val"
) -> list[JobSpec]:
    """The grid report, which needs every replicate's cells to compute the spread."""
    return [
        JobSpec(
            name=f"report_grid_{split}",
            commands=[
                f"{PY} scripts/report_grid.py --config {cfg_path} --run {run_name} "
                f"--split {split}"
            ],
            gpu=False,
            cpus=8,
            mem="32G",
            time="02:00:00",
            depends_on=deps,
        )
    ]


def stage_aggregate(
    cfg_path: str,
    run_name: str,
    deps: list[str],
    split: str = "val",
    replicates: tuple[int, ...] = DEFAULT_REPLICATES,
) -> list[JobSpec]:
    """Reduce the ROI shards, then collect everything.

    The ROI reduction lives here rather than in ``stage_roi_eval`` because that stage
    is a job array: its tasks run concurrently, so none of them can see every shard.
    This stage already depends on it.
    """
    reduce_roi = " && ".join(
        f"{PY} scripts/evaluate_roi.py --config {cfg_path} --split {split} "
        f"--replicate {r} --reduce"
        for r in replicates
    )
    cmd = (
        f"{reduce_roi} && "
        f"{PY} scripts/report_checkpoints.py --run {run_name} --config {cfg_path} && "
        f"{PY} scripts/aggregate_results.py --run {run_name} --config {cfg_path}"
    )
    return [
        JobSpec(
            name="aggregate",
            commands=[cmd],
            gpu=False,
            cpus=8,
            mem="48G",
            time="06:00:00",
            depends_on=deps,
        )
    ]


# Stages a full campaign runs, in dependency order.  ``repost`` is deliberately not
# here: the segment stage already produces both post halves at the inference
# resolution their parameters were selected at, and re-deriving them from the stored
# evaluation-resolution rasters would quietly change the numbers.  It stays available
# as an explicit stage for re-applying changed post-processing parameters.
ALL_STAGES = (
    "prepare",
    "audit",
    "classify",
    "roi",
    "roi_eval",
    "segment",
    "maskrcnn",
    "maskrcnn_infer",
    "evaluate",
    "aggregate",
)
OPTIONAL_STAGES = ("repost",)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument(
        "--stages",
        default="all",
        help="comma-separated stage names, or 'all'",
    )
    ap.add_argument("--split", default="val")
    ap.add_argument(
        "--replicates",
        default=",".join(str(r) for r in DEFAULT_REPLICATES),
        help="comma-separated replicate indices; each rotates the val/test hold-out "
        "and gets its own Mask R-CNN checkpoint",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--no-deps",
        action="store_true",
        help="submit without SLURM dependencies (for resuming a single stage)",
    )
    args = ap.parse_args()

    stages = ALL_STAGES if args.stages == "all" else tuple(args.stages.split(","))
    known_stages = ALL_STAGES + OPTIONAL_STAGES
    unknown = [s for s in stages if s not in known_stages]
    if unknown:
        print(f"unknown stages: {unknown}", file=sys.stderr)
        return 2

    replicates = tuple(int(v) for v in args.replicates.split(",") if v.strip() != "")
    if not replicates:
        print("no replicates requested", file=sys.stderr)
        return 2

    # Refuse the campaign now if the rotating hold-out cannot fit, rather than after
    # the audit stage has already queued a day of downstream work.
    cfg = load_dataset_config(args.config)
    try:
        eligible = json.loads(
            (cfg.derived_root / "splits.json").read_text()
        )["n_eligible"]
    except (OSError, KeyError):
        eligible = None
    if eligible is not None:
        try:
            check_seed_capacity(eligible, cfg, len(replicates))
        except SplitError as exc:
            print(f"[submit] refusing: {exc}", file=sys.stderr)
            return 2
    print(f"[submit] replicates={list(replicates)} stages={list(stages)}")

    run_dir = REPO / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    known = load_jobs(run_dir)

    def dep(*names: str) -> list[str]:
        if args.no_deps:
            return []
        out = []
        for name in names:
            entry = known.get(name)
            if not (entry and entry.get("job_id")):
                continue
            job_id = str(entry["job_id"])
            # SLURM refuses the whole submission with "Job dependency problem" once it has
            # purged a finished job, so resuming a stage hours later would be impossible
            # while the dependency is in fact already satisfied.  Drop the ones that
            # completed; keep anything still active, and keep a failed one so afterok
            # blocks rather than quietly proceeding on partial inputs.
            state = job_state(job_id)
            if state == "COMPLETED":
                print(f"[submit] dependency {name} ({job_id}) already COMPLETED, dropping")
                continue
            if state is None:
                print(
                    f"[submit] dependency {name} ({job_id}) unknown to SLURM, dropping",
                    file=sys.stderr,
                )
                continue
            out.append(job_id)
        return out

    submitted: dict[str, dict] = {}

    def go(specs: list[JobSpec]) -> None:
        for spec in specs:
            job_id = submit(spec, run_dir, dry_run=args.dry_run)
            entry = {
                "job_id": job_id,
                "name": spec.name,
                "n_tasks": len(spec.commands),
                "gpu": spec.gpu,
                "partition": spec.resolved_partition(),
                "depends_on": spec.depends_on,
                "submitted_utc": datetime.now(timezone.utc).isoformat(),
            }
            submitted[spec.name] = entry
            known[spec.name] = entry

    if "prepare" in stages:
        go(stage_prepare(args.config))
    if "audit" in stages:
        go(
            stage_audit(
                args.config,
                dep("prepare_rasters", "prepare_cache", "verify_decode"),
                replicates,
            )
        )
    if "classify" in stages:
        go(stage_classify(args.config, dep("audit"), replicates))
        go(
            stage_classify_eval(
                args.config,
                args.run_name,
                dep("classify"),
                args.split,
                replicates,
            )
        )
    if "roi" in stages:
        go(stage_roi(args.config, dep("audit"), replicates))
    if "roi_eval" in stages:
        go(
            stage_roi_eval(
                args.config, dep("roi_fit", "roi_train"), args.split, replicates
            )
        )
    if "segment" in stages:
        go(stage_segment(args.config, dep("roi_fit", "roi_train"), args.split, replicates))
    if "repost" in stages:
        go(stage_repost(args.config, dep(f"segment_{args.split}"), args.split, replicates))
    if "maskrcnn" in stages:
        go(stage_maskrcnn(args.config, dep("audit"), replicates))
    if "maskrcnn_infer" in stages:
        go(stage_maskrcnn_infer(args.config, dep("maskrcnn"), args.split, replicates))
    if "evaluate" in stages:
        go(
            stage_evaluate(
                args.config,
                args.run_name,
                # Not repost: the post cells come from the segment stage itself, so a
                # stale repost job must not gate (or silently precede) evaluation.
                dep(
                    f"segment_{args.split}",
                    f"maskrcnn_infer_{args.split}",
                ),
                args.split,
                replicates,
            )
        )
        go(
            stage_report_grid(
                args.config,
                args.run_name,
                dep(f"evaluate_{args.split}"),
                args.split,
            )
        )
    if "aggregate" in stages:
        go(
            stage_aggregate(
                args.config,
                args.run_name,
                dep(
                    f"report_grid_{args.split}",
                    "classify_eval_val",
                    f"roi_eval_{args.split}",
                ),
                args.split,
                replicates,
            )
        )

    if not args.dry_run:
        record_jobs(run_dir, submitted)
        (run_dir / "config_snapshot.yaml").write_bytes(Path(args.config).read_bytes())
        print(json.dumps({k: v["job_id"] for k, v in submitted.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
