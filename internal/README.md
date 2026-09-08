# internal/ — cluster-specific tooling

Everything here targets one specific SLURM installation. It is committed because the
benchmark numbers were produced with it and a reader should be able to see exactly how
jobs were shaped, not because it is expected to run elsewhere unmodified.

**None of it is needed to use the release.** To run the models on your own images see
`docs/inference.md`; to run the viewer see `docs/viewer.md`. Only reproducing the full
campaign end to end brings you here.

## Contents

| path | what it does |
| --- | --- |
| `slurm/worker.sh` | the single generic array worker; the job body is one line of a task list, and the submitter decides partition, memory, GPU and walltime |
| `scripts/slurm_util.py` | `JobSpec`, `sbatch` argument construction, job-state polling, `runs/<run>/slurm/jobs.json` |
| `scripts/submit_pipeline.py` | builds the task lists and submits every stage with the right dependencies |
| `scripts/monitor_slurm.py` | polls, retries infrastructure faults, refreshes `runs/<run>/STATUS_slurm.md` |
| `scripts/run_final_test.py` | the final-test lock: refuses to run without the frozen hash, appends to the test-access log, then submits the test-split jobs |
| `scripts/run_all.sh` | one command to submit a whole cohort |
| `scripts/finalize.sh` | aggregate and the summary report after a cohort finishes |
| `scripts/run_full_experiment.{sh,sbatch}` | the same sequence as a single detached driver job |
| `scripts/setup_env.sh` | uv venv on python 3.12 with the pinned wheels |
| `scripts/setup_tex.sh` | user-space TeX Live plus the LNCS class |
| `docs/known_failures.md` | every recorded job failure during the campaign and its diagnosis |

## Assumptions

These are hard-coded or defaulted, and are wrong for any other site:

- SLURM account `grana_maxillo`, GPU partition `all_usr_prod` with `--gres=gpu:1`;
- CPU-only stages also run on `all_usr_prod`, because the serial partition caps
  walltime at 4 h and its QOS rejects arrays over three tasks;
- a site environment script at `/work/grana_maxillo/lborghi/env.sh`;
- the repository checked out at `/work/grana_maxillo/lborghi/code/IOP-Compass`;
- a venv at `.venv/` inside the repository;
- the dataset root named in `configs/dataset_final_1000.yaml`.

## Overrides

| variable | changes |
| --- | --- |
| `IOPC_ACCOUNT` | the SLURM account |
| `IOPC_REPO_DIR` | the repository path the worker `cd`s into |
| `IOPC_TASKLIST` | the task list a worker reads (set by the submitter) |
| `IOPC_RUN_DIR` | where provenance JSON is written |
| `IOPC_ENV` | site script sourced by the `Makefile` before the venv |
| `TEXROOT` | the user-space TeX Live prefix |
| `SAT_CODE_DIR`, `SAT_WEIGHT_DIR` | SegmentAnyTooth source and weights; both required, defaults in `scripts/fetch_third_party.py` |
| `SAM3_CODE_DIR`, `SAM3_CHECKPOINT` | SAM 3 source and checkpoint; optional, needed only by the R3 ROI strategy |

The partition and constraint defaults live in `scripts/slurm_util.py`; change them
there for a different scheduler rather than patching each call site.
