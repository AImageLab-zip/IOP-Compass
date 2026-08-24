"""Helpers for building and submitting IOP-Compass SLURM jobs.

Cluster conventions were taken from the existing project scripts on this cluster:
``--account=grana_maxillo``, GPU work on ``all_usr_prod`` with ``--gres=gpu:1``,
CPU work on ``all_serial``, arrays supplied at submit time, ``uv`` venv activated
by the worker.  Per-user QOS caps GPUs at 8, so array throttles above that are
pointless.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKER = REPO / "internal" / "slurm" / "worker.sh"

ACCOUNT = os.environ.get("IOPC_ACCOUNT", "grana_maxillo")
GPU_PARTITION = os.environ.get("IOPC_GPU_PARTITION", "all_usr_prod")
CPU_PARTITION = os.environ.get("IOPC_CPU_PARTITION", "all_serial")
# GPUs with at least 24 GB; SAM 3 peaks around 4.5 GB and Mask R-CNN at 1024 px
# around 12 GB, so the small cards are excluded to avoid avoidable OOM retries.
#
# gpu_RTXPro6000B_96G is deliberately NOT listed: the RTX PRO 6000 Blackwell is
# sm_120, and the pinned PyTorch build compiles for sm_50-sm_90 only, so any task
# landing there dies with "CUDA error: no kernel image is available for execution
# on the device" as soon as it touches the GPU (it took out roi_eval_val task 1 on
# node `giacomo` in run final_1000).  Re-add it only together with a PyTorch
# build that advertises sm_120 in `torch.cuda.get_arch_list()`.
GPU_CONSTRAINT = os.environ.get(
    "IOPC_GPU_CONSTRAINT",
    "gpu_L40S_45G|gpu_A40_45G|gpu_RTX6000_24G|gpu_RTX_A5000_24G",
)
MAX_PARALLEL = int(os.environ.get("IOPC_MAX_PARALLEL", "8"))


@dataclass
class JobSpec:
    name: str
    commands: list[str]
    gpu: bool = False
    cpus: int = 4
    mem: str = "32G"
    time: str = "04:00:00"
    partition: str | None = None
    constraint: str | None = None
    max_parallel: int | None = None
    depends_on: list[str] = field(default_factory=list)
    dependency_type: str = "afterok"

    def resolved_partition(self) -> str:
        return self.partition or GPU_PARTITION


def write_tasklist(run_dir: Path, name: str, commands: list[str]) -> Path:
    path = run_dir / "slurm" / "tasklists" / f"{name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(commands) + "\n")
    return path


def build_sbatch_args(spec: JobSpec, run_dir: Path, tasklist: Path) -> list[str]:
    log_dir = run_dir / "slurm" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    n = len(spec.commands)
    throttle = spec.max_parallel or MAX_PARALLEL
    args = [
        "sbatch",
        "--parsable",
        f"--job-name=iopc_{spec.name}",
        f"--account={ACCOUNT}",
        f"--partition={spec.resolved_partition()}",
        f"--cpus-per-task={spec.cpus}",
        "--ntasks=1",
        f"--mem={spec.mem}",
        f"--time={spec.time}",
        f"--output={log_dir}/{spec.name}_%A_%a.out",
        f"--error={log_dir}/{spec.name}_%A_%a.err",
        f"--array=0-{n - 1}%{throttle}",
    ]
    # CPU-only stages stay on all_usr_prod too: all_serial caps walltime at 4h and
    # its QOS rejects any array submission over 3 tasks (QOSMaxSubmitJobPerUserLimit),
    # and several CPU stages here are longer arrays or exceed 4h.  They run without
    # --gres; all_usr_prod accepts single jobs and arrays that request no GPU.
    if spec.gpu:
        args.append("--gres=gpu:1")
        constraint = spec.constraint if spec.constraint is not None else GPU_CONSTRAINT
        if constraint:
            args.append(f"--constraint={constraint}")
    if spec.depends_on:
        args.append(
            f"--dependency={spec.dependency_type}:" + ":".join(spec.depends_on)
        )
    # NOTE: do not pass --export here.  On this cluster any job submitted with an
    # explicit --export (including --export=ALL,VAR=...) is cancelled by the
    # scheduler after ~2 s with no log output.  sbatch already forwards the
    # submitting environment, so the IOPC_* variables are handed over through the
    # subprocess environment in submit() instead.
    args.append(str(WORKER))
    return args


def git_state() -> tuple[str, int]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return commit, len([ln for ln in dirty.splitlines() if ln.strip()])
    except Exception:
        return "unknown", -1


def job_environment(run_dir: Path, tasklist: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["IOPC_TASKLIST"] = str(tasklist)
    env["IOPC_RUN_DIR"] = str(run_dir)
    env["IOPC_REPO_DIR"] = str(REPO)
    commit, dirty = git_state()
    env["IOPC_GIT_COMMIT"] = commit
    env["IOPC_GIT_DIRTY"] = str(dirty)
    return env


def submit(spec: JobSpec, run_dir: Path, dry_run: bool = False) -> str:
    tasklist = write_tasklist(run_dir, spec.name, spec.commands)
    args = build_sbatch_args(spec, run_dir, tasklist)
    if dry_run:
        print("DRY-RUN " + " ".join(shlex.quote(a) for a in args))
        return f"dry_{spec.name}"
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        env=job_environment(run_dir, tasklist),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"sbatch failed for {spec.name}: {proc.stderr.strip()}\n"
            + " ".join(shlex.quote(a) for a in args)
        )
    job_id = proc.stdout.strip().split(";")[0]
    print(f"[submit] {spec.name}: job {job_id} ({len(spec.commands)} tasks)")
    return job_id


def record_jobs(run_dir: Path, jobs: dict[str, dict]) -> Path:
    path = run_dir / "slurm" / "jobs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if path.exists():
        existing = json.loads(path.read_text())
    existing.update(jobs)
    path.write_text(json.dumps(existing, indent=1))
    return path


def load_jobs(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "slurm" / "jobs.json"
    return json.loads(path.read_text()) if path.exists() else {}


def job_state(job_id: str) -> str | None:
    """Coarse state of a submitted job, or ``None`` once SLURM has forgotten it.

    An array job is summarised by its tasks: still active if any task is, ``COMPLETED``
    only if every task completed, otherwise the first terminal failure seen.  Callers use
    this to tell "already satisfied" from "still to come", which a purged job id cannot
    express on its own.
    """
    proc = subprocess.run(
        ["sacct", "-j", str(job_id), "--format=State", "-X", "-n", "-P"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    states = {line.strip().split(" ")[0] for line in proc.stdout.splitlines() if line.strip()}
    if not states:
        return None
    active = {"PENDING", "RUNNING", "REQUEUED", "SUSPENDED", "COMPLETING", "RESIZING"}
    if states & active:
        return "ACTIVE"
    if states == {"COMPLETED"}:
        return "COMPLETED"
    return sorted(states - {"COMPLETED"})[0]
