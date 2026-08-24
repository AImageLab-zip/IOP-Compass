#!/usr/bin/env python
"""Monitor the SLURM campaign and keep runs/<run>/STATUS_slurm.md current.

    python internal/scripts/monitor_slurm.py --run final_1000 --once
    python internal/scripts/monitor_slurm.py --run final_1000 --watch --interval 300

Retries are deliberately narrow.  Infrastructure faults (PREEMPTED, NODE_FAIL,
TIMEOUT, transient filesystem errors) are retried at most twice, with more wall
time on a TIMEOUT.  Software faults - assertion errors, import errors, missing
weights, NaN loss, label-mapping errors, leakage - are never retried; they are
reported so they can be diagnosed.

The monitor never issues a broad ``scancel`` and never touches a job it did not
find in ``runs/<run>/slurm/jobs.json``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from slurm_util import (  # noqa: E402
    GPU_CONSTRAINT,
    job_environment,
    load_jobs,
    record_jobs,
)

TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "PREEMPTED",
    "OUT_OF_MEMORY",
    "BOOT_FAIL",
    "DEADLINE",
    "SPECIAL_EXIT",
}
RETRYABLE = {"PREEMPTED", "NODE_FAIL", "TIMEOUT", "BOOT_FAIL", "OUT_OF_MEMORY"}
MAX_RETRIES = 2

# Log patterns that mark a software fault: never retry these.
SOFTWARE_FAULT_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"AssertionError",
        r"ModuleNotFoundError",
        r"ImportError",
        r"FileNotFoundError",
        r"KeyError",
        r"ValueError",
        r"TypeError",
        r"loss is nan",
        r"nan loss",
        r"patient leakage",
        r"uninterpretable",
        r"missing SegmentAnyTooth",
        r"missing frozen",
        # GPU architecture the pinned PyTorch build has no kernels for.  Retrying
        # is pointless and, worse, may succeed on a different node and hide the
        # fact that the GPU constraint admits an unusable card: fail loudly
        # instead, so GPU_CONSTRAINT in slurm_util.py gets fixed.
        r"no kernel image is available",
        r"is not compatible with the current PyTorch installation",
    )
]
TRANSIENT_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"Input/output error",
        r"Stale file handle",
        r"Resource temporarily unavailable",
        r"Transport endpoint is not connected",
        r"CUDA error: unknown error",
        r"NCCL",
        # A task that asked for --gres=gpu:1 and still found no device: the allocation,
        # not the code, is wrong.  Worth one retry elsewhere.  (Seen after an in-place
        # `scontrol update ... Partition=` on pending tasks silently dropped their GPU
        # TRES -- never patch a queued job's partition, resubmit it instead.)
        r"No CUDA GPUs are available",
    )
]


@dataclass
class TaskState:
    job_id: str
    state: str
    exit_code: str = ""
    elapsed: str = ""
    max_rss: str = ""
    node: str = ""


@dataclass
class StageState:
    name: str
    job_id: str
    n_tasks: int
    tasks: list[TaskState] = field(default_factory=list)
    retries: int = 0
    # every job that has run this stage, so a retry still in the queue counts as active
    retry_job_ids: list[str] = field(default_factory=list)

    @property
    def all_job_ids(self) -> list[str]:
        return [self.job_id, *self.retry_job_ids]

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for task in self.tasks:
            out[task.state] = out.get(task.state, 0) + 1
        return out

    @property
    def finished(self) -> bool:
        return bool(self.tasks) and all(
            t.state.split()[0] in TERMINAL for t in self.tasks
        )

    @property
    def failed_tasks(self) -> list[TaskState]:
        return [
            t
            for t in self.tasks
            if t.state.split()[0] in TERMINAL and t.state.split()[0] != "COMPLETED"
        ]


def sacct(job_id: str) -> list[TaskState]:
    proc = subprocess.run(
        [
            "sacct",
            "-j",
            str(job_id),
            "-X",
            "-P",
            "-n",
            "-o",
            "JobID,State,ExitCode,Elapsed,MaxRSS,NodeList",
        ],
        capture_output=True,
        text=True,
    )
    tasks: list[TaskState] = []
    for line in proc.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 6:
            continue
        tasks.append(
            TaskState(
                job_id=parts[0],
                state=parts[1],
                exit_code=parts[2],
                elapsed=parts[3],
                max_rss=parts[4],
                node=parts[5],
            )
        )
    return tasks


def squeue_active(job_id: str) -> int:
    proc = subprocess.run(
        ["squeue", "-h", "-j", str(job_id), "-o", "%i"], capture_output=True, text=True
    )
    return len([ln for ln in proc.stdout.splitlines() if ln.strip()])


def classify_failure(run_dir: Path, stage: str, task_id: str) -> tuple[str, str]:
    """Return ``(kind, evidence)`` where kind is software / transient / unknown."""
    array_part = task_id.split("_")
    patterns = [f"{stage}_*_{array_part[-1]}.err", f"{stage}_*.err"]
    log_dir = run_dir / "slurm" / "logs"
    for pattern in patterns:
        for path in sorted(log_dir.glob(pattern)):
            try:
                text = path.read_text(errors="ignore")[-20000:]
            except OSError:
                continue
            for rx in SOFTWARE_FAULT_PATTERNS:
                m = rx.search(text)
                if m:
                    return "software", f"{path.name}: {m.group(0)}"
            for rx in TRANSIENT_PATTERNS:
                m = rx.search(text)
                if m:
                    return "transient", f"{path.name}: {m.group(0)}"
    return "unknown", ""


def _index_of(task: TaskState, fallback_indices: list[str]) -> str:
    """Array index a task belongs to.

    Retries are array jobs over the failed indices, so their ``JobID`` carries the
    same index as the attempt they replace.  A retry submitted by hand as a plain
    (non-array) job has no suffix, so it is attributed to the recorded indices.
    """
    suffix = task.job_id.split("_")[-1]
    if "_" in task.job_id and suffix.isdigit():
        return suffix
    return fallback_indices[0] if len(fallback_indices) == 1 else task.job_id


def reconcile_attempts(
    original: list[TaskState], retries: list[tuple[list[str], list[TaskState]]]
) -> list[TaskState]:
    """Collapse every attempt of an array index to its best outcome.

    Without this a retried index stays FAILED for the rest of the campaign, because
    the original array job's sacct record never changes: the stage then reports a
    failure forever and the driver aborts even though the work succeeded.
    """
    best: dict[str, TaskState] = {}
    order: list[str] = []

    def offer(index: str, task: TaskState) -> None:
        if index not in best:
            best[index] = task
            order.append(index)
            return
        current = best[index].state.split()[0]
        if current != "COMPLETED" and task.state.split()[0] == "COMPLETED":
            best[index] = task

    for task in original:
        offer(_index_of(task, []), task)
    for indices, tasks in retries:
        for task in tasks:
            offer(_index_of(task, indices), task)
    return [best[i] for i in order]


def collect(run_dir: Path) -> list[StageState]:
    jobs = load_jobs(run_dir)
    stages: list[StageState] = []
    for name, entry in jobs.items():
        job_id = str(entry.get("job_id", ""))
        if not job_id or job_id.startswith("dry_"):
            continue
        retry_ids = [str(j) for j in entry.get("retry_job_ids", [])]
        retry_indices = [str(i) for i in entry.get("retry_indices", [])]
        retries = [(retry_indices, sacct(rid)) for rid in retry_ids if rid]
        stages.append(
            StageState(
                name=name,
                job_id=job_id,
                n_tasks=int(entry.get("n_tasks", 1)),
                tasks=reconcile_attempts(sacct(job_id), retries),
                retries=int(entry.get("retries", 0)),
                retry_job_ids=retry_ids,
            )
        )
    return stages


def retry_stage(run_dir: Path, stage: StageState, failed: list[TaskState]) -> str | None:
    """Resubmit only the failed array indices of a stage."""
    jobs = load_jobs(run_dir)
    entry = jobs.get(stage.name, {})
    if entry.get("retries", 0) >= MAX_RETRIES:
        return None
    tasklist = run_dir / "slurm" / "tasklists" / f"{stage.name}.txt"
    if not tasklist.exists():
        return None

    indices = sorted(
        {t.job_id.split("_")[-1] for t in failed if "_" in t.job_id and t.job_id.split("_")[-1].isdigit()}
    )
    if not indices:
        indices = ["0"]

    timeout_seen = any(t.state.startswith("TIMEOUT") for t in failed)
    oom_seen = any(t.state.startswith("OUT_OF_MEMORY") for t in failed)
    original = entry.get("sbatch", {})
    time_limit = original.get("time", "12:00:00")
    mem = original.get("mem", "48G")
    if timeout_seen:
        hours = int(time_limit.split(":")[0]) if ":" in time_limit else 12
        time_limit = f"{min(hours * 2, 24)}:00:00"
    if oom_seen:
        value = int(re.sub(r"[^0-9]", "", mem) or 48)
        mem = f"{min(value * 2, 240)}G"

    args = [
        "sbatch",
        "--parsable",
        f"--job-name=iopc_{stage.name}_retry",
        f"--account={original.get('account', 'grana_maxillo')}",
        f"--partition={original.get('partition', 'all_usr_prod')}",
        f"--cpus-per-task={original.get('cpus', 4)}",
        "--ntasks=1",
        f"--mem={mem}",
        f"--time={time_limit}",
        "--gres=gpu:1",
        # Without this a retry inherits no GPU constraint and can land on a card the
        # pinned PyTorch has no kernels for, turning a retry into a second failure.
        f"--constraint={GPU_CONSTRAINT}",
        f"--output={run_dir}/slurm/logs/{stage.name}_retry_%A_%a.out",
        f"--error={run_dir}/slurm/logs/{stage.name}_retry_%A_%a.err",
        f"--array={','.join(indices)}%8",
        str(REPO / "internal" / "slurm" / "worker.sh"),
    ]
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        env=job_environment(run_dir, tasklist),
    )
    if proc.returncode != 0:
        print(f"[monitor] retry submit failed for {stage.name}: {proc.stderr.strip()}")
        return None
    new_id = proc.stdout.strip().split(";")[0]
    entry = dict(entry)
    entry["retries"] = entry.get("retries", 0) + 1
    entry["retry_job_ids"] = entry.get("retry_job_ids", []) + [new_id]
    entry["retry_indices"] = indices
    record_jobs(run_dir, {stage.name: entry})
    print(
        f"[monitor] retried {stage.name} indices={','.join(indices)} "
        f"as job {new_id} (attempt {entry['retries']}/{MAX_RETRIES})"
    )
    return new_id


def write_status(run_dir: Path, stages: list[StageState], failures: list[dict]) -> None:
    lines = [
        "# STATUS",
        "",
        f"Last updated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Run directory: `{run_dir}`",
        "",
        "## SLURM stages",
        "",
        "| stage | job | tasks | states | retries |",
        "| --- | --- | --- | --- | --- |",
    ]
    for stage in stages:
        counts = ", ".join(f"{k}={v}" for k, v in sorted(stage.counts.items())) or "queued"
        lines.append(
            f"| {stage.name} | {stage.job_id} | {stage.n_tasks} | {counts} | {stage.retries} |"
        )
    lines += ["", "## Failures", ""]
    if not failures:
        lines.append("None.")
    else:
        lines += ["| stage | task | state | classification | evidence |", "| --- | --- | --- | --- | --- |"]
        for f in failures:
            lines.append(
                f"| {f['stage']} | {f['task']} | {f['state']} | {f['kind']} | "
                f"{f['evidence'][:120]} |"
            )
    (run_dir / "STATUS_slurm.md").write_text("\n".join(lines) + "\n")


def tick(run_dir: Path, allow_retry: bool) -> tuple[bool, list[dict]]:
    stages = collect(run_dir)
    failures: list[dict] = []
    all_done = True
    for stage in stages:
        active = sum(squeue_active(jid) for jid in stage.all_job_ids)
        if active or not stage.finished:
            all_done = False
        failed = stage.failed_tasks
        if not failed:
            continue
        kinds = []
        for task in failed:
            kind, evidence = classify_failure(run_dir, stage.name, task.job_id)
            if task.state.split()[0] in RETRYABLE and kind != "software":
                kind = "transient" if kind == "unknown" else kind
            kinds.append(kind)
            failures.append(
                {
                    "stage": stage.name,
                    "task": task.job_id,
                    "state": task.state,
                    "kind": kind,
                    "evidence": evidence,
                    "retried": False,
                }
            )
        if allow_retry and stage.finished and all(k != "software" for k in kinds):
            retryable = [
                t
                for t, k in zip(failed, kinds)
                if k == "transient" or t.state.split()[0] in RETRYABLE
            ]
            if retryable and retry_stage(run_dir, stage, retryable):
                for f in failures:
                    if f["stage"] == stage.name:
                        f["retried"] = True
                all_done = False
    write_status(run_dir, stages, failures)
    return all_done, failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--no-retry", action="store_true")
    ap.add_argument("--max-ticks", type=int, default=0)
    args = ap.parse_args()

    run_dir = REPO / "runs" / args.run
    if not run_dir.exists():
        print(f"[monitor] no run directory {run_dir}", file=sys.stderr)
        return 2

    ticks = 0
    while True:
        done, failures = tick(run_dir, allow_retry=not args.no_retry)
        software = [f for f in failures if f["kind"] == "software"]
        print(
            f"[monitor] tick {ticks}: done={done} failures={len(failures)} "
            f"software_faults={len(software)}",
            flush=True,
        )
        for f in software:
            print(f"[monitor] SOFTWARE FAULT {f['stage']} {f['task']}: {f['evidence']}")
        ticks += 1
        if args.once or done:
            return 1 if software else 0
        if args.max_ticks and ticks >= args.max_ticks:
            return 0
        if not args.watch:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
