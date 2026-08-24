#!/bin/bash
# Generic IOP-Compass SLURM worker.
#
# The job body is one line of a task list; SLURM_ARRAY_TASK_ID (default 0)
# selects it.  Everything about the job - partition, memory, GPU, time - is set
# by the submitting process, so this file never needs editing.
#
# Required environment:
#   IOPC_TASKLIST   path to a file with one shell command per line
# Optional:
#   IOPC_RUN_DIR    run directory for provenance (default runs/<jobname>)
#   IOPC_GPU        1 when the task needs CUDA (default: auto from SLURM_JOB_GRES)
set -uo pipefail

REPO_DIR=${IOPC_REPO_DIR:-/work/grana_maxillo/lborghi/code/IOP-Compass}
cd "$REPO_DIR"

source /work/grana_maxillo/lborghi/env.sh
source "$REPO_DIR/.venv/bin/activate"

# Never let a job phone home with clinical data.
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$REPO_DIR/third_party/torch_home"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/third_party${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

: "${IOPC_TASKLIST:?IOPC_TASKLIST must point at a task list file}"
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
N_TASKS=$(grep -cve '^\s*$' "$IOPC_TASKLIST")
if (( TASK_ID < 0 || TASK_ID >= N_TASKS )); then
  echo "SLURM_ARRAY_TASK_ID=$TASK_ID out of range 0..$((N_TASKS-1))" >&2
  exit 2
fi
CMD=$(grep -ve '^\s*$' "$IOPC_TASKLIST" | sed -n "$((TASK_ID+1))p")

RUN_DIR=${IOPC_RUN_DIR:-$REPO_DIR/runs/${SLURM_JOB_NAME:-local}}
PROV_DIR="$RUN_DIR/slurm/provenance"
mkdir -p "$PROV_DIR"
STAMP="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${TASK_ID}"
PROV="$PROV_DIR/$STAMP.json"

# git may be unavailable on a compute node, so the submitting process records the
# commit in IOPC_GIT_COMMIT / IOPC_GIT_DIRTY and this is only the fallback.
GIT_COMMIT=${IOPC_GIT_COMMIT:-$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)}
GIT_DIRTY=${IOPC_GIT_DIRTY:-$(git -C "$REPO_DIR" status --porcelain 2>/dev/null | wc -l)}
START_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
START_EPOCH=$(date +%s)

echo "=== IOP-Compass worker ==="
echo "job=${SLURM_JOB_ID:-local} array=${SLURM_ARRAY_JOB_ID:-none} task=$TASK_ID"
echo "host=$(hostname) commit=$GIT_COMMIT dirty=$GIT_DIRTY"
echo "cmd=$CMD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

set +e
bash -c "$CMD"
EXIT_CODE=$?
set -e 2>/dev/null || true

END_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ELAPSED=$(( $(date +%s) - START_EPOCH ))

python - "$PROV" <<PYEOF
import json, os, platform, sys
payload = {
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "job_name": os.environ.get("SLURM_JOB_NAME"),
    "partition": os.environ.get("SLURM_JOB_PARTITION"),
    "account": os.environ.get("SLURM_JOB_ACCOUNT"),
    "hostname": platform.node(),
    "python": platform.python_version(),
    "git_commit": "$GIT_COMMIT",
    "git_dirty_files": int("$GIT_DIRTY"),
    "command": """$CMD""",
    "exit_code": int("$EXIT_CODE"),
    "start_utc": "$START_ISO",
    "end_utc": "$END_ISO",
    "elapsed_s": int("$ELAPSED"),
    "tasklist": os.environ.get("IOPC_TASKLIST"),
}
try:
    import torch
    payload["torch"] = torch.__version__
    payload["cuda"] = torch.version.cuda
    payload["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception as exc:
    payload["torch_error"] = str(exc)
with open(sys.argv[1], "w") as fh:
    json.dump(payload, fh, indent=1)
PYEOF

echo "=== finished task=$TASK_ID exit=$EXIT_CODE elapsed=${ELAPSED}s ==="
exit $EXIT_CODE
