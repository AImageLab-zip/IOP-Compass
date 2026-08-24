#!/bin/bash
# Submit and run one cohort's entire campaign unattended: training, validation
# eval, the one-time locked test-set eval, and report/paper regeneration.
# Blocks on SLURM completion at each stage (via monitor_slurm.py --watch), so
# nothing needs to be babysat — just wait for this script to exit.
#
#   bash internal/scripts/run_full_experiment.sh configs/dataset_final_1000.yaml final_1000
#
# Aborts (set -e) on the first failed step, including any unresolved SLURM
# software fault, so a broken campaign never silently proceeds to freeze or
# the test-set evaluation.
set -euo pipefail

CONFIG=${1:-configs/dataset_final_1000.yaml}
RUN=${2:-$(python - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["dataset"]["name"])
PY
)}

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "=== $RUN: submitting training + validation campaign ==="
bash internal/scripts/run_all.sh "$CONFIG" "$RUN"

source /work/grana_maxillo/lborghi/env.sh
source .venv/bin/activate
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/third_party"
export TORCH_HOME="$REPO_DIR/third_party/torch_home"

echo "=== $RUN: waiting for training + validation to finish ==="
python internal/scripts/monitor_slurm.py --run "$RUN" --watch

echo "=== $RUN: freezing configuration ==="
python scripts/freeze_experiment.py --config "$CONFIG" --run "$RUN"

echo "=== $RUN: submitting locked test-set evaluation ==="
python internal/scripts/run_final_test.py --config "$CONFIG" --run "$RUN" --final-test \
    --config-hash "$(cat "runs/$RUN/frozen_config.sha256")"

echo "=== $RUN: waiting for test-set evaluation to finish ==="
python internal/scripts/monitor_slurm.py --run "$RUN" --watch

echo "=== $RUN: aggregating results and regenerating figures ==="
bash internal/scripts/finalize.sh "$CONFIG" "$RUN" test

cat <<EOF

=== $RUN: done. Review: ===
  results/$RUN/aggregate.json
  results/$RUN/audit/excluded_files_report.md
EOF
