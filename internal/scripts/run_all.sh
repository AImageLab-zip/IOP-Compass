#!/bin/bash
# Submit the entire campaign for one cohort, then the locked test-set evaluation.
#
#   bash internal/scripts/run_all.sh                                    # final cohort
#   bash internal/scripts/run_all.sh configs/dataset_final_1000.yaml final_1000
#
# Everything up to and including the validation results is one dependency-chained
# submission; nothing has to be babysat. The test set is deliberately a second,
# explicit step, printed at the end, because it must be evaluated exactly once and
# only after the configuration is frozen.
set -euo pipefail

CONFIG=${1:-configs/dataset_final_1000.yaml}
RUN=${2:-$(python - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["dataset"]["name"])
PY
)}

# Benchmark-grid replicates.  Each rotates the validation/test hold-out and trains its
# own Mask R-CNN, so they are independent re-measurements; the rotation only fits while
# replicates * (n_val + n_test) <= eligible patients, which the audit stage enforces.
REPLICATES=${3:-1,2,3}

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source /work/grana_maxillo/lborghi/env.sh
source .venv/bin/activate
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/third_party"
export TORCH_HOME="$REPO_DIR/third_party/torch_home"

echo "=== config: $CONFIG    run: $RUN    replicates: $REPLICATES ==="
python scripts/fetch_third_party.py --record "runs/$RUN/third_party.json"

echo
echo "=== submitting the campaign (validation only; the test set stays locked) ==="
python internal/scripts/submit_pipeline.py --config "$CONFIG" --run-name "$RUN" \
    --stages all --split val --replicates "$REPLICATES"

cat <<EOF

=== submitted ===
Watch it:      python internal/scripts/monitor_slurm.py --run $RUN --watch
Live status:   runs/$RUN/STATUS_slurm.md      (failures are listed at the end)
Validation:    results/$RUN/aggregate.json    (written by the aggregate stage)

When every job is COMPLETED and you are satisfied with the validation numbers,
freeze and evaluate the held-out test split exactly once:

  python scripts/freeze_experiment.py --config $CONFIG --run $RUN
  python internal/scripts/run_final_test.py --config $CONFIG --run $RUN --final-test \\
      --config-hash \$(cat runs/$RUN/frozen_config.sha256)
  python internal/scripts/monitor_slurm.py --run $RUN --watch

Then regenerate everything from the test-split numbers with one command:

  bash internal/scripts/finalize.sh $CONFIG $RUN test
EOF
