#!/bin/bash
# Collect the recorded results into results/<run>/aggregate.json and the summary
# report.
#
#   bash internal/scripts/finalize.sh                                      # final cohort
#   bash internal/scripts/finalize.sh configs/dataset_final_1000.yaml final_1000 test
#
# Safe to re-run: it only reads result files and rewrites the aggregate and the
# summary report.
set -euo pipefail
CONFIG=${1:-configs/dataset_final_1000.yaml}
RUN=${2:-final_1000}
SPLIT=${3:-test}

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source /work/grana_maxillo/lborghi/env.sh
source .venv/bin/activate
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/third_party"
export TORCH_HOME="$REPO_DIR/third_party/torch_home"

python scripts/report_grid.py --config "$CONFIG" --run "$RUN" --split "$SPLIT"
python scripts/aggregate_results.py --run "$RUN" --config "$CONFIG"
python internal/scripts/monitor_slurm.py     --run "$RUN" --once --no-retry || true
