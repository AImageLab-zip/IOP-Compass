#!/bin/bash
# Lightweight SLURM driver for run_full_experiment.sh, so it survives past
# this cluster's 4h interactive/command limit. Requests no GPU and minimal
# CPU/memory itself: this job only submits and waits on the real training/
# eval jobs (via monitor_slurm.py --watch), it never runs compute directly.
#
#   sbatch internal/scripts/run_full_experiment.sbatch configs/dataset_final_1000.yaml final_1000
#
# all_serial (no-GPU partition) caps walltime at 4h, too short for a run that
# waits out the whole campaign, so this stays on all_usr_prod with no --gres:
# re-verified on this cluster that all_usr_prod jobs run fine without a GPU
# reservation (see internal/scripts/slurm_util.py).
#SBATCH --job-name=iopc_driver
#SBATCH --account=grana_maxillo
#SBATCH --partition=all_usr_prod
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_driver_logs/%x_%j.out
#SBATCH --error=slurm_driver_logs/%x_%j.err
set -euo pipefail

REPO_DIR="/work/grana_maxillo/lborghi/code/IOP-Compass"
cd "$REPO_DIR"

bash internal/scripts/run_full_experiment.sh "$1" "$2"
