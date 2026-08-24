#!/bin/bash
# Create the IOP-Compass python environment with uv.
# Usage: bash internal/scripts/setup_env.sh
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source /work/grana_maxillo/lborghi/env.sh

uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  "torch==2.7.1+cu126" "torchvision==0.22.1+cu126"

uv pip install \
  "numpy<2" \
  "opencv-python-headless>=4.11" \
  "timm<1" \
  "ultralytics>=8.3.116" \
  "segment-anything-hq==0.3" \
  "scipy" "pandas" "matplotlib" "pyyaml" "tqdm" "Pillow" \
  "einops" "ftfy" "regex" "hydra-core" "omegaconf" "iopath" "huggingface_hub" \
  "pytest" "scikit-image" "pycocotools" "tabulate"

uv pip freeze > requirements-lock.txt
python -c "import torch, torchvision; print('torch', torch.__version__, 'tv', torchvision.__version__, 'cuda', torch.version.cuda)"
echo "SETUP_ENV_OK"
