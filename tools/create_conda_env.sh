#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

if conda env list | awk '{print $1}' | grep -qx "das3occ"; then
  echo "[INFO] Conda env 'das3occ' already exists, skipping create."
else
  conda env create -f "${PROJECT_ROOT}/env/environment.yml"
  echo "[OK] Created env 'das3occ'"
fi

conda run -n das3occ pip install --upgrade pip
conda run -n das3occ pip install \
  "torch==1.10.2+cu113" \
  "torchvision==0.11.3+cu113" \
  -f "https://download.pytorch.org/whl/cu113/torch_stable.html"
conda run -n das3occ pip install \
  "mmcv-full==1.4.0" \
  -f "https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html"
conda run -n das3occ pip install "mmdet==2.20.0"
conda run -n das3occ pip install -r "${PROJECT_ROOT}/env/requirements.txt"

echo "[OK] Installed DAOcc-compatible deps in 'das3occ'"
