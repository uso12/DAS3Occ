#!/usr/bin/env bash
set -euo pipefail

GPUS=${1:-8}
RUN_ROOT=${2:-work_dirs/ablation_matrix}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
PORT_BASE=${PORT_BASE:-29600}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
CONFIG_ROOT="${PROJECT_ROOT}/configs/nuscenes/occ3d/ablations"

EXTRA_ARGS=("${@:3}")

NAMES=(baseline guidance_only full)
CONFIGS=(
  "${CONFIG_ROOT}/das3occ_occ3d_nus_ablation_baseline.yaml"
  "${CONFIG_ROOT}/das3occ_occ3d_nus_ablation_guidance_only.yaml"
  "${CONFIG_ROOT}/das3occ_occ3d_nus_ablation_full.yaml"
)

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  cfg="${CONFIGS[$i]}"
  run_dir="${RUN_ROOT}/${name}"
  port=$((PORT_BASE + i))

  echo "[DAS3Occ][Ablation] start ${name}"
  echo "  config: ${cfg}"
  echo "  run_dir: ${run_dir}"
  echo "  gpus: ${GPUS}, nnodes: ${NNODES}, node_rank: ${NODE_RANK}, port: ${port}"

  NNODES="${NNODES}" NODE_RANK="${NODE_RANK}" MASTER_ADDR="${MASTER_ADDR}" PORT="${port}" \
    bash "${SCRIPT_DIR}/dist_train.sh" "${cfg}" "${GPUS}" --run-dir "${run_dir}" "${EXTRA_ARGS[@]}"
done
