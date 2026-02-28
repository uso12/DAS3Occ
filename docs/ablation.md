# Ablation Matrix

This matrix isolates the three core DAS3Occ additions:

- Detection guidance (`model.use_detection_guidance`)
- Temporal memory (`model.heads.occ.use_temporal_memory`)
- Hard-negative suppression (`model.heads.occ.hard_negative_weight`)

## Configs

- `configs/nuscenes/occ3d/ablations/das3occ_occ3d_nus_ablation_baseline.yaml`
  - guidance: off, temporal memory: off, hard-negative suppression: off
- `configs/nuscenes/occ3d/ablations/das3occ_occ3d_nus_ablation_guidance_only.yaml`
  - guidance: on, temporal memory: off, hard-negative suppression: off
- `configs/nuscenes/occ3d/ablations/das3occ_occ3d_nus_ablation_full.yaml`
  - guidance: on, temporal memory: on, hard-negative suppression: on

## Run All Ablations (8 GPUs)

```bash
cd /home/ruiyu12/DAS3Occ
conda activate das3occ
export DAOCC_ROOT=/home/ruiyu12/DAOcc

bash tools/run_ablation_matrix.sh 8 work_dirs/ablation_matrix_8gpu
```

## Run All Ablations (1 GPU long-run)

```bash
bash tools/run_ablation_matrix.sh 1 work_dirs/ablation_matrix_1gpu_long
```

## Multi-node

`tools/run_ablation_matrix.sh` forwards distributed knobs to `dist_train.sh`:

- `NNODES`
- `NODE_RANK`
- `MASTER_ADDR`
- `PORT_BASE`

Example:

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 PORT_BASE=29700 \
  bash tools/run_ablation_matrix.sh 8 work_dirs/ablation_matrix_2node
```

## Evaluate Each Ablation

```bash
bash tools/dist_test.sh configs/nuscenes/occ3d/ablations/das3occ_occ3d_nus_ablation_baseline.yaml \
  work_dirs/ablation_matrix_8gpu/baseline/latest.pth 8

bash tools/dist_test.sh configs/nuscenes/occ3d/ablations/das3occ_occ3d_nus_ablation_guidance_only.yaml \
  work_dirs/ablation_matrix_8gpu/guidance_only/latest.pth 8

bash tools/dist_test.sh configs/nuscenes/occ3d/ablations/das3occ_occ3d_nus_ablation_full.yaml \
  work_dirs/ablation_matrix_8gpu/full/latest.pth 8
```
