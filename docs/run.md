# Training and Evaluation

## 8-GPU Training (4090/H200-grade setup)

```bash
cd /home/ruiyu12/DAS3Occ
conda activate das3occ
export DAOCC_ROOT=/home/ruiyu12/DAOcc

bash tools/dist_train.sh configs/nuscenes/occ3d/das3occ_occ3d_nus_w_mask.yaml 8 \
  --run-dir work_dirs/das3occ_occ3d_8gpu
```

## 1-GPU Long Training (fallback)

```bash
bash tools/dist_train.sh configs/nuscenes/occ3d/das3occ_occ3d_nus_w_mask_1gpu_long.yaml 1 \
  --run-dir work_dirs/das3occ_occ3d_1gpu_long
```

## 8-GPU Evaluation

```bash
bash tools/dist_test.sh configs/nuscenes/occ3d/das3occ_occ3d_nus_w_mask.yaml \
  /path/to/checkpoint.pth 8
```

## Single-GPU Evaluation

```bash
python tools/test.py configs/nuscenes/occ3d/das3occ_occ3d_nus_w_mask.yaml \
  /path/to/checkpoint.pth --launcher none --eval bbox
```

## Multi-node knobs (optional)

`tools/dist_train.sh` and `tools/dist_test.sh` support:
- `NNODES`
- `NODE_RANK`
- `MASTER_ADDR`
- `PORT`

Example:

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 PORT=29501 bash tools/dist_train.sh ... 8
```

## Ablation Matrix

Run all three ablations (baseline / guidance-only / full):

```bash
bash tools/run_ablation_matrix.sh 8 work_dirs/ablation_matrix_8gpu
```

See `docs/ablation.md` for full matrix details and evaluation commands.
