# DAS3Occ

`DAS3Occ` (Detection-Assisted Sparse Spatio-Temporal Occupancy) is a new standalone project scaffold designed to merge high-mIoU detection-assisted occupancy ideas (DAOcc) with sparse temporal renovation ideas (STCOcc), while adding explicit hard-negative suppression.

Primary target:
- beat the best mIoU in the local baseline set (DAOcc: **54.33 mIoU** on Occ3D-nuScenes, per its README)

Core design:
- DAOcc-style dual-head training (object detection + occupancy)
- Detection-guided occupancy feature gating
- Lightweight temporal feature memory (sequence-aware EMA fusion)
- Hard-negative suppression outside detector-supported regions
- 8x4090/H200-first distributed workflow, plus 1x4090 long-run fallback

## Project Layout

```text
DAS3Occ/
├── configs/
├── docs/
├── env/
├── src/das3occ/
├── tools/
├── data/
└── work_dirs/
```

## Quick Start

1. Install environment: `docs/install.md`
2. Link dataset with symlinks: `docs/data.md`
3. Launch 8-GPU training/eval: `docs/run.md`
4. Run ablation matrix: `docs/ablation.md`

## Base Project Selection

This project uses **DAOcc-style config/runtime** as the base because it has the strongest reported mIoU among the listed local repos, and extends it with STCOcc-inspired temporal/sparse ideas.

## Notes

- Existing repos on this machine are treated as read-only references.
- New code/config/docs are created only inside this `DAS3Occ` directory.
