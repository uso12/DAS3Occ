# Data Preparation and Symlink Workflow

This project follows DAOcc-style external dataset symlinks.

## Expected external root

Your external root should contain a `nuscenes` directory and optionally top-level `gts` / info pkl files.

Example:

```text
<EXTERNAL_ROOT>/
├── nuscenes/
│   ├── samples/
│   ├── sweeps/
│   ├── maps/
│   ├── v1.0-trainval/
│   ├── gts/
│   ├── nuscenes_infos_train_w_3occ.pkl
│   └── nuscenes_infos_val_w_3occ.pkl
├── gts/  # optional alternative location
├── nuscenes_infos_train_w_3occ.pkl  # optional alternative location
└── nuscenes_infos_val_w_3occ.pkl
```

## Create project-local symlinks

```bash
cd /home/ruiyu12/DAS3Occ
bash tools/link_data.sh /path/to/external_root
```

This creates/updates symlinks under `DAS3Occ/data/` without modifying source datasets.

## Quick check

```bash
ls -la data
ls -la data/nuscenes | head
```
