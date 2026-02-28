# Installation

## 1. Create Environment and Install Dependencies

`DAS3Occ` uses a DAOcc-compatible stack:
- `torch==1.10.2+cu113`
- `mmcv-full==1.4.0`
- `mmdet==2.20.0`

```bash
cd /home/ruiyu12/DAS3Occ
bash tools/create_conda_env.sh
conda activate das3occ
```

## 2. Set Compiler/CUDA Build Env

For CUDA extension compatibility on Ada/Hopper machines with this stack:

```bash
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export TORCH_CUDA_ARCH_LIST=8.6+PTX
```

`tools/train.py` and `tools/test.py` enforce these values internally as well.

## 3. Set Base Repository Path

`DAS3Occ` loads runtime components from your DAOcc checkout via `DAOCC_ROOT`.

```bash
export DAOCC_ROOT=/home/ruiyu12/DAOcc
```

## 4. Verify Entry Points

```bash
python tools/train.py -h
python tools/test.py -h
```
