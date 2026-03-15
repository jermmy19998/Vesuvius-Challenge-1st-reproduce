# Vesuvius Surface Detection Reproduction

This folder reproduces the solution flow:

- `preprocess.py`: training data processing / preprocessing only (single file)
- `train.py`: training only
- `infer.py`: inference + ensemble + postprocess (single file)
- `configs/train/*.yaml`: training model manifest
- `configs/train/models_active_set2/*.yaml`: one yaml per active model
- `configs/train/models_unused/*.yaml`: one yaml per unused model

Per request:

- Active training: `configs/train/models_active_set2.yaml`
- Unused training: `configs/train/models_unused.yaml`
- Different training models use different yaml files (manifest references per-model yamls)
- Inference uses no yaml (all infer settings come from `infer.py` args/defaults)

## Kaggle Install Fix (acvl-utils / connected-components-3d)

If Kaggle shows:

```text
ERROR: Could not find a version that satisfies the requirement connected-components-3d (from acvl-utils)
ERROR: No matching distribution found for connected-components-3d
```

use this setup cell first (and make sure Notebook Internet is enabled):

```python
import sys
import subprocess


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


run([sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])
run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "--no-cache-dir",
    "--index-url",
    "https://pypi.org/simple",
    "connected-components-3d==3.26.1",
])
run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "--no-cache-dir",
    "--index-url",
    "https://pypi.org/simple",
    "acvl-utils==0.2.5",
    "nnunetv2==2.6.4",
])

import cc3d
print("cc3d ok:", cc3d.__version__)
```
## Train

1. Data processing only:

```bash
python source/preprocess.py \
  --input_dir ./data \
  --working_dir ./work \
  --output_dir ./output \
  --modes 1,2,5,7 \
  --num_workers 8
```

2. Train only:

```bash
python source/train.py \
  --working_dir ./work \
  --output_dir ./output \
  --modes 1,2,5,7
```

## Infer (4-model 7/5/2/1)

Run:

```bash
python source/infer.py \
  --input_dir /kaggle/input/vesuvius-challenge-surface-detection \
  --working_dir /tmp/vesuvius_multi_scratch \
  --output_dir /kaggle/working \
  --models_results_dir /kaggle/input/<your-model-root-or-output-nnunet-results> \
  --active_modes 7,5,2,1 \
  --gpu_ids 0,1
```

## Args Quick Reference

- `preprocess.py`
- `--input_dir`: Dataset root used for preprocessing. Default: `./data`.
- `--working_dir`: Intermediate workspace (nnUNet raw/preprocessed scratch). Default: `./work`.
- `--output_dir`: Output root for generated assets/logs. Default: `./output`.
- `--modes`: Comma-separated mode list to prepare, for example `1,2,5,7`.
- `--num_workers`: Worker count for preprocessing. Default: `8`.

- `train.py`
- `--working_dir`: Preprocessed data workspace. Default: `./work`.
- `--output_dir`: Output root for training artifacts. Default: `./output`.
- `--modes`: Comma-separated mode list to train.
- `--configuration`: nnUNet configuration name. Default: `3d_fullres`.
- `--fold`: Fold selection. Default: `all`.

- `infer.py` (main)
- `--input_dir`: Test data root (expects `test_images`). Default: `./data`.
- `--working_dir`: Inference scratch directory. Default: `./tmp/vesuvius_multi_scratch`.
- `--output_dir`: Final outputs (`submission.zip` and slice PNG grids). Default: `./output`.
- `--models_results_dir`: Model directory. Empty means `<output_dir>/nnUNet_results`.
- `--active_modes`: Inference modes. Default: `7,5,2,1`.
- `--gpu_ids`: GPU list like `0,1`; empty means auto detect.
- `--prob_threshold`: Binary mask threshold. Default: `0.26`.
- `--fusion_scheme`: `DIRECT_WEIGHTED` or `PAIR_ENSEMBLE`.
## Notes

- Inference weights and threshold follow the referenced notebook:
  - active modes: `7,5,2,1`
  - weights: `0.42, 0.18, 0.28, 0.12`
  - threshold: `0.26`
- Fusion behavior can be switched with `--fusion_scheme` (`DIRECT_WEIGHTED` default, or `PAIR_ENSEMBLE`).
- Postprocess parameters follow the same notebook config:
  - `small_obj_thresh=20000`
  - `border_width=3`
  - `max_sheets=40`
  - `max_patch_size=64`
  - `close_par=7`
  - `close_perp=7`
- The final submission file is `<output_dir>/submission.zip`.
- `working_dir` is scratch space for intermediate files (`test_input`, per-mode `npz`, temporary TIFFs).



