# Vesuvius Surface Detection Reproduction

This folder reproduces the solution flow:

- `preprocess.py`: training data processing / preprocessing only (single file)
- `train.py`: training only
- `infer.py`: inference + ensemble + postprocess (single file)
- `configs/train/*.yaml`: training model manifest
- `configs/train/models_active_set2/*.yaml`: one yaml per active model

Per request:

- Active training: `configs/train/models_active_set2.yaml`
- Different active training models use different yaml files (manifest references per-model yamls)
- Inference uses no yaml (all infer settings come from `infer.py` args/defaults)

## References

- Solution reference: https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/1st-place-solution-for-the-vesuvius-challenge-su
- Notebook reference: https://www.kaggle.com/code/tonylica/nnunet-4-model-7-5-2-1-final-submit-so-long?scriptVersionId=300373304

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
  --modes 1 \
  --gpu_ids 0 \
  --swanlab
```

3. Multi-process launch (one mode per process):

```bash
nohup python ./source/train.py --modes 1 --gpu_ids 0 --swanlab > ./source/train_m1.log 2>&1 &
nohup python ./source/train.py --modes 2 --gpu_ids 1 --swanlab > ./source/train_m2.log 2>&1 &
nohup python ./source/train.py --modes 5 --gpu_ids 2 --swanlab > ./source/train_m5.log 2>&1 &
nohup python ./source/train.py --modes 7 --gpu_ids 3 --swanlab > ./source/train_m7.log 2>&1 &
```

`mode 2` and `mode 5` require pretrained checkpoints from `mode 1`.

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
- `--modes`: Single mode per run, for example `1`.
- `--gpu_ids`: Single GPU id per run, for example `0`.
- `--configuration`: nnUNet configuration name. Default: `3d_fullres`.
- `--fold`: Fold selection. Default: `all`.
- `--swanlab`: Enable SwanLab logging for loss/metric/lr.
- `--swanlab_run_prefix`: Optional SwanLab name prefix. Default: empty.

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

- Inference weights and threshold follow the notebook reference above:
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


