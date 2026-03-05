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

