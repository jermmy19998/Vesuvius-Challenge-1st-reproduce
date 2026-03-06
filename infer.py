import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import morphology as sk_morphology
from skimage.measure import euler_number
from tqdm.auto import tqdm

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from numba import jit
except Exception:  # pragma: no cover
    def jit(*args, **kwargs):
        def _inner(func):
            return func

        return _inner

try:
    import cupy
    import cupyx
except Exception:  # pragma: no cover
    cupy = None
    cupyx = None


STRUCT_3X3X3 = np.ones((3, 3, 3), dtype=np.uint8)


def _in_notebook_runtime() -> bool:
    return ("ipykernel" in sys.modules) or bool(os.environ.get("JPY_PARENT_PID"))


def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser("Vesuvius Surface Detection - Inference Reproduction")
    parser.add_argument("--input_dir", type=str, default="/kaggle/input/vesuvius-challenge-surface-detection")
    parser.add_argument(
        "--working_dir",
        type=str,
        default="/kaggle/working",
        help="Unified output root. Stores temporary inference files, debug artifacts, PNG grids, and submission.zip.",
    )
    parser.add_argument(
        "--weight_dir",
        type=str,
        default="/kaggle/input/datasets/seeingtimes/1st-reproduce-test",
        help="Directory that stores trained nnUNet model folders (kept separate from temp/output).",
    )
    parser.add_argument(
        "--models_results_dir",
        type=str,
        default="",
        help="Deprecated alias of --weight_dir. If set, it overrides --weight_dir.",
    )
    parser.add_argument(
        "--active_modes",
        type=str,
        default="7,5,2,1",
        help="Comma-separated model modes to run. Example: 7,5,2,1",
    )
    parser.add_argument(
        "--fusion_scheme",
        type=str,
        choices=["DIRECT_WEIGHTED", "PAIR_ENSEMBLE"],
        default="DIRECT_WEIGHTED",
    )
    parser.add_argument("--configuration", type=str, default="3d_fullres")
    parser.add_argument("--fold", type=str, default="all")
    parser.add_argument("--npp", type=int, default=1)
    parser.add_argument("--nps", type=int, default=1)
    parser.add_argument("--prob_threshold", type=float, default=0.26)
    parser.add_argument("--process_in_chunks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cases_per_chunk", type=int, default=32)
    parser.add_argument("--zip_tiff_immediately", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--emulate_float16_maps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_cupy_postprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean_run_caches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean_nnunet_preprocessed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean_nnunet_results", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug_disk_report", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disk_report_include_working", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zip_name", type=str, default="submission.zip")
    parser.add_argument("--small_obj_thresh", type=int, default=20000)
    parser.add_argument("--border_width", type=int, default=3)
    parser.add_argument("--max_sheets", type=int, default=40)
    parser.add_argument("--max_patch_size", type=int, default=64)
    parser.add_argument("--close_par", type=int, default=7)
    parser.add_argument("--close_perp", type=int, default=7)
    parser.add_argument("--pair54_w5", type=float, default=0.7)
    parser.add_argument("--pair54_w4", type=float, default=0.3)
    parser.add_argument("--pair21_w2", type=float, default=0.7)
    parser.add_argument("--pair21_w1", type=float, default=0.3)
    parser.add_argument("--final_w54", type=float, default=0.6)
    parser.add_argument("--final_w21", type=float, default=0.4)
    parser.add_argument("--gpu_ids", type=str, default="", help='GPU ids like "0,1". Empty means auto-detect.')
    if _in_notebook_runtime():
        args, _ = parser.parse_known_args(args=[] if argv is None else argv)
        return args
    return parser.parse_args(args=argv)


def log(msg: str):
    import time

    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def banner(title: str):
    log("=" * 60)
    log(title)
    log("=" * 60)

def create_spacing_json(path: Path, spacing=(1.0, 1.0, 1.0)):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"spacing": list(spacing)}, f)


def _normalize_for_display(slice_2d: np.ndarray) -> np.ndarray:
    arr = np.asarray(slice_2d, dtype=np.float32)
    min_v = float(np.nanmin(arr))
    max_v = float(np.nanmax(arr))
    if max_v <= min_v:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_v) / (max_v - min_v)


def show_random_slices_grid(original_volume: np.ndarray, mask_volume: np.ndarray, case_id: str, save_png_path: Optional[Path] = None):
    if plt is None:
        log("Skip visualization: matplotlib is not available.")
        return

    if original_volume.ndim != 3 or mask_volume.ndim != 3:
        log(f"Skip visualization for {case_id}: expected 3D volumes.")
        return

    depth = int(min(original_volume.shape[0], mask_volume.shape[0]))
    if depth <= 0:
        log(f"Skip visualization for {case_id}: empty volume.")
        return

    pick_count = 3
    if depth >= pick_count:
        selected_slices = sorted(random.sample(range(depth), k=pick_count))
    else:
        selected_slices = sorted(random.choices(range(depth), k=pick_count))

    fig, axes = plt.subplots(pick_count, 3, figsize=(12, 4 * pick_count))
    axes = np.asarray(axes).reshape(pick_count, 3)

    column_titles = ["Original", "Mask", "Overlay"]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=12)

    for row, z_idx in enumerate(selected_slices):
        original_slice = _normalize_for_display(original_volume[z_idx])
        mask_slice = (mask_volume[z_idx] > 0).astype(np.float32)
        overlay = np.stack([original_slice, original_slice, original_slice], axis=-1)
        overlay[..., 0] = np.where(mask_slice > 0, 1.0, overlay[..., 0])
        overlay[..., 1] = np.where(mask_slice > 0, overlay[..., 1] * 0.35, overlay[..., 1])
        overlay[..., 2] = np.where(mask_slice > 0, overlay[..., 2] * 0.35, overlay[..., 2])

        axes[row, 0].imshow(original_slice, cmap="gray")
        axes[row, 1].imshow(mask_slice, cmap="gray")
        axes[row, 2].imshow(overlay)
        axes[row, 0].set_ylabel(f"Slice {z_idx}", fontsize=11)
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
        axes[row, 2].axis("off")

    fig.suptitle(f"Random Slice Visualization - Case {case_id}", fontsize=14)
    plt.tight_layout()
    if save_png_path is not None:
        save_png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_png_path, dpi=200, bbox_inches="tight")
        log(f"Saved slice grid: {save_png_path}")
    plt.show()


def _sh(cmd: str, cwd: Optional[Path] = None) -> str:
    p = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd is not None else None,
    )
    out = (p.stdout or "") + (p.stderr or "")
    return out.strip()


def setup_cache_and_tmp_env(working_dir: Path) -> tuple[Path, Path]:
    cache_root = working_dir / "_cache"
    tmp_root = working_dir / "_tmp"
    cache_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    os.environ["PIP_CACHE_DIR"] = str(cache_root / "pip")
    os.environ["NUMBA_CACHE_DIR"] = str(cache_root / "numba")
    os.environ["CUPY_CACHE_DIR"] = str(cache_root / "cupy")
    os.environ["TORCH_HOME"] = str(cache_root / "torch")
    os.environ["MPLCONFIGDIR"] = str(cache_root / "mpl")

    os.environ["TMPDIR"] = str(tmp_root)
    os.environ["TEMP"] = str(tmp_root)
    os.environ["TMP"] = str(tmp_root)
    os.environ["PIP_NO_CACHE_DIR"] = "1"
    return cache_root, tmp_root


def disk_report(
    tag: str,
    enabled: bool,
    include_working: bool,
    working_dir: Path,
    output_dir: Path,
    nnunet_preprocessed: Path,
    nnunet_results: Path,
    cache_root: Path,
):
    if not enabled:
        return
    log("=" * 60)
    log(f"DISK REPORT: {tag}")
    log("=" * 60)
    try:
        du_tmp = shutil.disk_usage("/tmp")
        log(
            f"/tmp total={du_tmp.total / 1024**3:.2f}G "
            f"used={du_tmp.used / 1024**3:.2f}G free={du_tmp.free / 1024**3:.2f}G"
        )
    except Exception as e:
        log(f"disk_usage(/tmp) failed: {e!r}")

    if include_working:
        try:
            du_out = shutil.disk_usage(str(output_dir))
            log(
                f"{output_dir} total={du_out.total / 1024**3:.2f}G "
                f"used={du_out.used / 1024**3:.2f}G free={du_out.free / 1024**3:.2f}G"
            )
        except Exception as e:
            log(f"disk_usage({output_dir}) failed: {e!r}")

    for cmd in (
        "df -h / /tmp || true",
        f'df -h "{output_dir}" || true',
        f'du -sh "{working_dir}" 2>/dev/null || true',
        f'du -sh "{nnunet_preprocessed}" 2>/dev/null || true',
        f'du -sh "{nnunet_results}" 2>/dev/null || true',
        f'du -sh "{cache_root}" 2>/dev/null || true',
    ):
        out = _sh(cmd, cwd=working_dir)
        if out:
            for line in out.splitlines():
                log(line)


@dataclass
class NnunetPaths:
    base: Path
    raw: Path
    preprocessed: Path
    results: Path


def setup_nnunet_environment(working_dir: Path, compile_flag: str = "false") -> NnunetPaths:
    base = working_dir / "nnUNet_data"
    raw = base / "nnUNet_raw"
    preprocessed = base / "nnUNet_preprocessed"
    results = working_dir / "nnUNet_results"
    for d in (base, raw, preprocessed, results):
        d.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(raw)
    os.environ["nnUNet_preprocessed"] = str(preprocessed)
    os.environ["nnUNet_results"] = str(results)
    os.environ["nnUNet_compile"] = compile_flag
    os.environ.setdefault("nnUNet_USE_BLOSC2", "1")

    log(f"nnUNet_raw={raw}")
    log(f"nnUNet_preprocessed={preprocessed}")
    log(f"nnUNet_results={results}")
    return NnunetPaths(base=base, raw=raw, preprocessed=preprocessed, results=results)


def resolve_command(cmd_name: str) -> str:
    direct = shutil.which(cmd_name)
    if direct:
        return direct
    py_bin = Path(sys.executable).resolve().parent
    for c in (py_bin / cmd_name, py_bin / f"{cmd_name}.exe"):
        if c.exists():
            return str(c)
    raise FileNotFoundError(f"{cmd_name} not found. Current python: {sys.executable}. Install nnunetv2.")


def symlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


@dataclass
class InferModelSpec:
    mode: int
    tag: str
    plans_name: str
    trainer: str
    weight: float
    ckpt_name: str
    source_path: Path

    @property
    def dataset_id(self) -> int:
        return 100 + int(self.mode)

    @property
    def dataset_name(self) -> str:
        return f"Dataset{self.dataset_id:03d}_VesuviusSurface_M{self.mode}"

    @property
    def model_name(self) -> str:
        return f"M{self.mode}_{self.tag}"


def _model_source_path(
    models_results_root: Path,
    mode: int,
    trainer: str,
    plans_name: str,
    configuration: str,
) -> Path:
    dataset_name = f"Dataset{100 + mode:03d}_VesuviusSurface_M{mode}"
    return models_results_root / dataset_name / f"{trainer}__{plans_name}__{configuration}"


def build_default_infer_specs(models_results_root: Path, configuration: str) -> dict[int, InferModelSpec]:
    # Model settings follow the referenced notebook setup.
    models = [
        {
            "mode": 1,
            "tag": "MPlans_4000_patch128",
            "plans_name": "nnUNetResEncUNetMPlans",
            "trainer": "nnUNetTrainer_4000epochs",
            "weight": 0.12,
            "ckpt_name": "checkpoint_final.pth",
        },
        {
            "mode": 2,
            "tag": "XLPlans_250_patch192_best",
            "plans_name": "nnUNetResEncUNetXLPlans",
            "trainer": "nnUNetTrainer_250epochs",
            "weight": 0.28,
            "ckpt_name": "checkpoint_best.pth",
        },
        {
            "mode": 3,
            "tag": "XLPlans_250_patch192_final",
            "plans_name": "nnUNetResEncUNetXLPlans",
            "trainer": "nnUNetTrainer_250epochs",
            "weight": 0.0,
            "ckpt_name": "checkpoint_final.pth",
        },
        {
            "mode": 4,
            "tag": "XLPlans_250_patch256_best",
            "plans_name": "nnUNetResEncUNetXLPlans",
            "trainer": "nnUNetTrainer_250epochs",
            "weight": 0.0,
            "ckpt_name": "checkpoint_best.pth",
        },
        {
            "mode": 5,
            "tag": "XLPlans_250_patch256_final",
            "plans_name": "nnUNetResEncUNetXLPlans",
            "trainer": "nnUNetTrainer_250epochs",
            "weight": 0.18,
            "ckpt_name": "checkpoint_final.pth",
        },
        {
            "mode": 6,
            "tag": "XLPlans_500_patch288_E478",
            "plans_name": "nnUNetResEncUNetXLPlans",
            "trainer": "nnUNetTrainer_500epochs",
            "weight": 0.0,
            "ckpt_name": "checkpoint_478e.pth",
        },
        {
            "mode": 7,
            "tag": "LPlans_4000_patch192_fromscratch_final",
            "plans_name": "nnUNetResEncUNetLPlans",
            "trainer": "nnUNetTrainer_4000epochs",
            "weight": 0.42,
            "ckpt_name": "checkpoint_final.pth",
        },
        {
            "mode": 8,
            "tag": "LPlans_4000_patch224_fromscratch_best",
            "plans_name": "nnUNetResEncUNetLPlans",
            "trainer": "nnUNetTrainer_4000epochs",
            "weight": 0.0,
            "ckpt_name": "checkpoint_best.pth",
        },
        {
            "mode": 9,
            "tag": "LPlans_4000_patch160_fromscratch_best",
            "plans_name": "nnUNetResEncUNetLPlans",
            "trainer": "nnUNetTrainer_4000epochs",
            "weight": 0.0,
            "ckpt_name": "checkpoint_best.pth",
        },
        {
            "mode": 10,
            "tag": "LPlans_4000_patch224_fromscratch_final",
            "plans_name": "nnUNetResEncUNetLPlans",
            "trainer": "nnUNetTrainer_4000epochs",
            "weight": 0.0,
            "ckpt_name": "checkpoint_final.pth",
        },
    ]

    specs: dict[int, InferModelSpec] = {}
    for m in models:
        spec = InferModelSpec(
            mode=int(m["mode"]),
            tag=str(m["tag"]),
            plans_name=str(m["plans_name"]),
            trainer=str(m["trainer"]),
            weight=float(m["weight"]),
            ckpt_name=str(m["ckpt_name"]),
            source_path=_model_source_path(
                models_results_root=models_results_root,
                mode=int(m["mode"]),
                trainer=str(m["trainer"]),
                plans_name=str(m["plans_name"]),
                configuration=configuration,
            ),
        )
        specs[spec.mode] = spec
    return specs


def _resolve_ckpt(meta_src: Path, ckpt_name: str) -> Path:
    candidates = [meta_src / "fold_all" / ckpt_name, meta_src / ckpt_name]
    for p in candidates:
        if p.exists():
            return p
    hits = sorted(meta_src.rglob("checkpoint_*.pth"))
    if hits:
        for p in hits:
            if p.name == "checkpoint_final.pth":
                return p
        return hits[0]
    raise FileNotFoundError(f"no checkpoint found under {meta_src} (ckpt_name={ckpt_name})")


def register_model(spec: InferModelSpec, nnunet_results: Path, configuration: str):
    meta_src = spec.source_path.resolve()
    if not meta_src.exists():
        raise FileNotFoundError(f"model source path not found: {meta_src}")

    model_root = nnunet_results / spec.dataset_name / f"{spec.trainer}__{spec.plans_name}__{configuration}"
    fold_dir = model_root / "fold_all"
    fold_dir.mkdir(parents=True, exist_ok=True)

    ckpt_src = _resolve_ckpt(meta_src, spec.ckpt_name)
    ckpt_dst = fold_dir / "checkpoint_final.pth"
    symlink_or_copy(ckpt_src, ckpt_dst)

    for fname in ("dataset.json", "plans.json", "dataset_fingerprint.json"):
        src = meta_src / fname
        dst = model_root / fname
        if not src.exists():
            raise FileNotFoundError(f"missing {fname} at {src}")
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)

    resolved = ckpt_dst.resolve()
    log(f"Registered {spec.model_name}")
    log(f"  model_root: {model_root}")
    log(f"  ckpt_src : {ckpt_src} (size={ckpt_src.stat().st_size / 1024 / 1024:.1f} MB)")
    log(f"  ckpt_link: {ckpt_dst} -> {resolved}")


def list_all_test_tifs(input_dir: Path) -> list[Path]:
    if input_dir.name == "test_images" and input_dir.is_dir():
        test_images_dir = input_dir
    elif (input_dir / "test_images").is_dir():
        test_images_dir = input_dir / "test_images"
    else:
        candidates = sorted(p for p in input_dir.rglob("test_images") if p.is_dir())
        if len(candidates) == 1:
            test_images_dir = candidates[0]
        else:
            raise RuntimeError(
                f"cannot resolve test_images folder from input_dir={input_dir}. "
                "Please pass dataset root containing test_images/ or pass test_images directly."
            )
    log(f"resolved test_images_dir={test_images_dir}")
    tifs = sorted(test_images_dir.glob("*.tif"))
    log(f"found test cases: {len(tifs)}")
    if not tifs:
        raise RuntimeError(f"no test tif found under {test_images_dir}")
    return tifs


def prepare_test_data_subset(tifs: list[Path], test_input_dir: Path) -> list[str]:
    if test_input_dir.exists():
        shutil.rmtree(test_input_dir, ignore_errors=True)
    test_input_dir.mkdir(parents=True, exist_ok=True)

    case_ids: list[str] = []
    for p in tqdm(tifs, desc="Preparing test data"):
        case_id = p.stem
        case_ids.append(case_id)
        img_dst = test_input_dir / f"{case_id}_0000.tif"
        symlink_or_copy(p, img_dst)
        create_spacing_json(test_input_dir / f"{case_id}_0000.json")
        create_spacing_json(test_input_dir / f"{case_id}.json")
    return case_ids


@lru_cache(maxsize=1)
def detect_save_prob_flag() -> str:
    exe = resolve_command("nnUNetv2_predict")
    proc = subprocess.run([exe, "-h"], capture_output=True, text=True)
    txt = (proc.stdout or "") + "\n" + (proc.stderr or "")
    candidates = [
        "--save_probabilities",
        "--save_probabilities_in_separate_folder",
        "--save_probabilities_in_npz",
        "--export_probabilities",
    ]
    for c in candidates:
        if c in txt:
            log(f"Detected probability flag: {c}")
            return c
    m = re.search(r"--save[_-]prob\w+", txt)
    if m:
        flag = m.group(0)
        log(f"Detected probability flag (regex): {flag}")
        return flag
    raise RuntimeError("could not find probability-saving flag in nnUNetv2_predict -h")


def _popen(cmd: str, cwd: Path, env: Optional[dict] = None) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(cwd),
        env=env,
    )


def _stream_process_output_async(p: subprocess.Popen, prefix: str) -> threading.Thread:
    assert p.stdout is not None

    def _worker():
        for line in p.stdout:
            print(f"[{prefix}] {line.rstrip()}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


def _build_predict_cmd(
    spec: InferModelSpec,
    test_input_dir: Path,
    out_dir: Path,
    configuration: str,
    fold: str,
    npp: int,
    nps: int,
    save_prob_flag: str,
) -> str:
    exe = resolve_command("nnUNetv2_predict")
    return (
        f'"{exe}" '
        f"-d {spec.dataset_id:03d} -c {configuration} -f {fold} "
        f"-i {test_input_dir} -o {out_dir} "
        f"-p {spec.plans_name} -tr {spec.trainer} "
        f"-npp {npp} -nps {nps} --verbose {save_prob_flag}"
    )


def _run_one_model(
    spec: InferModelSpec,
    out_dir: Path,
    gpu_id: str,
    save_prob_flag: str,
    test_input_dir: Path,
    configuration: str,
    fold: str,
    npp: int,
    nps: int,
    cwd: Path,
    stage: int,
    disk_reporter: Optional[Callable[[str], None]] = None,
):
    if disk_reporter is not None:
        disk_reporter(f"BEFORE Stage {stage} (single)")
    cmd = _build_predict_cmd(spec, test_input_dir, out_dir, configuration, fold, npp, nps, save_prob_flag)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    log(f"Stage {stage} | GPU{gpu_id}: {cmd}")
    p = _popen(cmd, cwd=cwd, env=env)
    t = _stream_process_output_async(p, spec.model_name)
    rc = p.wait()
    t.join(timeout=2)
    if rc != 0:
        if disk_reporter is not None:
            disk_reporter(f"AFTER Stage {stage} (single) - FAILED")
        raise RuntimeError(f"nnUNetv2_predict failed for {spec.model_name}: rc={rc}")
    if disk_reporter is not None:
        disk_reporter(f"AFTER Stage {stage} (single)")


def _run_two_models_parallel(
    spec1: InferModelSpec,
    out1: Path,
    gpu1: str,
    spec2: InferModelSpec,
    out2: Path,
    gpu2: str,
    save_prob_flag: str,
    test_input_dir: Path,
    configuration: str,
    fold: str,
    npp: int,
    nps: int,
    cwd: Path,
    stage: int,
    disk_reporter: Optional[Callable[[str], None]] = None,
):
    if disk_reporter is not None:
        disk_reporter(f"BEFORE Stage {stage} (parallel)")
    cmd1 = _build_predict_cmd(spec1, test_input_dir, out1, configuration, fold, npp, nps, save_prob_flag)
    cmd2 = _build_predict_cmd(spec2, test_input_dir, out2, configuration, fold, npp, nps, save_prob_flag)
    env1 = os.environ.copy()
    env2 = os.environ.copy()
    env1["CUDA_VISIBLE_DEVICES"] = gpu1
    env2["CUDA_VISIBLE_DEVICES"] = gpu2
    log(f"Stage {stage} parallel")
    log(f"GPU{gpu1}: {cmd1}")
    log(f"GPU{gpu2}: {cmd2}")
    p1 = _popen(cmd1, cwd=cwd, env=env1)
    p2 = _popen(cmd2, cwd=cwd, env=env2)
    t1 = _stream_process_output_async(p1, spec1.model_name)
    t2 = _stream_process_output_async(p2, spec2.model_name)
    rc1 = p1.wait()
    rc2 = p2.wait()
    t1.join(timeout=2)
    t2.join(timeout=2)
    if rc1 != 0 or rc2 != 0:
        if disk_reporter is not None:
            disk_reporter(f"AFTER Stage {stage} (parallel) - FAILED")
        raise RuntimeError(f"nnUNetv2_predict failed: rc1={rc1}, rc2={rc2}")
    if disk_reporter is not None:
        disk_reporter(f"AFTER Stage {stage} (parallel)")


def parse_gpu_ids(raw_gpu_ids: str) -> list[str]:
    raw_gpu_ids = raw_gpu_ids.strip()
    if raw_gpu_ids:
        return [x.strip() for x in raw_gpu_ids.split(",") if x.strip()]
    try:
        import torch

        n = int(torch.cuda.device_count())
    except Exception:
        n = 0
    if n >= 2:
        return ["0", "1"]
    if n == 1:
        return ["0"]
    return []


def run_active_models_in_stages(
    active_modes: list[int],
    specs: dict[int, InferModelSpec],
    out_dirs: dict[int, Path],
    gpu_ids: list[str],
    save_prob_flag: str,
    test_input_dir: Path,
    configuration: str,
    fold: str,
    npp: int,
    nps: int,
    cwd: Path,
    disk_reporter: Optional[Callable[[str], None]] = None,
):
    if not gpu_ids:
        raise RuntimeError("no CUDA GPU detected")
    stage = 1
    i = 0
    while i < len(active_modes):
        if len(gpu_ids) >= 2 and i + 1 < len(active_modes):
            m1, m2 = active_modes[i], active_modes[i + 1]
            _run_two_models_parallel(
                spec1=specs[m1],
                out1=out_dirs[m1],
                gpu1=gpu_ids[0],
                spec2=specs[m2],
                out2=out_dirs[m2],
                gpu2=gpu_ids[1],
                save_prob_flag=save_prob_flag,
                test_input_dir=test_input_dir,
                configuration=configuration,
                fold=fold,
                npp=npp,
                nps=nps,
                cwd=cwd,
                stage=stage,
                disk_reporter=disk_reporter,
            )
            i += 2
        else:
            m1 = active_modes[i]
            _run_one_model(
                spec=specs[m1],
                out_dir=out_dirs[m1],
                gpu_id=gpu_ids[0],
                save_prob_flag=save_prob_flag,
                test_input_dir=test_input_dir,
                configuration=configuration,
                fold=fold,
                npp=npp,
                nps=nps,
                cwd=cwd,
                stage=stage,
                disk_reporter=disk_reporter,
            )
            i += 1
        stage += 1


def find_case_npz(pred_dir: Path, case_id: str) -> Path:
    candidates = [
        pred_dir / f"{case_id}.npz",
        pred_dir / "probabilities" / f"{case_id}.npz",
        pred_dir / f"{case_id}_0000.npz",
        pred_dir / "probabilities" / f"{case_id}_0000.npz",
    ]
    for c in candidates:
        if c.exists():
            return c
    hits = sorted(pred_dir.rglob(f"{case_id}*.npz"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"No NPZ found for {case_id} under {pred_dir}")


def load_surface_prob_from_npz(npz_path: Path, emulate_float16_maps: bool) -> np.ndarray:
    with np.load(npz_path) as data:
        keys = list(data.keys())
        arr = None
        for k in ("probabilities", "softmax", "predicted_probabilities", "probs"):
            if k in data:
                arr = data[k]
                break
        if arr is None:
            arr = data[keys[0]]
    arr = np.asarray(arr)
    if arr.ndim != 4:
        raise ValueError(f"Unexpected prob tensor shape {arr.shape} in {npz_path}")
    if arr.shape[0] <= 10 and arr.shape[1] > 10:
        p = arr[1].astype(np.float32)
    elif arr.shape[-1] <= 10 and arr.shape[0] > 10:
        p = arr[..., 1].astype(np.float32)
    else:
        p = arr[1].astype(np.float32)
    if emulate_float16_maps:
        p = p.astype(np.float16).astype(np.float32)
    return p


def normalize_weights(active_modes: list[int], weights: dict[int, float]) -> dict[int, float]:
    weighted_modes: list[tuple[int, float]] = []
    total = 0.0
    for m in active_modes:
        v = float(weights.get(m, 0.0))
        if v > 0:
            weighted_modes.append((m, v))
            total += v
    if total <= 0:
        raise ValueError(f"All weights are zero/non-positive for ACTIVE_MODES={active_modes}.")
    return {m: (v / total) for m, v in weighted_modes}


def fuse_probability_from_nnunet(
    case_id: str,
    active_modes: list[int],
    weights: dict[int, float],
    fusion_scheme: str,
    pred_dirs: dict[int, Path],
    emulate_float16_maps: bool,
    pair54_w5: float,
    pair54_w4: float,
    pair21_w2: float,
    pair21_w1: float,
    final_w54: float,
    final_w21: float,
) -> np.ndarray:
    if fusion_scheme == "DIRECT_WEIGHTED":
        normalized_weights = normalize_weights(active_modes, weights)
        p = None
        for m, w in normalized_weights.items():
            npz = find_case_npz(pred_dirs[m], case_id)
            pm = load_surface_prob_from_npz(npz, emulate_float16_maps=emulate_float16_maps)
            if p is None:
                p = w * pm
            else:
                p += w * pm
        assert p is not None
        return p

    if fusion_scheme == "PAIR_ENSEMBLE":
        required = {1, 2, 4, 5}
        if not required.issubset(set(active_modes)):
            raise ValueError(
                f"PAIR_ENSEMBLE requires ACTIVE_MODES to include {sorted(required)}; got {active_modes}"
            )

        p5 = load_surface_prob_from_npz(
            find_case_npz(pred_dirs[5], case_id),
            emulate_float16_maps=emulate_float16_maps,
        )
        p4 = load_surface_prob_from_npz(
            find_case_npz(pred_dirs[4], case_id),
            emulate_float16_maps=emulate_float16_maps,
        )
        p54 = (pair54_w5 * p5) + (pair54_w4 * p4)

        p2 = load_surface_prob_from_npz(
            find_case_npz(pred_dirs[2], case_id),
            emulate_float16_maps=emulate_float16_maps,
        )
        p1 = load_surface_prob_from_npz(
            find_case_npz(pred_dirs[1], case_id),
            emulate_float16_maps=emulate_float16_maps,
        )
        p21 = (pair21_w2 * p2) + (pair21_w1 * p1)
        return (final_w54 * p54) + (final_w21 * p21)

    raise ValueError(f"Unknown fusion_scheme={fusion_scheme}")


EULER_COEFS3D_26_NP = np.array(
    [
        0,
        1,
        1,
        0,
        1,
        0,
        -2,
        -1,
        1,
        -2,
        0,
        -1,
        0,
        -1,
        -1,
        0,
        1,
        0,
        -2,
        -1,
        -2,
        -1,
        -1,
        -2,
        -6,
        -3,
        -3,
        -2,
        -3,
        -2,
        0,
        -1,
        1,
        -2,
        0,
        -1,
        -6,
        -3,
        -3,
        -2,
        -2,
        -1,
        -1,
        -2,
        -3,
        0,
        -2,
        -1,
        0,
        -1,
        -1,
        0,
        -3,
        -2,
        0,
        -1,
        -3,
        0,
        -2,
        -1,
        0,
        1,
        1,
        0,
        1,
        -2,
        -6,
        -3,
        0,
        -1,
        -3,
        -2,
        -2,
        -1,
        -3,
        0,
        -1,
        -2,
        -2,
        -1,
        0,
        -1,
        -3,
        -2,
        -1,
        0,
        0,
        -1,
        -3,
        0,
        0,
        1,
        -2,
        -1,
        1,
        0,
        -2,
        -1,
        -3,
        0,
        -3,
        0,
        0,
        1,
        -1,
        4,
        0,
        3,
        0,
        3,
        1,
        2,
        -1,
        -2,
        -2,
        -1,
        -2,
        -1,
        1,
        0,
        0,
        3,
        1,
        2,
        1,
        2,
        2,
        1,
        1,
        -6,
        -2,
        -3,
        -2,
        -3,
        -1,
        0,
        0,
        -3,
        -1,
        -2,
        -1,
        -2,
        -2,
        -1,
        -2,
        -3,
        -1,
        0,
        -1,
        0,
        4,
        3,
        -3,
        0,
        0,
        1,
        0,
        1,
        3,
        2,
        0,
        -3,
        -1,
        -2,
        -3,
        0,
        0,
        1,
        -1,
        0,
        0,
        -1,
        -2,
        1,
        -1,
        0,
        -1,
        -2,
        -2,
        -1,
        0,
        1,
        3,
        2,
        -2,
        1,
        -1,
        0,
        1,
        2,
        2,
        1,
        0,
        -3,
        -3,
        0,
        -1,
        -2,
        0,
        1,
        -1,
        0,
        -2,
        1,
        0,
        -1,
        -1,
        0,
        -1,
        -2,
        0,
        1,
        -2,
        -1,
        3,
        2,
        -2,
        1,
        1,
        2,
        -1,
        0,
        2,
        1,
        -1,
        0,
        -2,
        1,
        -2,
        1,
        1,
        2,
        -2,
        3,
        -1,
        2,
        -1,
        2,
        0,
        1,
        0,
        -1,
        -1,
        0,
        -1,
        0,
        2,
        1,
        -1,
        2,
        0,
        1,
        0,
        1,
        1,
        0,
    ],
    dtype=np.int32,
)


def cupy_euler_number(image, connectivity: int):
    if cupy is None or cupyx is None:
        raise RuntimeError("cupy is not available")
    image_cp = cupy.asarray(image)
    image_cp = (image_cp > 0).astype(int)
    image_cp = cupy.pad(image_cp, pad_width=1, mode="constant")
    if image_cp.ndim != 3:
        raise ValueError("Input image is not 3D.")
    if connectivity == 2:
        raise NotImplementedError(
            "For 3D images, Euler number is implemented for connectivities 1 and 3 only"
        )
    config = cupy.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 4], [0, 2, 8]],
            [[0, 0, 0], [0, 16, 64], [0, 32, 128]],
        ]
    )
    coefs = cupy.asarray(EULER_COEFS3D_26_NP[::-1] if connectivity == 1 else EULER_COEFS3D_26_NP)
    xf = cupyx.scipy.ndimage.convolve(image_cp, config, mode="constant", cval=0)
    h = cupy.bincount(xf.ravel(), minlength=256)
    return int((0.125 * coefs @ h).item())


def get_best_projection_cupy(sheet_vol):
    if cupy is None:
        raise RuntimeError("cupy is not available")
    best_axis, best_area, best_mask = 0, 0, None
    for axis in range(3):
        proj_mask = cupy.max(sheet_vol, axis)
        proj_area = int(cupy.sum(proj_mask).item())
        if proj_area > best_area:
            best_axis = axis
            best_area = proj_area
            best_mask = proj_mask
    return best_axis, best_mask


def get_best_projection(sheet_vol: np.ndarray):
    best_axis, best_area, best_mask = 0, -1, None
    for axis in range(3):
        proj_mask = np.max(sheet_vol, axis=axis)
        proj_area = int(np.sum(proj_mask))
        if proj_area > best_area:
            best_axis, best_area, best_mask = axis, proj_area, proj_mask
    assert best_mask is not None
    return best_axis, best_mask


def get_hole_patches(sheet_vol: np.ndarray, proj_axis: int, proj_mask: np.ndarray, params: dict):
    proj_mask = proj_mask.astype(np.uint8)
    heightmap = np.argmax(sheet_vol, axis=proj_axis).astype(np.int32)
    bdr = int(params["border_width"])
    if bdr > 0:
        heightmap = heightmap[bdr:-bdr, bdr:-bdr]
        proj_mask = proj_mask[bdr:-bdr, bdr:-bdr]

    num_holes, hole_labels_img, hole_stats, _ = cv2.connectedComponentsWithStats(
        1 - proj_mask, connectivity=8, ltype=cv2.CV_16U
    )
    patches = []
    for i in range(1, num_holes):
        x0, y0, hole_w, hole_h, _ = hole_stats[i]
        if (
            x0 == 0
            or y0 == 0
            or (y0 + hole_h >= heightmap.shape[0])
            or (x0 + hole_w >= heightmap.shape[1])
        ):
            continue

        ph, pw = hole_h + 2, hole_w + 2
        interp_hm_h = np.full((ph, pw), -1, dtype=np.float32)
        interp_hm_v = np.full((ph, pw), -1, dtype=np.float32)
        wts_h = np.full((ph, pw), 0, dtype=np.float32)
        wts_v = np.full((ph, pw), 0, dtype=np.float32)

        hole_mask = hole_labels_img[y0 - 1 : y0 + ph - 1, x0 - 1 : x0 + pw - 1] == i
        not_hole_mask = ~hole_mask

        for y in range(ph):
            hm_row = heightmap[y + y0 - 1, x0 - 1 :]
            mask_row = hole_mask[y]
            wts_h_row = wts_h[y]
            interp_hm_h_row = interp_hm_h[y]
            ramping = False
            x = 0
            while x < pw:
                if not ramping:
                    if (not mask_row[x]) and (x < pw - 1) and mask_row[x + 1]:
                        ramping = True
                        ramp_start = (x, hm_row[x])
                else:
                    if not mask_row[x]:
                        ramping = False
                        ramp_end = (x, hm_row[x])
                        slope = (ramp_end[1] - ramp_start[1]) / (ramp_end[0] - ramp_start[0])
                        ht = ramp_start[1] + slope
                        for xx in range(ramp_start[0] + 1, ramp_end[0]):
                            interp_hm_h_row[xx] = ht
                            ht += slope
                            wts_h_row[xx] = 1 / min(xx - ramp_start[0], ramp_end[0] - xx)
                        x -= 1
                x += 1

        for x in range(pw):
            hm_col = heightmap[y0 - 1 :, x + x0 - 1]
            mask_col = hole_mask[:, x]
            wts_v_col = wts_v[:, x]
            interp_hm_v_col = interp_hm_v[:, x]
            ramping = False
            y = 0
            while y < ph:
                if not ramping:
                    if (not mask_col[y]) and (y < ph - 1) and mask_col[y + 1]:
                        ramping = True
                        ramp_start = (y, hm_col[y])
                else:
                    if not mask_col[y]:
                        ramping = False
                        ramp_end = (y, hm_col[y])
                        slope = (ramp_end[1] - ramp_start[1]) / (ramp_end[0] - ramp_start[0])
                        ht = ramp_start[1] + slope
                        for yy in range(ramp_start[0] + 1, ramp_end[0]):
                            interp_hm_v_col[yy] = ht
                            ht += slope
                            wts_v_col[yy] = 1 / min(yy - ramp_start[0], ramp_end[0] - yy)
                        y -= 1
                y += 1

        interp_hts = np.round(
            (wts_h * interp_hm_h + wts_v * interp_hm_v) / (wts_h + wts_v + 1e-8)
        ).astype(np.uint16)
        interp_hts[not_hole_mask] = heightmap[y0 - 1 : y0 - 1 + ph, x0 - 1 : x0 - 1 + pw][not_hole_mask]

        patch = np.zeros((ph, pw, 2), dtype=np.int32)
        for yy in range(1, hole_h + 1):
            for xx in range(1, hole_w + 1):
                nbr_hts = interp_hts[yy - 1 : yy + 2, xx - 1 : xx + 2]
                patch[yy, xx, 0] = np.min(nbr_hts)
                patch[yy, xx, 1] = np.max(nbr_hts)

        patch[:, :, 0][not_hole_mask] = -1
        patch[:, :, 1][not_hole_mask] = -1
        patch = np.ascontiguousarray(patch[1:-1, 1:-1, :])
        patches.append((patch, x0 + bdr, y0 + bdr, hole_w, hole_h))
    return patches, proj_axis


@jit(nopython=True)
def insert_patches_in_volume(patches, proj_axis, vol):
    pz_max = vol.shape[proj_axis]
    axes = (0, 1, 2) if proj_axis == 0 else (1, 0, 2) if proj_axis == 1 else (2, 0, 1)
    vol_t = np.transpose(vol, axes)
    for patch_info in patches:
        patch, px0, py0, pw, ph = patch_info
        for py in range(py0, py0 + ph):
            for px in range(px0, px0 + pw):
                for pz in range(patch[py - py0, px - px0, 0], patch[py - py0, px - px0, 1] + 1):
                    if (pz > 0) and (pz < pz_max):
                        vol_t[pz][py][px] = 1
    return vol


@jit(nopython=True)
def plug_small_holes_numba(in_vol, z_locs, y_locs, x_locs, lut):
    count = 0
    z_max = in_vol.shape[0] - 1
    y_max = in_vol.shape[1] - 1
    x_max = in_vol.shape[2] - 1
    for idx in range(len(z_locs)):
        z = z_locs[idx]
        y = y_locs[idx]
        x = x_locs[idx]
        if z >= z_max or y >= y_max or x >= x_max:
            continue
        lut_code = (
            in_vol[z, y, x]
            + 2 * in_vol[z, y, x + 1]
            + 4 * in_vol[z, y + 1, x]
            + 8 * in_vol[z, y + 1, x + 1]
            + 16 * in_vol[z + 1, y, x]
            + 32 * in_vol[z + 1, y, x + 1]
            + 64 * in_vol[z + 1, y + 1, x]
            + 128 * in_vol[z + 1, y + 1, x + 1]
        )
        lut_val = lut[lut_code]
        if lut_val[0, 0, 0] < 256:
            in_vol[z : z + 2, y : y + 2, x : x + 2] = lut_val
            count += 1
    return count


def plug_small_holes(in_vol: np.ndarray, lut):
    dilated_vol = ndi.binary_dilation(in_vol, structure=STRUCT_3X3X3)
    z, y, x = np.nonzero(dilated_vol)
    out_vol = np.asarray(in_vol, dtype=np.uint32).copy()
    count = plug_small_holes_numba(out_vol, z.astype(np.int64), y.astype(np.int64), x.astype(np.int64), lut)
    count += plug_small_holes_numba(out_vol, z.astype(np.int64), y.astype(np.int64), x.astype(np.int64), lut)
    return out_vol.astype(np.uint8), count


def plug_small_holes_cupy(in_vol, lut):
    if cupy is None or cupyx is None:
        raise RuntimeError("cupy is not available")
    dilation_struct = cupy.asarray(sk_morphology.footprint_rectangle((3, 3, 3)))
    dilated_vol = cupyx.scipy.ndimage.binary_dilation(in_vol, dilation_struct)

    z_locs, y_locs, x_locs = [cupy.asnumpy(a).astype(np.int64) for a in cupy.nonzero(dilated_vol)]
    out_vol = cupy.asnumpy(in_vol).astype(np.uint32, copy=True)
    count = plug_small_holes_numba(out_vol, z_locs, y_locs, x_locs, lut)
    count += plug_small_holes_numba(out_vol, z_locs, y_locs, x_locs, lut)
    return out_vol.astype(np.uint8), count


def code_to_array(code: int):
    result = np.zeros((8,), dtype=np.uint32)
    b = f"{code:08b}"
    for i in range(8):
        result[7 - i] = int(b[i])
    return result.reshape((2, 2, 2))


def create_plug_lut():
    lut = [None] * 256
    for i in range(256):
        lut[i] = np.full((8,), 512, dtype=np.uint32).reshape((2, 2, 2))

    indices = {
        6: 7, 9: 11, 18: 19, 20: 21, 22: 23, 24: 27, 25: 27, 26: 27, 28: 29, 30: 31,
        33: 35, 36: 39, 37: 39, 38: 39, 40: 42, 41: 43, 44: 46, 45: 47,
        52: 53, 54: 55, 56: 58, 57: 59, 60: 63, 61: 63, 62: 63,
        65: 69, 66: 71, 67: 71, 70: 71, 72: 76, 73: 77, 74: 78, 75: 79,
        82: 83, 86: 87, 88: 92, 89: 93, 90: 95, 91: 95, 94: 95,
        96: 112, 97: 113, 98: 114, 99: 115, 100: 116, 101: 117, 102: 119, 103: 119,
        104: 232, 105: 251, 106: 234, 108: 236, 107: 251, 109: 253, 110: 238, 111: 239,
        118: 119, 120: 248, 121: 127, 122: 127, 123: 127, 124: 127, 125: 127, 126: 127,
        129: 139, 130: 138, 131: 139, 132: 140, 133: 141, 134: 142, 135: 143, 137: 139,
        144: 176, 145: 177, 146: 178, 147: 179, 148: 212, 149: 213, 150: 223, 151: 223,
        152: 184, 153: 187, 154: 186, 155: 187, 156: 220, 157: 221, 158: 191, 159: 191,
        161: 163, 164: 172, 165: 175, 166: 174, 167: 175, 169: 171, 173: 175,
        180: 244, 181: 245, 182: 247, 183: 247, 185: 187, 188: 252, 189: 253, 190: 254,
        193: 197, 194: 202, 195: 207, 198: 206, 199: 207, 201: 205, 203: 207,
        210: 242, 211: 243, 214: 223, 215: 223, 217: 221, 218: 223, 219: 223, 222: 223,
        225: 241, 227: 243, 229: 245, 230: 238, 231: 247, 233: 239, 235: 251, 237: 253,
        246: 247, 249: 253,
    }
    for k, v in indices.items():
        lut[k] = code_to_array(v)
    return np.array(lut, dtype=np.uint32)


@lru_cache(maxsize=1)
def _cached_plug_lut() -> np.ndarray:
    return create_plug_lut()


@lru_cache(maxsize=16)
def _cached_closing_elements(close_par: int, close_perp: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sz = int(close_par)
    ctr = (sz - 1) // 2
    r_par = (int(close_par) - 1) // 2
    r_perp = (int(close_perp) - 1) // 2

    closing_element = np.zeros((sz, sz, sz), dtype=np.uint8)
    for z in range(sz):
        for y in range(sz):
            for x in range(sz):
                v = (
                    ((z - ctr) / (r_perp + 0.25)) ** 2
                    + ((y - ctr) / (r_par + 0.25)) ** 2
                    + ((x - ctr) / (r_par + 0.25)) ** 2
                )
                if v <= 1:
                    closing_element[z, y, x] = 1

    return (
        np.copy(np.transpose(closing_element, (0, 1, 2))),
        np.copy(np.transpose(closing_element, (1, 0, 2))),
        np.copy(np.transpose(closing_element, (1, 2, 0))),
    )


def _postprocess_cpu(in_vol: np.ndarray, params: dict) -> np.ndarray:
    in_vol = in_vol.astype(np.uint8).copy()
    bw = int(params["border_width"])
    if bw > 0:
        in_vol[0:bw, :, :] = in_vol[:, 0:bw, :] = in_vol[:, :, 0:bw] = 0
        in_vol[-bw:, :, :] = in_vol[:, -bw:, :] = in_vol[:, :, -bw:] = 0

    pruned_vol = sk_morphology.remove_small_objects(in_vol.astype(bool), min_size=int(params["small_obj_thresh"]))
    sheet_labels, num_sheets = ndi.label(pruned_vol.astype(np.uint8), structure=STRUCT_3X3X3)

    out_vol = np.zeros(in_vol.shape, dtype=np.uint8)
    lut = _cached_plug_lut()

    close_par = int(params.get("close_par", 7))
    close_perp = int(params.get("close_perp", 7))
    closing_elements = _cached_closing_elements(close_par=close_par, close_perp=close_perp)

    max_patch_size = int(params.get("max_patch_size", 64))
    max_sheets = int(params["max_sheets"])

    for sheet_idx in range(1, min(int(num_sheets), max_sheets) + 1):
        sheet_vol = (sheet_labels == sheet_idx).astype(np.uint8)
        proj_axis, proj_mask = get_best_projection(sheet_vol)
        close_elem = closing_elements[proj_axis]

        sheet_vol_closed = np.maximum(
            sheet_vol,
            ndi.binary_closing(sheet_vol, structure=close_elem).astype(np.uint8),
        )
        patches, _ = get_hole_patches(sheet_vol_closed, proj_axis, proj_mask, params)
        small_patches = [p for p in patches if p[3] * p[4] < max_patch_size**2]

        sheet_vol_patched = sheet_vol_closed
        if len(small_patches) > 0:
            sheet_vol_patched = insert_patches_in_volume(tuple(small_patches), proj_axis, sheet_vol_closed.copy())
            sheet_vol_patched = np.maximum(
                sheet_vol_patched,
                ndi.binary_closing(sheet_vol_patched, structure=close_elem).astype(np.uint8),
            )

        sheet_vol_fixed, _ = plug_small_holes(sheet_vol_patched, lut)
        num_holes_after = 1 - euler_number(sheet_vol_fixed.astype(bool), connectivity=1)
        if num_holes_after > 0:
            sheet_vol_nopatch, _ = plug_small_holes(sheet_vol_closed, lut)
            num_holes_before = 1 - euler_number(sheet_vol_nopatch.astype(bool), connectivity=1)
            if num_holes_after > num_holes_before:
                sheet_vol_fixed = sheet_vol_nopatch

        occupied = ndi.binary_dilation(out_vol != 0, structure=STRUCT_3X3X3)
        out_vol[(sheet_vol_fixed != 0) & (~occupied)] = sheet_idx

    out_vol = ndi.binary_fill_holes(out_vol).astype(np.uint8)
    if bw > 0:
        out_vol[0:bw, :, :] = out_vol[:, 0:bw, :] = out_vol[:, :, 0:bw] = 0
        out_vol[-bw:, :, :] = out_vol[:, -bw:, :] = out_vol[:, :, -bw:] = 0
    return (out_vol != 0).astype(np.uint8)


def _postprocess_cupy(in_vol: np.ndarray, params: dict) -> np.ndarray:
    if cupy is None or cupyx is None:
        raise RuntimeError("cupy is not available")

    in_vol = in_vol.astype(np.uint8).copy()
    bw = int(params["border_width"])
    if bw > 0:
        in_vol[0:bw, :, :] = in_vol[:, 0:bw, :] = in_vol[:, :, 0:bw] = 0
        in_vol[-bw:, :, :] = in_vol[:, -bw:, :] = in_vol[:, :, -bw:] = 0

    pruned_vol = sk_morphology.remove_small_objects(
        in_vol.astype(bool),
        min_size=int(params["small_obj_thresh"]),
    )

    struct_element333 = cupy.asarray(sk_morphology.footprint_rectangle((3, 3, 3)))
    sheet_labels_vol, num_sheets = cupyx.scipy.ndimage.label(cupy.asarray(pruned_vol), struct_element333)

    out_vol = cupy.zeros(in_vol.shape, dtype=np.uint8)
    sheet_vol = cupy.empty(sheet_labels_vol.shape, dtype=np.uint8)
    lut = _cached_plug_lut()

    close_par = int(params.get("close_par", 7))
    close_perp = int(params.get("close_perp", 7))
    sz = close_par
    ctr = (sz - 1) // 2
    r_par = (close_par - 1) // 2
    r_perp = (close_perp - 1) // 2

    closing_element = np.zeros((sz, sz, sz), dtype=np.uint8)
    for z in range(sz):
        for y in range(sz):
            for x in range(sz):
                v = (
                    ((z - ctr) / (r_perp + 0.25)) ** 2
                    + ((y - ctr) / (r_par + 0.25)) ** 2
                    + ((x - ctr) / (r_par + 0.25)) ** 2
                )
                if v <= 1:
                    closing_element[z, y, x] = 1

    closing_elements = [
        cupy.asarray(np.copy(np.transpose(closing_element, (0, 1, 2)))),
        cupy.asarray(np.copy(np.transpose(closing_element, (1, 0, 2)))),
        cupy.asarray(np.copy(np.transpose(closing_element, (1, 2, 0)))),
    ]

    max_patch_size = int(params.get("max_patch_size", 64))
    max_sheets = int(params["max_sheets"])

    for sheet_idx in range(1, min(int(num_sheets), max_sheets) + 1):
        sheet_vol.fill(0)
        sheet_vol[cupy.where(sheet_labels_vol == sheet_idx)] = 1

        proj_axis, proj_mask = get_best_projection_cupy(sheet_vol)
        close_elem = closing_elements[proj_axis]

        sheet_vol_closed = cupy.maximum(
            sheet_vol,
            cupyx.scipy.ndimage.binary_closing(sheet_vol, close_elem).astype(np.uint8),
        )

        patches, _ = get_hole_patches(cupy.asnumpy(sheet_vol_closed), proj_axis, cupy.asnumpy(proj_mask), params)
        small_patches = [p for p in patches if p[3] * p[4] < max_patch_size**2]
        sheet_vol_patched = sheet_vol_closed
        if len(small_patches) > 0:
            sheet_vol_patched = cupy.asarray(
                insert_patches_in_volume(tuple(small_patches), proj_axis, cupy.asnumpy(sheet_vol_closed))
            )
            sheet_vol_patched = cupy.maximum(
                sheet_vol_patched,
                cupyx.scipy.ndimage.binary_closing(sheet_vol_patched, close_elem).astype(np.uint8),
            )

        sheet_vol_fixed, _ = plug_small_holes_cupy(sheet_vol_patched, lut)
        sheet_vol_fixed = cupy.asarray(sheet_vol_fixed)

        num_holes_after = 1 - cupy_euler_number(sheet_vol_fixed, connectivity=1)
        if num_holes_after > 0:
            sheet_vol_nopatch, _ = plug_small_holes_cupy(sheet_vol_closed, lut)
            sheet_vol_nopatch = cupy.asarray(sheet_vol_nopatch)
            num_holes_before = 1 - cupy_euler_number(sheet_vol_nopatch, connectivity=1)
            if num_holes_after > num_holes_before:
                sheet_vol_fixed = sheet_vol_nopatch

        occupied = cupyx.scipy.ndimage.binary_dilation((out_vol != 0), struct_element333)
        out_vol[cupy.where((sheet_vol_fixed != 0) & (occupied == 0))] = sheet_idx

    out_vol = cupyx.scipy.ndimage.binary_fill_holes(out_vol).astype(cupy.uint8)
    if bw > 0:
        out_vol[0:bw, :, :] = out_vol[:, 0:bw, :] = out_vol[:, :, 0:bw] = 0
        out_vol[-bw:, :, :] = out_vol[:, -bw:, :] = out_vol[:, :, -bw:] = 0
    return cupy.asnumpy((out_vol != 0)).astype(np.uint8)


def postprocess(in_vol: np.ndarray, params: dict, use_cupy_postprocess: bool) -> np.ndarray:
    if use_cupy_postprocess and cupy is not None and cupyx is not None:
        return _postprocess_cupy(in_vol, params)
    return _postprocess_cpu(in_vol, params)


def _chunked(seq: list[Path], k: int) -> list[list[Path]]:
    return [seq[i : i + k] for i in range(0, len(seq), k)]


def _safe_rmtree(path: Path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _safe_clean_dir(path: Path):
    if path.exists():
        _safe_rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _parse_int_csv(raw: str) -> list[int]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if not parts:
        return []
    return [int(x) for x in parts]


def run_inference_pipeline(args):
    input_dir = Path(args.input_dir).resolve()
    working_dir = Path(args.working_dir).resolve()
    output_dir = working_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_dir = Path(args.weight_dir).resolve() if str(args.weight_dir).strip() else (output_dir / "nnUNet_results")
    if str(args.models_results_dir).strip():
        models_results_dir = Path(args.models_results_dir).resolve()
    else:
        models_results_dir = weight_dir
    if (models_results_dir / "nnUNet_results").is_dir() and not any(models_results_dir.glob("Dataset*_VesuviusSurface_M*")):
        models_results_dir = models_results_dir / "nnUNet_results"
    if not models_results_dir.exists():
        raise FileNotFoundError(f"weight/models_results directory not found: {models_results_dir}")

    configuration = str(args.configuration)
    fold = str(args.fold)
    fusion_scheme = str(args.fusion_scheme)
    specs = build_default_infer_specs(models_results_root=models_results_dir, configuration=configuration)
    active_modes = _parse_int_csv(args.active_modes)
    if not active_modes:
        active_modes = [7, 5, 2, 1]
    for m in active_modes:
        if m not in specs:
            raise ValueError(f"mode {m} is not supported. available={sorted(specs.keys())}")

    process_in_chunks = bool(args.process_in_chunks)
    cases_per_chunk = int(args.cases_per_chunk)
    zip_tiff_immediately = bool(args.zip_tiff_immediately)
    emulate_float16_maps = bool(args.emulate_float16_maps)
    use_cupy_postprocess = bool(args.use_cupy_postprocess)
    npp = int(args.npp)
    nps = int(args.nps)
    prob_t = float(args.prob_threshold)
    pp_params = {
        "small_obj_thresh": int(args.small_obj_thresh),
        "border_width": int(args.border_width),
        "max_sheets": int(args.max_sheets),
        "max_patch_size": int(args.max_patch_size),
        "close_par": int(args.close_par),
        "close_perp": int(args.close_perp),
    }

    clean_run_caches = bool(args.clean_run_caches)
    clean_nnunet_preprocessed = bool(args.clean_nnunet_preprocessed)
    clean_nnunet_results = bool(args.clean_nnunet_results)
    debug_disk_report = bool(args.debug_disk_report)
    disk_report_include_working = bool(args.disk_report_include_working)
    zip_name = str(args.zip_name)

    pair54_w5 = float(args.pair54_w5)
    pair54_w4 = float(args.pair54_w4)
    pair21_w2 = float(args.pair21_w2)
    pair21_w1 = float(args.pair21_w1)
    final_w54 = float(args.final_w54)
    final_w21 = float(args.final_w21)

    cache_root, tmp_root = setup_cache_and_tmp_env(working_dir)

    banner("Vesuvius nnUNet Inference")
    log(f"input_dir={input_dir}")
    log(f"working_dir={working_dir}")
    log(f"output_dir={output_dir}")
    log(f"weight_dir={weight_dir}")
    log(f"models_results_dir={models_results_dir}")
    log(f"active_modes={active_modes}")
    log(f"fusion_scheme={fusion_scheme}")
    log(f"configuration={configuration}, fold={fold}, npp={npp}, nps={nps}")
    log(f"prob_threshold={prob_t}")
    log(f"use_cupy_postprocess={use_cupy_postprocess} (cupy_available={cupy is not None and cupyx is not None})")
    log(f"cache_root={cache_root}")
    log(f"tmp_root={tmp_root}")

    paths = setup_nnunet_environment(working_dir=working_dir, compile_flag="false")
    test_input_dir = output_dir / "tmp_test_input"
    final_tiff_dir = output_dir / "tmp_predictions_tiff"
    pred_dirs = {m: output_dir / f"tmp_nnunet_out_m{m}" for m in active_modes}
    sub_zip = output_dir / zip_name

    def _disk(tag: str):
        disk_report(
            tag=tag,
            enabled=debug_disk_report,
            include_working=disk_report_include_working,
            working_dir=working_dir,
            output_dir=output_dir,
            nnunet_preprocessed=paths.preprocessed,
            nnunet_results=paths.results,
            cache_root=cache_root,
        )

    _disk("RUN START (pre-clean)")
    if clean_run_caches:
        _safe_rmtree(cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)
    if clean_nnunet_preprocessed:
        _safe_rmtree(paths.preprocessed)
    if clean_nnunet_results:
        _safe_rmtree(paths.results)

    _safe_clean_dir(final_tiff_dir)
    _safe_clean_dir(test_input_dir)
    for p in pred_dirs.values():
        _safe_clean_dir(p)
    paths.preprocessed.mkdir(parents=True, exist_ok=True)
    paths.results.mkdir(parents=True, exist_ok=True)
    setup_nnunet_environment(working_dir=working_dir, compile_flag="false")
    _disk("AFTER CLEANUP")

    log("[1/5] register model checkpoints")
    for m in active_modes:
        register_model(spec=specs[m], nnunet_results=paths.results, configuration=configuration)

    log("[2/5] detect probability-saving flag")
    save_prob_flag = detect_save_prob_flag()

    log("[3/5] collect test cases")
    all_tifs = list_all_test_tifs(input_dir)
    chunks = [all_tifs] if not process_in_chunks else _chunked(all_tifs, cases_per_chunk)

    if sub_zip.exists():
        sub_zip.unlink()

    weights = {m: specs[m].weight for m in active_modes}
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    log(f"gpu_ids={gpu_ids}")
    if fusion_scheme == "DIRECT_WEIGHTED":
        normalized_weights = normalize_weights(active_modes, weights)
        log(f"normalized_weights={normalized_weights}")
    else:
        log(
            "pair_ensemble_weights="
            f"(p54_w5={pair54_w5}, p54_w4={pair54_w4}, p21_w2={pair21_w2}, "
            f"p21_w1={pair21_w1}, final_w54={final_w54}, final_w21={final_w21})"
        )

    log("[4/5] predict + fuse + postprocess + zip")
    with zipfile.ZipFile(sub_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for ci, chunk_tifs in enumerate(chunks, start=1):
            log(f"chunk {ci}/{len(chunks)} | cases={len(chunk_tifs)}")

            case_ids = prepare_test_data_subset(chunk_tifs, test_input_dir=test_input_dir)
            case_tif_map = {p.stem: p for p in chunk_tifs}
            for p in pred_dirs.values():
                _safe_clean_dir(p)

            _disk(f"CHUNK {ci}: BEFORE PREDICT")
            run_active_models_in_stages(
                active_modes=active_modes,
                specs=specs,
                out_dirs=pred_dirs,
                gpu_ids=gpu_ids,
                save_prob_flag=save_prob_flag,
                test_input_dir=test_input_dir,
                configuration=configuration,
                fold=fold,
                npp=npp,
                nps=nps,
                cwd=working_dir,
                disk_reporter=_disk,
            )
            _disk(f"CHUNK {ci}: AFTER PREDICT")

            for cid in tqdm(case_ids, desc=f"Chunk {ci}: Fuse+PP+Zip"):
                p = fuse_probability_from_nnunet(
                    case_id=cid,
                    active_modes=active_modes,
                    weights=weights,
                    fusion_scheme=fusion_scheme,
                    pred_dirs=pred_dirs,
                    emulate_float16_maps=emulate_float16_maps,
                    pair54_w5=pair54_w5,
                    pair54_w4=pair54_w4,
                    pair21_w2=pair21_w2,
                    pair21_w1=pair21_w1,
                    final_w54=final_w54,
                    final_w21=final_w21,
                )
                mask = (p >= prob_t).astype(np.uint8)
                pp_mask = postprocess(mask, pp_params, use_cupy_postprocess=use_cupy_postprocess)
                original_tif_path = case_tif_map.get(cid, input_dir / "test_images" / f"{cid}.tif")
                if original_tif_path.exists():
                    original_volume = tifffile.imread(original_tif_path)
                    grid_png_path = output_dir / f"{cid}_slices_grid.png"
                    show_random_slices_grid(
                        original_volume=original_volume,
                        mask_volume=pp_mask,
                        case_id=cid,
                        save_png_path=grid_png_path,
                    )
                else:
                    log(f"Skip visualization for {cid}: source image not found at {original_tif_path}")

                tif_path = final_tiff_dir / f"{cid}.tif"
                tifffile.imwrite(tif_path, pp_mask)
                if zip_tiff_immediately:
                    zf.write(tif_path, tif_path.name)
                    tif_path.unlink()

                for m in active_modes:
                    try:
                        npz = find_case_npz(pred_dirs[m], cid)
                        npz.unlink()
                    except Exception:
                        pass

            for p in pred_dirs.values():
                _safe_rmtree(p)
            _safe_rmtree(test_input_dir)
            _disk(f"CHUNK {ci}: AFTER CLEANUP")

        if not zip_tiff_immediately:
            for tp in tqdm(sorted(final_tiff_dir.glob("*.tif")), desc="Zipping (final pass)"):
                zf.write(tp, tp.name)
                tp.unlink()

    log("[5/5] sanity check + cleanup")
    with zipfile.ZipFile(sub_zip, "r") as zf:
        n_zip = sum(1 for n in zf.namelist() if n.endswith(".tif"))
    log(f"SANITY: test_cases={len(all_tifs)} | zip_tifs={n_zip}")
    _safe_rmtree(final_tiff_dir)
    _safe_rmtree(test_input_dir)
    _disk("RUN END")
    log(f"submission={sub_zip} ({sub_zip.stat().st_size / 1024 / 1024:.1f} MB)")


def main():
    args = parse_args()
    run_inference_pipeline(args)


if __name__ == "__main__":
    main()








