import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .logging_utils import log


@dataclass
class NnunetPaths:
    base: Path
    raw: Path
    preprocessed: Path
    results: Path


def get_dataset_name(dataset_id: int, suffix: str = "VesuviusSurface", mode: int | None = None) -> str:
    name = f"Dataset{dataset_id:03d}_{suffix}"
    if mode is not None:
        name += f"_M{mode}"
    return name


def setup_nnunet_environment(
    working_dir: Path,
    output_dir: Path,
    compile_flag: str = "true",
) -> NnunetPaths:
    base = working_dir / "nnUNet_data"
    raw = base / "nnUNet_raw"
    preprocessed = base / "nnUNet_preprocessed"
    results = output_dir / "nnUNet_results"

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


def symlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)

