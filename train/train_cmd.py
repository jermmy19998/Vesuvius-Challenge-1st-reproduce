import os
import re
from pathlib import Path
from typing import Callable, Optional

from source.common import CommandProgress, resolve_command, run_command

_FFT_CONV_TORCH29_DEPRECATION_PREFIX = (
    "Using a non-tuple sequence for multidimensional indexing is deprecated"
)
_FFT_CONV_MODULE_REGEX = r"fft_conv_pytorch\.fft_conv"
_TRAINER_EPOCHS_RE = re.compile(r"nnUNetTrainer_(\d+)epochs$")


def _build_train_env(gpu_id: Optional[int]) -> dict[str, str]:
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    ignore_filter = (
        f"ignore:{_FFT_CONV_TORCH29_DEPRECATION_PREFIX}:UserWarning:{_FFT_CONV_MODULE_REGEX}"
    )
    ignore_filter_fallback = f"ignore:{_FFT_CONV_TORCH29_DEPRECATION_PREFIX}:UserWarning"
    existing = env.get("PYTHONWARNINGS", "").strip()
    if not existing:
        env["PYTHONWARNINGS"] = f"{ignore_filter},{ignore_filter_fallback}"
    else:
        existing_filters = [x.strip() for x in existing.split(",") if x.strip()]
        new_filters = [ignore_filter, ignore_filter_fallback]
        for f in existing_filters:
            if f not in new_filters:
                new_filters.append(f)
        env["PYTHONWARNINGS"] = ",".join(new_filters)
    return env


def _trainer_total_epochs(trainer: str) -> Optional[int]:
    if trainer == "nnUNetTrainer":
        return 1000
    if trainer == "nnUNetTrainer_1epoch":
        return 1
    m = _TRAINER_EPOCHS_RE.match(trainer)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def run_nnunet_train(
    dataset_id: int,
    configuration: str,
    fold: str,
    plans_name: str,
    trainer: str,
    pretrained_weights: Optional[Path],
    timeout_sec: Optional[int],
    logs_dir: Optional[Path],
    gpu_id: Optional[int] = None,
    on_output_line: Optional[Callable[[str], None]] = None,
) -> tuple[bool, str]:
    exe = resolve_command("nnUNetv2_train")
    cmd: list[str] = [
        exe,
        f"{dataset_id:03d}",
        configuration,
        fold,
        "-p",
        plans_name,
        "-tr",
        trainer,
    ]
    if pretrained_weights is not None:
        cmd.extend(["-pretrained_weights", str(pretrained_weights)])

    train_name = f"Training_{dataset_id:03d}"
    if gpu_id is not None:
        train_name = f"{train_name}_gpu{gpu_id}"

    total_epochs = _trainer_total_epochs(trainer)
    ok, merged = run_command(
        cmd=cmd,
        name=train_name,
        timeout_sec=timeout_sec,
        logs_dir=logs_dir,
        env=_build_train_env(gpu_id=gpu_id),
        progress=CommandProgress(
            label=f"nnUNet training {dataset_id:03d}",
            total=total_epochs,
            unit="epochs",
            parse_epoch_from_output=True,
            heartbeat_sec=0,
            emit_every=1,
        ),
        suppress_output_substrings=[
            "fft_conv_pytorch/fft_conv.py:139: UserWarning: Using a non-tuple sequence for multidimensional indexing is deprecated",
            "output = output[crop_slices].contiguous()",
        ],
        on_output_line=on_output_line,
        shell=False,
    )
    return ok, merged
