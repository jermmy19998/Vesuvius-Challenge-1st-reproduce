import argparse
import os
import sys
from pathlib import Path
from typing import Optional

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.common import banner, log, setup_nnunet_environment
from source.preprocess import ensure_preprocessed_ready
from source.train.checkpoint_utils import (
    build_epoch_checkpoint_archiver,
    model_output_dir,
    prepare_resume_checkpoint,
    resolve_pretrained_checkpoint,
)
from source.train.specs import TrainModelSpec, load_train_specs
from source.train.swanlab_utils import (
    build_swanlab_line_logger,
    extract_epoch,
    extract_metrics_from_line,
    finish_swanlab_run,
    start_swanlab_run,
    swanlab_log,
)
from source.train.train_cmd import run_nnunet_train


def _apply_thread_env_defaults():
    # Keep user-provided env unchanged; only fill defaults for stability.
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")


def parse_args():
    here = THIS_DIR
    parser = argparse.ArgumentParser("Vesuvius Surface Detection - Train Only")
    parser.add_argument("--working_dir", type=str, default="./work")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument(
        "--active_train_yaml",
        type=str,
        default=str(here / "configs" / "train" / "models_active_set2.yaml"),
        help="Active training models yaml (used).",
    )
    parser.add_argument(
        "--unused_train_yaml",
        type=str,
        default=str(here / "configs" / "train" / "models_unused.yaml"),
        help="Unused training models yaml (kept separate by request, not executed).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="1",
        help="Single mode to train, must exist in active_train_yaml.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="",
        help="Single GPU id. Empty means inherit CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--configuration", type=str, default="3d_fullres")
    parser.add_argument("--fold", type=str, default="all")
    parser.add_argument(
        "--n_proc_da",
        type=int,
        default=4,
        help="nnUNet data augmentation worker processes per GPU. Set <=0 to use nnUNet default.",
    )
    parser.add_argument(
        "--nnunet_compile",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Set nnUNet_compile. Use false to reduce compile overhead and memory pressure.",
    )
    parser.add_argument(
        "--resume_ckpt",
        action="store_true",
        help="Resume current mode training from latest checkpoint (nnUNetv2_train --c).",
    )
    parser.add_argument(
        "--pretrained_ckpt_policy",
        type=str,
        default="final",
        choices=["final", "best", "auto"],
        help="Checkpoint policy for pretrained_from_mode: final, best, or auto(final->best).",
    )
    parser.add_argument("--command_logs_dir", type=str, default="")
    parser.add_argument("--timeout_sec", type=int, default=0)
    parser.add_argument(
        "--save_every_epochs",
        type=int,
        default=50,
        help="Archive checkpoint_latest to checkpoint_<N>e.pth every N completed epochs. <=0 disables archive.",
    )
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--swanlab_project", type=str, default="vesuvius-nnunet-train")
    parser.add_argument("--swanlab_workspace", type=str, default="")
    parser.add_argument("--swanlab_run_prefix", type=str, default="")
    return parser.parse_args()


def _parse_modes(raw: str, available: list[int]) -> list[int]:
    if not raw.strip():
        return sorted(set(int(x) for x in available))
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_gpu_ids(raw: str) -> list[int]:
    value = raw.strip()
    if not value:
        return []
    gpu_ids: list[int] = []
    seen: set[int] = set()
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        gpu_id = int(token)
        if gpu_id < 0:
            raise ValueError(f"gpu id must be >= 0, got: {gpu_id}")
        if gpu_id in seen:
            raise ValueError(f"duplicate gpu id is not allowed: {gpu_id}")
        seen.add(gpu_id)
        gpu_ids.append(gpu_id)
    return gpu_ids


def _iter_metric_records(output_text: str):
    step: Optional[int] = None
    auto_step = 0
    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        epoch = extract_epoch(line)
        if epoch is not None and epoch > 0:
            step = epoch

        metrics = extract_metrics_from_line(line)
        if not metrics:
            continue

        if step is None:
            auto_step += 1
            current_step = auto_step
        else:
            current_step = step

        yield current_step, metrics


def _failure_detail(output_text: str) -> str:
    lines = [line.strip() for line in output_text.splitlines() if line.strip()]
    if not lines:
        return "unknown failure"

    priority_terms = (
        "terminated by signal",
        "timeout after",
        "failed to start",
        "exited with returncode",
        "traceback",
        "runtimeerror:",
        "error:",
    )
    for line in reversed(lines):
        lower = line.lower()
        if any(term in lower for term in priority_terms):
            return line
    return lines[-1]


def _train_one_mode(
    args,
    mode: int,
    mode_index: int,
    total_modes: int,
    spec: TrainModelSpec,
    specs: dict[int, TrainModelSpec],
    paths,
    timeout_sec: Optional[int],
    logs_dir: Path,
    gpu_id: Optional[int],
):
    log("-" * 60)
    gpu_desc = f"gpu={gpu_id}" if gpu_id is not None else "gpu=inherit_env"
    log(f"[{mode_index}/{total_modes}] mode={mode} | {spec.tag} | {gpu_desc}")
    log(
        f"dataset={spec.dataset_name} plans={spec.plans_name} trainer={spec.trainer} "
        f"patch={spec.patch_size} batch={spec.batch_size}"
    )

    ensure_preprocessed_ready(
        preprocessed_root=paths.preprocessed,
        dataset_name=spec.dataset_name,
        configuration=args.configuration,
    )
    fold_dir = model_output_dir(paths.results, spec, args.configuration, args.fold)

    pretrained_weights = None
    if spec.pretrained_from_mode is not None:
        if spec.pretrained_from_mode not in specs:
            raise ValueError(
                f"mode {mode} requires pretrained_from_mode={spec.pretrained_from_mode}, "
                "but it is not in active yaml."
            )
        pretrained_weights = resolve_pretrained_checkpoint(
            results_root=paths.results,
            source_spec=specs[spec.pretrained_from_mode],
            configuration=args.configuration,
            fold=args.fold,
            policy=str(args.pretrained_ckpt_policy),
        )
        log(f"pretrained_weights={pretrained_weights}")

    resume_enabled = bool(args.resume_ckpt)
    if resume_enabled:
        try:
            resume_path = prepare_resume_checkpoint(fold_dir, log_fn=log)
            log(f"resume checkpoint ready: {resume_path}")
            if pretrained_weights is not None:
                log(
                    "resume is enabled and checkpoint exists; "
                    "ignore pretrained_weights to satisfy nnUNet constraints"
                )
                pretrained_weights = None
        except RuntimeError as e:
            log(f"resume checkpoint unavailable, fallback to fresh training: {e}")
            resume_enabled = False

    swanlab_mod, swanlab_run = start_swanlab_run(args=args, mode=mode, spec=spec, log_fn=log)
    swanlab_line_logger = None
    swanlab_line_count = None
    if swanlab_mod is not None:
        swanlab_line_logger, swanlab_line_count = build_swanlab_line_logger(
            swanlab_mod=swanlab_mod,
            run=swanlab_run,
            log_fn=log,
        )

    archive_line_logger, archive_finalize, archive_count_get = build_epoch_checkpoint_archiver(
        fold_dir=fold_dir,
        save_every_epochs=int(args.save_every_epochs),
        extract_epoch_fn=extract_epoch,
        log_fn=log,
    )

    def _on_training_line(raw_line: str):
        if swanlab_line_logger is not None:
            swanlab_line_logger(raw_line)
        archive_line_logger(raw_line)

    try:
        ok, train_output = run_nnunet_train(
            dataset_id=spec.dataset_id,
            configuration=args.configuration,
            fold=args.fold,
            plans_name=spec.plans_name,
            trainer=spec.trainer,
            pretrained_weights=pretrained_weights,
            resume=resume_enabled,
            timeout_sec=timeout_sec,
            logs_dir=logs_dir,
            gpu_id=gpu_id,
            n_proc_da=args.n_proc_da,
            on_output_line=_on_training_line,
        )
        archive_finalize(train_output=train_output, train_ok=ok)
        log(f"checkpoint archived records for mode {mode}: {archive_count_get()}")

        if swanlab_mod is not None:
            log_count = swanlab_line_count() if swanlab_line_count is not None else 0
            # Fallback for environments where line callbacks are unavailable.
            if log_count == 0:
                for step, metrics in _iter_metric_records(train_output):
                    payload = dict(metrics)
                    payload["epoch"] = float(step)
                    swanlab_log(swanlab_mod, swanlab_run, payload, step=step, log_fn=log)
                    log_count += 1
            log(f"swanlab logged {log_count} records for mode {mode}")

        if not ok:
            raise RuntimeError(f"training failed for mode {mode}: {_failure_detail(train_output)}")
    finally:
        if swanlab_mod is not None:
            finish_swanlab_run(swanlab_mod, swanlab_run, log_fn=log)


def main():
    args = parse_args()
    _apply_thread_env_defaults()
    specs = load_train_specs(Path(args.active_train_yaml).resolve())
    modes = _parse_modes(args.modes, list(specs.keys()))
    gpu_ids = _parse_gpu_ids(args.gpu_ids)

    if len(modes) != 1:
        raise ValueError(
            f"this script now supports exactly one mode per run, got {modes}. "
            "Please run separate commands (for example: --modes 1)."
        )
    if len(gpu_ids) > 1:
        raise ValueError(
            f"this script now supports at most one gpu id per run, got {gpu_ids}. "
            "Please run separate commands (for example: --gpu_ids 0)."
        )
    for m in modes:
        if m not in specs:
            raise ValueError(f"mode {m} is not defined in {args.active_train_yaml}")

    working_dir = Path(args.working_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    logs_dir = Path(args.command_logs_dir).resolve() if args.command_logs_dir else (output_dir / "logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timeout_sec: Optional[int] = args.timeout_sec if args.timeout_sec > 0 else None

    banner("Vesuvius nnUNet Train Only")
    log(f"working_dir={working_dir}")
    log(f"output_dir={output_dir}")
    log(f"active_train_yaml={Path(args.active_train_yaml).resolve()}")
    log(f"modes={modes}")
    log(f"gpu_ids={gpu_ids if gpu_ids else 'inherit_env'}")
    log(f"configuration={args.configuration}, fold={args.fold}")
    log(f"n_proc_da={args.n_proc_da}")
    log(f"nnUNet_compile={args.nnunet_compile}")
    log(f"resume_ckpt={bool(args.resume_ckpt)}")
    log(f"save_every_epochs={int(args.save_every_epochs)}")
    log(f"pretrained_ckpt_policy={args.pretrained_ckpt_policy}")
    log(f"command_logs_dir={logs_dir}")
    log(f"swanlab={bool(args.swanlab)} project={args.swanlab_project}")
    log(
        "threads="
        f"OMP:{os.environ.get('OMP_NUM_THREADS', '')}, "
        f"MKL:{os.environ.get('MKL_NUM_THREADS', '')}, "
        f"OPENBLAS:{os.environ.get('OPENBLAS_NUM_THREADS', '')}, "
        f"NUMEXPR:{os.environ.get('NUMEXPR_NUM_THREADS', '')}"
    )

    unused_yaml = Path(args.unused_train_yaml).resolve()
    if unused_yaml.exists():
        log(f"unused_train_yaml exists (kept separate): {unused_yaml}")

    paths = setup_nnunet_environment(
        working_dir=working_dir,
        output_dir=output_dir,
        compile_flag=str(args.nnunet_compile),
    )
    mode = modes[0]
    gpu_id = gpu_ids[0] if gpu_ids else None
    log("training_schedule=single_mode_per_process")
    _train_one_mode(
        args=args,
        mode=mode,
        mode_index=1,
        total_modes=1,
        spec=specs[mode],
        specs=specs,
        paths=paths,
        timeout_sec=timeout_sec,
        logs_dir=logs_dir,
        gpu_id=gpu_id,
    )
    log("all requested training modes finished")


if __name__ == "__main__":
    main()
