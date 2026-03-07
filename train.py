import argparse
import re
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Optional

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.common import banner, log, setup_nnunet_environment
from source.preprocess import ensure_preprocessed_ready
from source.train.specs import TrainModelSpec, load_train_specs
from source.train.train_cmd import run_nnunet_train


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
        "--modes",
        type=str,
        default="1,2,5,7",
        help="Comma-separated modes to train, must exist in active_train_yaml.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="",
        help=(
            "Comma-separated GPU ids. Runs up to len(gpu_ids) modes in parallel; "
            "remaining modes wait in queue."
        ),
    )
    parser.add_argument("--configuration", type=str, default="3d_fullres")
    parser.add_argument("--fold", type=str, default="all")
    parser.add_argument("--command_logs_dir", type=str, default="")
    parser.add_argument("--timeout_sec", type=int, default=0)
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--swanlab_project", type=str, default="vesuvius-nnunet-train")
    parser.add_argument("--swanlab_workspace", type=str, default="")
    parser.add_argument("--swanlab_run_prefix", type=str, default="train")
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


def _model_output_dir(results_root: Path, spec: TrainModelSpec, configuration: str, fold: str) -> Path:
    return results_root / spec.dataset_name / f"{spec.trainer}__{spec.plans_name}__{configuration}" / f"fold_{fold}"


def _resolve_pretrained_checkpoint(
    results_root: Path,
    source_spec: TrainModelSpec,
    configuration: str,
    fold: str,
) -> Path:
    fold_dir = _model_output_dir(results_root, source_spec, configuration, fold)
    c_final = fold_dir / "checkpoint_final.pth"
    c_best = fold_dir / "checkpoint_best.pth"
    if c_final.exists():
        return c_final
    if c_best.exists():
        return c_best
    raise FileNotFoundError(f"pretrained checkpoint not found: {fold_dir}")


def _extract_epoch(line: str) -> Optional[int]:
    tokens = _tokenize_metric_line(line)
    for idx, token in enumerate(tokens):
        if token != "epoch":
            continue
        for candidate in tokens[idx + 1 : idx + 4]:
            if "/" in candidate:
                candidate = candidate.split("/", 1)[0]
            if candidate.isdigit():
                value = int(candidate)
                if value > 0:
                    return value
    return None


def _to_float(raw: str) -> Optional[float]:
    normalized = raw.strip().rstrip(",;)]}")
    if not normalized:
        return None
    try:
        value = float(normalized)
    except Exception:
        return None
    if value != value:
        return None
    if value == float("inf") or value == float("-inf"):
        return None
    return value


def _normalize_metric_key(raw_key: str) -> Optional[str]:
    key = raw_key.lower().strip().replace("-", "_").replace(".", "_")
    key = key.replace("learningrate", "learning_rate")

    if key == "lr" or key.endswith("_lr") or "learning_rate" in key:
        return "train/lr"

    if "loss" in key:
        if key.startswith(("val", "valid", "validation")) or "_val" in key or "val_" in key:
            return "val/loss"
        return "train/loss"

    if "dice" in key or "metric" in key:
        if key.startswith(("val", "valid", "validation")) or "_val" in key or "val_" in key:
            return "val/metric"
        if key.startswith(("train", "tr")):
            return "train/metric"
        return "val/metric"

    return None


def _tokenize_metric_line(line: str) -> list[str]:
    normalized = line.lower()
    for sep in (":", "=", ",", "|", "(", ")", "[", "]"):
        normalized = normalized.replace(sep, " ")
    return [x for x in normalized.split() if x]


def _is_metric_key_candidate(token: str) -> bool:
    if not token or not token[0].isalpha():
        return False
    if len(token) > 48:
        return False
    for ch in token:
        if not (ch.isalnum() or ch in ("_", ".", "-", "/")):
            return False
    return True


def _extract_number_after_keywords(line: str, keywords: list[str], phrase_keywords: list[tuple[str, ...]]) -> Optional[float]:
    tokens = _tokenize_metric_line(line)
    for idx, token in enumerate(tokens):
        if token in keywords:
            for candidate in tokens[idx + 1 : idx + 4]:
                value = _to_float(candidate)
                if value is not None:
                    return value
    for phrase in phrase_keywords:
        width = len(phrase)
        for idx in range(0, len(tokens) - width + 1):
            if tuple(tokens[idx : idx + width]) == phrase:
                for candidate in tokens[idx + width : idx + width + 3]:
                    value = _to_float(candidate)
                    if value is not None:
                        return value
    return None


def _extract_metrics_from_line(line: str) -> dict[str, float]:
    metrics: dict[str, float] = {}

    tokens = _tokenize_metric_line(line)
    for idx in range(0, len(tokens) - 1):
        raw_key = tokens[idx]
        if not _is_metric_key_candidate(raw_key):
            continue
        raw_value = tokens[idx + 1]
        value = _to_float(raw_value)
        if value is None:
            continue
        normalized = _normalize_metric_key(raw_key)
        if normalized is None:
            continue
        metrics[normalized] = value

    lower = line.lower()

    if "train/loss" not in metrics and "val/loss" not in metrics and "loss" in lower:
        loss_value = _extract_number_after_keywords(
            line=line,
            keywords=["loss"],
            phrase_keywords=[],
        )
        if loss_value is not None:
            if "val" in lower or "valid" in lower or "validation" in lower:
                metrics["val/loss"] = loss_value
            else:
                metrics["train/loss"] = loss_value

    if "train/lr" not in metrics and ("learning rate" in lower or " learning_rate" in lower or " lr" in lower):
        lr_value = _extract_number_after_keywords(
            line=line,
            keywords=["lr", "learning_rate"],
            phrase_keywords=[("learning", "rate")],
        )
        if lr_value is not None:
            metrics["train/lr"] = lr_value

    if "train/metric" not in metrics and "val/metric" not in metrics and ("dice" in lower or "metric" in lower):
        metric_value = _extract_number_after_keywords(
            line=line,
            keywords=["dice", "metric"],
            phrase_keywords=[],
        )
        if metric_value is not None:
            if "val" in lower or "valid" in lower or "validation" in lower:
                metrics["val/metric"] = metric_value
            else:
                metrics["train/metric"] = metric_value

    return metrics


def _iter_metric_records(output_text: str):
    step: Optional[int] = None
    auto_step = 0
    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        epoch = _extract_epoch(line)
        if epoch is not None and epoch > 0:
            step = epoch

        metrics = _extract_metrics_from_line(line)
        if not metrics:
            continue

        if step is None:
            auto_step += 1
            current_step = auto_step
        else:
            current_step = step

        yield current_step, metrics


def _start_swanlab_run(args, mode: int, spec: TrainModelSpec):
    if not bool(args.swanlab):
        return None, None

    try:
        import swanlab  # type: ignore
    except Exception as e:
        log(f"swanlab unavailable, skip logging: {e!r}")
        return None, None

    run_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", f"{args.swanlab_run_prefix}_m{mode}_{spec.tag}")
    config = {
        "mode": int(mode),
        "dataset_id": int(spec.dataset_id),
        "dataset_name": str(spec.dataset_name),
        "trainer": str(spec.trainer),
        "plans_name": str(spec.plans_name),
        "patch_size": int(spec.patch_size),
        "batch_size": int(spec.batch_size),
    }

    kwargs: dict[str, Any] = {
        "project": str(args.swanlab_project),
        "experiment_name": run_name,
        "config": config,
    }
    workspace = str(args.swanlab_workspace).strip()
    if workspace:
        kwargs["workspace"] = workspace

    try:
        run = swanlab.init(**kwargs)
    except TypeError:
        fallback_kwargs: dict[str, Any] = {
            "project": str(args.swanlab_project),
            "name": run_name,
            "config": config,
        }
        if workspace:
            fallback_kwargs["workspace"] = workspace
        run = swanlab.init(**fallback_kwargs)

    log(f"swanlab run started: {run_name}")
    return swanlab, run


def _swanlab_log(swanlab_mod: Any, run: Any, payload: dict[str, float], step: int):
    if not payload:
        return

    if run is not None and hasattr(run, "log"):
        try:
            run.log(payload, step=step)
            return
        except TypeError:
            run.log(payload)
            return

    if hasattr(swanlab_mod, "log"):
        try:
            swanlab_mod.log(payload, step=step)
        except TypeError:
            swanlab_mod.log(payload)


def _finish_swanlab_run(swanlab_mod: Any, run: Any):
    if run is not None and hasattr(run, "finish"):
        run.finish()
        return
    if hasattr(swanlab_mod, "finish"):
        swanlab_mod.finish()


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

    pretrained_weights = None
    if spec.pretrained_from_mode is not None:
        if spec.pretrained_from_mode not in specs:
            raise ValueError(
                f"mode {mode} requires pretrained_from_mode={spec.pretrained_from_mode}, "
                "but it is not in active yaml."
            )
        pretrained_weights = _resolve_pretrained_checkpoint(
            results_root=paths.results,
            source_spec=specs[spec.pretrained_from_mode],
            configuration=args.configuration,
            fold=args.fold,
        )
        log(f"pretrained_weights={pretrained_weights}")

    swanlab_mod, swanlab_run = _start_swanlab_run(args=args, mode=mode, spec=spec)
    try:
        ok, train_output = run_nnunet_train(
            dataset_id=spec.dataset_id,
            configuration=args.configuration,
            fold=args.fold,
            plans_name=spec.plans_name,
            trainer=spec.trainer,
            pretrained_weights=pretrained_weights,
            timeout_sec=timeout_sec,
            logs_dir=logs_dir,
            gpu_id=gpu_id,
        )

        if swanlab_mod is not None:
            log_count = 0
            for step, metrics in _iter_metric_records(train_output):
                payload = dict(metrics)
                payload["epoch"] = float(step)
                _swanlab_log(swanlab_mod, swanlab_run, payload, step=step)
                log_count += 1
            log(f"swanlab logged {log_count} records for mode {mode}")

        if not ok:
            raise RuntimeError(f"training failed for mode {mode}")
    finally:
        if swanlab_mod is not None:
            _finish_swanlab_run(swanlab_mod, swanlab_run)


def _run_parallel_with_gpu_queue(
    args,
    modes: list[int],
    specs: dict[int, TrainModelSpec],
    paths,
    timeout_sec: Optional[int],
    logs_dir: Path,
    gpu_ids: list[int],
):
    pending: list[tuple[int, int]] = [(idx, mode) for idx, mode in enumerate(modes, start=1)]
    requested_modes = set(modes)
    completed_modes: set[int] = set()
    available_gpus = list(gpu_ids)
    running: dict[Any, tuple[int, int]] = {}
    errors: list[str] = []

    def _pick_ready_index() -> Optional[int]:
        for index, (_, mode) in enumerate(pending):
            dep_mode = specs[mode].pretrained_from_mode
            if dep_mode is None or dep_mode not in requested_modes or dep_mode in completed_modes:
                return index
        return None

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        while pending or running:
            launched = False
            while (not errors) and available_gpus:
                ready_index = _pick_ready_index()
                if ready_index is None:
                    break
                mode_index, mode = pending.pop(ready_index)
                gpu_id = available_gpus.pop(0)
                spec = specs[mode]
                future = pool.submit(
                    _train_one_mode,
                    args=args,
                    mode=mode,
                    mode_index=mode_index,
                    total_modes=len(modes),
                    spec=spec,
                    specs=specs,
                    paths=paths,
                    timeout_sec=timeout_sec,
                    logs_dir=logs_dir,
                    gpu_id=gpu_id,
                )
                running[future] = (mode, gpu_id)
                launched = True

            if not running:
                if errors:
                    break
                if not launched and pending:
                    blocked = ", ".join(
                        f"mode {mode} <- {specs[mode].pretrained_from_mode}"
                        for _, mode in pending
                    )
                    raise RuntimeError(f"no schedulable mode found, dependency blocked: {blocked}")
                continue

            done, _ = wait(set(running.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                mode, gpu_id = running.pop(future)
                available_gpus.append(gpu_id)
                try:
                    future.result()
                    completed_modes.add(mode)
                except Exception as e:
                    errors.append(f"mode {mode} on gpu {gpu_id}: {e}")

    if errors:
        raise RuntimeError("parallel training failed:\n" + "\n".join(errors))


def main():
    args = parse_args()
    specs = load_train_specs(Path(args.active_train_yaml).resolve())
    modes = _parse_modes(args.modes, list(specs.keys()))
    gpu_ids = _parse_gpu_ids(args.gpu_ids)

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
    log(f"command_logs_dir={logs_dir}")
    log(f"swanlab={bool(args.swanlab)} project={args.swanlab_project}")

    paths = setup_nnunet_environment(working_dir=working_dir, output_dir=output_dir, compile_flag="true")
    if gpu_ids:
        log(f"training_schedule=parallel_queue max_parallel={len(gpu_ids)}")
        _run_parallel_with_gpu_queue(
            args=args,
            modes=modes,
            specs=specs,
            paths=paths,
            timeout_sec=timeout_sec,
            logs_dir=logs_dir,
            gpu_ids=gpu_ids,
        )
    else:
        log("training_schedule=sequential_in_modes_order")
        for idx, mode in enumerate(modes, start=1):
            spec = specs[mode]
            _train_one_mode(
                args=args,
                mode=mode,
                mode_index=idx,
                total_modes=len(modes),
                spec=spec,
                specs=specs,
                paths=paths,
                timeout_sec=timeout_sec,
                logs_dir=logs_dir,
                gpu_id=None,
            )

    log("all requested training modes finished")


if __name__ == "__main__":
    main()

