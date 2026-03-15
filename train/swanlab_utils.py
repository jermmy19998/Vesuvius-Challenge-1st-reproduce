import inspect
import re
from typing import Any, Optional

_FLOAT_RE = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_EPOCH_RE = re.compile(r"(?i)\bepoch(?:\s+|:|=)\s*(\d+)\b")
_EPOCH_RATIO_RE = re.compile(r"(?i)\bepoch(?:\s+|:|=)\s*(\d+)\s*/\s*(\d+)\b")
_KEY_VALUE_RE = re.compile(rf"(?i)\b([a-z][a-z0-9_./-]{{1,48}})\b\s*(?::|=|\s)\s*({_FLOAT_RE})")


def _patch_pynvml_shutdown(log_fn) -> None:
    try:
        import pynvml  # type: ignore
    except Exception:
        return

    if getattr(pynvml, "_vesuvius_safe_shutdown_patch", False):
        return

    shutdown = getattr(pynvml, "nvmlShutdown", None)
    uninitialized_error = getattr(pynvml, "NVMLError_Uninitialized", None)
    if shutdown is None or uninitialized_error is None:
        return

    def _safe_nvml_shutdown(*args, **kwargs):
        try:
            return shutdown(*args, **kwargs)
        except uninitialized_error:
            # SwanLab may call nvmlShutdown more than once during interpreter teardown.
            return None

    try:
        pynvml.nvmlShutdown = _safe_nvml_shutdown
        pynvml._vesuvius_safe_shutdown_patch = True
    except Exception as e:
        log_fn(f"pynvml shutdown patch skipped: {e!r}")


def extract_epoch(line: str) -> Optional[int]:
    m_ratio = _EPOCH_RATIO_RE.search(line)
    if m_ratio:
        return int(m_ratio.group(1))
    m = _EPOCH_RE.search(line)
    if m:
        return int(m.group(1))
    return None


def _to_float(raw: str) -> Optional[float]:
    try:
        value = float(raw)
    except Exception:
        return None
    if value != value:
        return None
    if value in (float("inf"), float("-inf")):
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


def _extract_number_after_keyword(line: str, keyword_pattern: str) -> Optional[float]:
    m = re.search(rf"(?i)\b(?:{keyword_pattern})\b[^0-9+\-]*({_FLOAT_RE})", line)
    if not m:
        return None
    return _to_float(m.group(1))


def extract_metrics_from_line(line: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for m in _KEY_VALUE_RE.finditer(line):
        normalized = _normalize_metric_key(m.group(1))
        if normalized is None:
            continue
        value = _to_float(m.group(2))
        if value is None:
            continue
        metrics[normalized] = value

    lower = line.lower()
    if "train/loss" not in metrics and "val/loss" not in metrics and "loss" in lower:
        value = _extract_number_after_keyword(line, "loss")
        if value is not None:
            metrics["val/loss" if "val" in lower else "train/loss"] = value

    if "train/lr" not in metrics and ("learning rate" in lower or " learning_rate" in lower or " lr" in lower):
        value = _extract_number_after_keyword(line, "lr|learning\\s*rate|learning_rate")
        if value is not None:
            metrics["train/lr"] = value

    if "train/metric" not in metrics and "val/metric" not in metrics and ("dice" in lower or "metric" in lower):
        value = _extract_number_after_keyword(line, "dice|metric")
        if value is not None:
            metrics["val/metric" if "val" in lower else "train/metric"] = value
    return metrics


def start_swanlab_run(args, mode: int, spec, log_fn):
    if not bool(args.swanlab):
        return None, None
    try:
        import swanlab  # type: ignore
    except Exception as e:
        log_fn(f"swanlab unavailable, skip logging: {e!r}")
        return None, None

    _patch_pynvml_shutdown(log_fn)

    run_prefix = str(args.swanlab_run_prefix).strip()
    run_name_raw = f"m{mode}_{spec.tag}" if not run_prefix else f"{run_prefix}_m{mode}_{spec.tag}"
    run_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", run_name_raw)
    config = {
        "mode": int(mode),
        "dataset_id": int(spec.dataset_id),
        "dataset_name": str(spec.dataset_name),
        "trainer": str(spec.trainer),
        "plans_name": str(spec.plans_name),
        "patch_size": int(spec.patch_size),
        "batch_size": int(spec.batch_size),
    }

    kwargs: dict[str, Any] = {"project": str(args.swanlab_project), "config": config}
    key = "experiment_name"
    try:
        params = inspect.signature(swanlab.init).parameters
        key = "experiment_name" if "experiment_name" in params else ("name" if "name" in params else "experiment_name")
    except Exception:
        key = "experiment_name"
    kwargs[key] = run_name

    workspace = str(args.swanlab_workspace).strip()
    if workspace:
        kwargs["workspace"] = workspace

    try:
        run = swanlab.init(**kwargs)
    except Exception as init_error:
        log_fn(f"swanlab init failed, skip logging: {init_error!r} (run_name_key={key})")
        return None, None

    log_fn(f"swanlab run started: {run_name}")
    return swanlab, run


def swanlab_log(swanlab_mod: Any, run: Any, payload: dict[str, float], step: int, log_fn):
    if not payload:
        return
    if run is not None and hasattr(run, "log"):
        try:
            run.log(payload, step=step)
            return
        except TypeError:
            try:
                run.log(payload)
                return
            except Exception as e:
                log_fn(f"swanlab run.log failed, fallback to module log: {e!r}")
        except Exception as e:
            log_fn(f"swanlab run.log failed, fallback to module log: {e!r}")

    if hasattr(swanlab_mod, "log"):
        try:
            swanlab_mod.log(payload, step=step)
        except TypeError:
            try:
                swanlab_mod.log(payload)
            except Exception as e:
                log_fn(f"swanlab module.log failed: {e!r}")
        except Exception as e:
            log_fn(f"swanlab module.log failed: {e!r}")


def finish_swanlab_run(swanlab_mod: Any, run: Any, log_fn):
    try:
        if run is not None and hasattr(run, "finish"):
            run.finish()
            return
        if hasattr(swanlab_mod, "finish"):
            swanlab_mod.finish()
    except Exception as e:
        log_fn(f"swanlab finish failed, ignored: {e!r}")


def build_swanlab_line_logger(swanlab_mod: Any, run: Any, log_fn):
    step: Optional[int] = None
    auto_step = 0
    logged_records = 0

    def on_line(raw_line: str):
        nonlocal step, auto_step, logged_records
        line = raw_line.strip()
        if not line:
            return
        epoch = extract_epoch(line)
        if epoch is not None and epoch > 0:
            step = epoch
        metrics = extract_metrics_from_line(line)
        if not metrics:
            return
        if step is None:
            auto_step += 1
            current_step = auto_step
        else:
            current_step = step
        payload = dict(metrics)
        payload["epoch"] = float(current_step)
        swanlab_log(swanlab_mod, run, payload, step=current_step, log_fn=log_fn)
        logged_records += 1

    def count() -> int:
        return logged_records

    return on_line, count
