import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional

from source.train.specs import TrainModelSpec

_EPOCH_CKPT_RE = re.compile(r"^checkpoint_(\d+)e\.pth$")


def model_output_dir(results_root: Path, spec: TrainModelSpec, configuration: str, fold: str) -> Path:
    return results_root / spec.dataset_name / f"{spec.trainer}__{spec.plans_name}__{configuration}" / f"fold_{fold}"


def resolve_pretrained_checkpoint(
    results_root: Path,
    source_spec: TrainModelSpec,
    configuration: str,
    fold: str,
    policy: str,
) -> Path:
    fold_dir = model_output_dir(results_root, source_spec, configuration, fold)
    c_final = fold_dir / "checkpoint_final.pth"
    c_best = fold_dir / "checkpoint_best.pth"
    if policy == "final":
        if c_final.exists():
            return c_final
        raise FileNotFoundError(f"pretrained checkpoint (final) not found: {c_final}")
    if policy == "best":
        if c_best.exists():
            return c_best
        raise FileNotFoundError(f"pretrained checkpoint (best) not found: {c_best}")
    if c_final.exists():
        return c_final
    if c_best.exists():
        return c_best
    raise FileNotFoundError(f"pretrained checkpoint (auto final->best) not found: {fold_dir}")


def _checkpoint_readable(path: Path) -> tuple[bool, str]:
    try:
        if not path.exists():
            return False, "not found"
        if path.stat().st_size <= 0:
            return False, "empty file"
        with zipfile.ZipFile(path, "r") as zf:
            if not zf.infolist():
                return False, "empty checkpoint zip"
        return True, ""
    except zipfile.BadZipFile as e:
        return False, repr(e)
    except Exception as e:
        return False, repr(e)


def _epoch_from_ckpt_name(name: str) -> Optional[int]:
    m = _EPOCH_CKPT_RE.match(name)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _pick_max_epoch_ckpt(fold_dir: Path) -> Optional[Path]:
    pairs: list[tuple[int, Path]] = []
    for p in fold_dir.glob("checkpoint_*e.pth"):
        epoch = _epoch_from_ckpt_name(p.name)
        if epoch is not None:
            pairs.append((epoch, p))
    if not pairs:
        return None
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs[0][1]


def prepare_resume_checkpoint(fold_dir: Path, log_fn: Callable[[str], None]) -> Path:
    latest = fold_dir / "checkpoint_latest.pth"
    latest_ok, latest_msg = _checkpoint_readable(latest)
    if latest_ok:
        valid_candidates: list[Path] = [latest]
    else:
        valid_candidates = []

    if latest.exists() and not latest_ok:
        backup = fold_dir / f"checkpoint_latest.pth.corrupt.{time.strftime('%Y%m%d_%H%M%S')}"
        try:
            latest.rename(backup)
            log_fn(f"resume checkpoint broken, moved aside: {backup} ({latest_msg})")
        except Exception as e:
            log_fn(f"resume checkpoint broken, rename failed: {e!r}")

    max_epoch_ckpt = _pick_max_epoch_ckpt(fold_dir)
    candidates: list[Path] = []
    if max_epoch_ckpt is not None:
        candidates.append(max_epoch_ckpt)
    candidates.append(fold_dir / "checkpoint_final.pth")
    candidates.append(fold_dir / "checkpoint_best.pth")

    seen: set[str] = set()
    for cand in candidates:
        key = str(cand.resolve()) if cand.exists() else str(cand)
        if key in seen:
            continue
        seen.add(key)
        ok, _ = _checkpoint_readable(cand)
        if not ok:
            continue
        valid_candidates.append(cand)

    if valid_candidates:
        chosen = max(
            valid_candidates,
            key=lambda p: (p.stat().st_mtime, 1 if p == latest else 0),
        )
        if chosen == latest:
            return latest

        if latest_ok:
            log_fn(
                f"resume checkpoint latest is stale, newer valid checkpoint found: "
                f"{chosen.name} (mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(chosen.stat().st_mtime))})"
            )
        shutil.copy2(chosen, latest)
        copied_ok, copied_msg = _checkpoint_readable(latest)
        if not copied_ok:
            raise RuntimeError(
                f"resume fallback copy failed: copied {chosen.name} to checkpoint_latest.pth "
                f"but result is unreadable ({copied_msg})"
            )
        log_fn(f"resume fallback: use {chosen.name} as checkpoint_latest.pth")
        return latest

    raise RuntimeError(
        f"no valid resume checkpoint under {fold_dir}. "
        f"checked latest/max-epoch/final/best; latest_error={latest_msg}"
    )


def build_epoch_checkpoint_archiver(
    fold_dir: Path,
    save_every_epochs: int,
    extract_epoch_fn: Callable[[str], Optional[int]],
    log_fn: Callable[[str], None],
):
    last_seen_epoch: Optional[int] = None
    archived: set[int] = set()
    archive_count = 0

    def _archive(epoch_finished: int):
        nonlocal archive_count
        if save_every_epochs <= 0:
            return
        if epoch_finished <= 0 or epoch_finished % save_every_epochs != 0:
            return
        if epoch_finished in archived:
            return

        src = fold_dir / "checkpoint_latest.pth"
        dst = fold_dir / f"checkpoint_{epoch_finished}e.pth"
        if dst.exists():
            dst_ok, _ = _checkpoint_readable(dst)
            if dst_ok:
                archived.add(epoch_finished)
                return

        src_ok, src_msg = _checkpoint_readable(src)
        if not src_ok:
            log_fn(
                f"checkpoint archive skipped at epoch {epoch_finished}: "
                f"checkpoint_latest unreadable ({src_msg})"
            )
            return

        try:
            shutil.copy2(src, dst)
        except Exception as e:
            log_fn(f"checkpoint archive failed at epoch {epoch_finished}: {e!r}")
            return

        dst_ok, dst_msg = _checkpoint_readable(dst)
        if not dst_ok:
            try:
                dst.unlink()
            except Exception:
                pass
            log_fn(f"checkpoint archive invalid at epoch {epoch_finished}, removed: {dst.name} ({dst_msg})")
            return

        archived.add(epoch_finished)
        archive_count += 1
        log_fn(f"checkpoint archived: {dst.name}")

    def on_line(raw_line: str):
        nonlocal last_seen_epoch
        epoch = extract_epoch_fn(raw_line)
        if epoch is None or epoch <= 0:
            return
        if last_seen_epoch is None:
            last_seen_epoch = epoch
            return
        if epoch > last_seen_epoch:
            # When line reports "Epoch K", epoch K-1 is done.
            _archive(epoch - 1)
            last_seen_epoch = epoch

    def finalize(train_output: str, train_ok: bool):
        if not train_ok:
            return
        last_epoch: Optional[int] = None
        for raw_line in train_output.splitlines():
            epoch = extract_epoch_fn(raw_line)
            if epoch is not None and epoch > 0:
                if last_epoch is None or epoch > last_epoch:
                    last_epoch = epoch
        if last_epoch is not None:
            _archive(last_epoch)

    def count() -> int:
        return archive_count

    return on_line, finalize, count
