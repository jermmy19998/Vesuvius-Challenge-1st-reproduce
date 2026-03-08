import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from pathlib import Path
from typing import Callable, Optional

from .logging_utils import log


@dataclass
class CommandProgress:
    label: str
    total: Optional[int] = None
    unit: str = "steps"
    parse_epoch_from_output: bool = False
    watch_dir: Optional[Path] = None
    watch_glob: str = "*"
    heartbeat_sec: int = 20
    emit_every: int = 1


_EPOCH_RE = re.compile(r"(?i)\bepoch(?:\s+|:|=)\s*(\d+)\b")
_EPOCH_RATIO_RE = re.compile(r"(?i)\bepoch(?:\s+|:|=)\s*(\d+)\s*/\s*(\d+)\b")


def _format_progress_bar(
    current: int,
    total: Optional[int],
    tick: int,
    unit: str,
    width: int = 24,
) -> str:
    if total is not None and total > 0:
        safe_cur = max(0, min(current, total))
        ratio = safe_cur / total
        filled = int(ratio * width)
        return (
            f"[{'#' * filled}{'.' * (width - filled)}] "
            f"{ratio * 100:6.2f}% ({safe_cur}/{total} {unit})"
        )

    # Indeterminate bar when total is unknown.
    pos = tick % width
    chars = ["." for _ in range(width)]
    chars[pos] = ">"
    return f"[{''.join(chars)}] ??.??% ({current}/? {unit})"


def _count_progress_files(progress: CommandProgress) -> Optional[int]:
    if progress.watch_dir is None:
        return None
    try:
        return sum(1 for _ in progress.watch_dir.glob(progress.watch_glob))
    except Exception:
        return None


def _parse_epoch_from_line(line: str) -> tuple[Optional[int], Optional[int]]:
    m_ratio = _EPOCH_RATIO_RE.search(line)
    if m_ratio:
        return int(m_ratio.group(1)), int(m_ratio.group(2))
    m = _EPOCH_RE.search(line)
    if m:
        return int(m.group(1)), None
    return None, None


def resolve_command(cmd_name: str) -> str:
    direct = shutil.which(cmd_name)
    if direct:
        return direct

    py_bin = Path(sys.executable).resolve().parent
    candidates = [py_bin / cmd_name, py_bin / f"{cmd_name}.exe"]
    for c in candidates:
        if c.exists():
            return str(c)

    raise FileNotFoundError(
        f"{cmd_name} not found. Current python: {sys.executable}. "
        "Install nnunetv2 in this environment."
    )


def _safe_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_") or "command"


def _write_command_log(
    logs_dir: Optional[Path],
    name: str,
    cmd_display: str,
    returncode: int,
    stdout: str,
    stderr: str,
) -> Optional[Path]:
    if logs_dir is None:
        return None
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{_safe_name(name)}_{ts}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"name: {name}\n")
        f.write(f"returncode: {returncode}\n")
        f.write(f"command: {cmd_display}\n\n")
        if stdout:
            f.write("===== STDOUT =====\n")
            f.write(stdout)
            if not stdout.endswith("\n"):
                f.write("\n")
        if stderr:
            f.write("\n===== STDERR =====\n")
            f.write(stderr)
            if not stderr.endswith("\n"):
                f.write("\n")
    return log_path


def run_command(
    cmd: str | list[str],
    name: str,
    timeout_sec: Optional[int] = None,
    logs_dir: Optional[Path] = None,
    env: Optional[dict] = None,
    cwd: Optional[Path] = None,
    shell: bool = False,
    progress: Optional[CommandProgress] = None,
    suppress_output_substrings: Optional[list[str]] = None,
    on_output_line: Optional[Callable[[str], None]] = None,
) -> tuple[bool, str]:
    if isinstance(cmd, list):
        cmd_display = " ".join(shlex.quote(str(x)) for x in cmd)
    else:
        cmd_display = cmd
    log(f"{name} | {cmd_display}")

    q: Queue[str] = Queue()
    captured: list[str] = []
    timeout_hit = False
    suppress_terms = [x for x in (suppress_output_substrings or []) if x]
    callback_warning_printed = False

    def _reader(pipe):
        if pipe is None:
            return
        for line in iter(pipe.readline, ""):
            q.put(line)
        pipe.close()

    try:
        p = subprocess.Popen(
            cmd,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=str(cwd) if cwd is not None else None,
            bufsize=1,
        )
    except Exception as e:
        stderr = f"{name} failed to start: {e}"
        log_path = _write_command_log(logs_dir, name, cmd_display, 1, "", stderr)
        log(f"{name} failed (returncode=1)")
        if log_path is not None:
            log(f"{name} full log: {log_path}")
        print(stderr, flush=True)
        return False, stderr
    t = threading.Thread(target=_reader, args=(p.stdout,), daemon=True)
    t.start()

    started = time.time()
    last_progress_emit = started
    last_watch_poll = started
    progress_tick = 0
    progress_current = 0
    progress_total = progress.total if progress is not None else None
    last_reported_progress = 0

    if progress is not None:
        bar = _format_progress_bar(progress_current, progress_total, progress_tick, progress.unit)
        log(f"{name} progress | {progress.label} {bar} | elapsed=0s")

    while True:
        try:
            line = q.get(timeout=0.5)
            suppressed = any(term in line for term in suppress_terms)
            if not suppressed:
                captured.append(line)
                print(line, end="", flush=True)
                if on_output_line is not None:
                    try:
                        on_output_line(line)
                    except Exception as callback_error:
                        if not callback_warning_printed:
                            log(f"{name} output callback error, ignored: {callback_error!r}")
                            callback_warning_printed = True

            if (not suppressed) and progress is not None and progress.parse_epoch_from_output:
                epoch_cur, epoch_total = _parse_epoch_from_line(line)
                if epoch_total is not None and epoch_total > 0:
                    progress_total = epoch_total
                if epoch_cur is not None and epoch_cur > 0 and epoch_cur >= progress_current:
                    progress_current = epoch_cur
                    should_emit = (
                        progress_current == 1
                        or (progress_total is not None and progress_current >= progress_total)
                        or (progress.emit_every > 0 and progress_current % progress.emit_every == 0)
                    )
                    if should_emit:
                        progress_tick += 1
                        bar = _format_progress_bar(
                            progress_current,
                            progress_total,
                            progress_tick,
                            progress.unit,
                        )
                        log(
                            f"{name} progress | {progress.label} {bar} "
                            f"| elapsed={int(time.time() - started)}s"
                        )
                        last_reported_progress = progress_current
                        last_progress_emit = time.time()
        except Empty:
            pass

        if timeout_sec is not None and (time.time() - started) > timeout_sec and p.poll() is None:
            timeout_hit = True
            p.kill()

        now = time.time()
        if progress is not None and progress.watch_dir is not None and (now - last_watch_poll) >= 2:
            watched = _count_progress_files(progress)
            if watched is not None and watched >= progress_current:
                progress_current = watched
                should_emit_watch = (
                    progress_current > last_reported_progress
                    and (
                        progress_current == 1
                        or (progress_total is not None and progress_current >= progress_total)
                        or (progress.emit_every > 0 and progress_current % progress.emit_every == 0)
                    )
                )
                if should_emit_watch:
                    progress_tick += 1
                    bar = _format_progress_bar(progress_current, progress_total, progress_tick, progress.unit)
                    log(
                        f"{name} progress | {progress.label} {bar} "
                        f"| elapsed={int(now - started)}s"
                    )
                    last_reported_progress = progress_current
                    last_progress_emit = now
            last_watch_poll = now

        if (
            progress is not None
            and progress.heartbeat_sec > 0
            and (now - last_progress_emit) >= progress.heartbeat_sec
        ):
            progress_tick += 1
            bar = _format_progress_bar(progress_current, progress_total, progress_tick, progress.unit)
            log(
                f"{name} progress | {progress.label} {bar} "
                f"| elapsed={int(now - started)}s"
            )
            last_reported_progress = progress_current
            last_progress_emit = now

        if p.poll() is not None:
            break

    t.join(timeout=2)
    while True:
        try:
            line = q.get_nowait()
            if any(term in line for term in suppress_terms):
                continue
            captured.append(line)
            print(line, end="", flush=True)
            if on_output_line is not None:
                try:
                    on_output_line(line)
                except Exception as callback_error:
                    if not callback_warning_printed:
                        log(f"{name} output callback error, ignored: {callback_error!r}")
                        callback_warning_printed = True
        except Empty:
            break

    returncode = p.returncode if p.returncode is not None else 1
    stdout = "".join(captured)
    stderr = ""
    if timeout_hit:
        returncode = 124
        stderr = f"{name} timeout after {timeout_sec}s"

    log_path = _write_command_log(logs_dir, name, cmd_display, returncode, stdout, stderr)
    merged = f"{stdout}\n{stderr}"

    if returncode != 0:
        if progress is not None:
            progress_tick += 1
            bar = _format_progress_bar(progress_current, progress_total, progress_tick, progress.unit)
            log(
                f"{name} progress | {progress.label} {bar} "
                f"| elapsed={int(time.time() - started)}s (failed)"
            )
        log(f"{name} failed (returncode={returncode})")
        if log_path is not None:
            log(f"{name} full log: {log_path}")
        if stderr.strip():
            print(stderr[-4000:], flush=True)
        elif stdout.strip():
            print(stdout[-4000:], flush=True)
        return False, merged

    if progress is not None:
        if progress_total is not None and progress_total > 0:
            progress_current = max(progress_current, progress_total)
        progress_tick += 1
        bar = _format_progress_bar(progress_current, progress_total, progress_tick, progress.unit)
        log(
            f"{name} progress | {progress.label} {bar} "
            f"| elapsed={int(time.time() - started)}s (done)"
        )

    log(f"{name} done")
    if log_path is not None:
        log(f"{name} full log: {log_path}")
    return True, merged
