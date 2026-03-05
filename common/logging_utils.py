import time


def log(msg: str):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def banner(title: str):
    log("=" * 60)
    log(title)
    log("=" * 60)
