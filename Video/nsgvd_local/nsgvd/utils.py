"""Shared helpers: logging, crash-safe (atomic) file writes, hashing, progress/ETA.

Deliberately free of torch/cv2 imports so CPU-only stages start fast.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_FMT = "%(asctime)s %(levelname)-7s %(message)s"


def get_logger(name: str = "nsgvd", rank: int | None = None) -> logging.Logger:
    """Stdout logger; the rank tag keeps interleaved multi-GPU logs readable."""
    full = name if rank is None else f"{name}.rank{rank}"
    logger = logging.getLogger(full)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        tag = "" if rank is None else f"[gpu{rank}] "
        handler.setFormatter(logging.Formatter(_FMT.replace("%(message)s", tag + "%(message)s"), "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_session_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# ─────────────────────────── crash-safe writes ───────────────────────────
def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass  # not supported on every filesystem; the rename is still atomic


def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp{os.getpid()}")


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write-to-temp + fsync + rename: a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=1, sort_keys=False).encode("utf-8"))


def atomic_save_npy(path: str | Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    with open(tmp, "wb") as fh:
        np.save(fh, np.ascontiguousarray(array), allow_pickle=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def atomic_torch_save(path: str | Path, obj: Any) -> None:
    import torch  # local import keeps CPU-only stages torch-free

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    torch.save(obj, tmp)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a torn last line from a hard kill is skipped, not fatal
    return records


# ─────────────────────────── hashing / seeds ───────────────────────────
def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_int(text: str, bits: int = 32) -> int:
    """Process-independent hash (Python's hash() is salted per process)."""
    return int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:8], "big") % (1 << bits)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def natural_key(text: str) -> list:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def safe_name(text: str, max_len: int = 40) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")
    return cleaned[:max_len] or "x"


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def dir_size(path: str | Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def chunked(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


class RateMeter:
    """Prints `done/total | rate | ETA` lines (plan: progress with rate + ETA)."""

    def __init__(self, total: int, unit: str = "videos"):
        self.total, self.unit, self.done = total, unit, 0
        self.start = time.time()

    def update(self, n: int) -> str:
        self.done += n
        elapsed = max(time.time() - self.start, 1e-6)
        rate = self.done / elapsed
        remaining = (self.total - self.done) / rate if rate > 0 else float("inf")
        return (f"{self.done}/{self.total} {self.unit} | {rate:.2f} {self.unit}/s | "
                f"elapsed {_fmt_secs(elapsed)} | ETA {_fmt_secs(remaining)}")


def _fmt_secs(seconds: float) -> str:
    if seconds == float("inf"):
        return "?"
    seconds = int(seconds)
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"
