"""Thin wrapper around the ``rclone`` CLI used to sync artefacts with Drive."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

LOGGER = logging.getLogger(__name__)
_RETRY_FLAGS = ("--retries", "3", "--low-level-retries", "10")


class RcloneError(RuntimeError):
    """Raised when an rclone invocation that must succeed fails."""


def _run(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    command = ["rclone", *args]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise RcloneError(
            f"{' '.join(command)} failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result


def upload_file(local: Path, remote: str) -> bool:
    """Copy ``local`` to the exact remote path ``remote``; never raises."""
    result = _run(["copyto", str(local), remote, *_RETRY_FLAGS], check=False)
    if result.returncode != 0:
        LOGGER.warning("Upload of %s failed: %s", local.name, result.stderr.strip())
    return result.returncode == 0


def download_file(remote: str, local: Path) -> bool:
    """Copy the remote file to ``local``; returns False if it is unavailable."""
    local.parent.mkdir(parents=True, exist_ok=True)
    result = _run(["copyto", remote, str(local), *_RETRY_FLAGS], check=False)
    return result.returncode == 0 and local.exists()


def exists(remote_file: str) -> bool:
    """Return True if ``remote_file`` exists."""
    directory, _, name = remote_file.rpartition("/")
    result = _run(["lsf", "--files-only", "--include", name, directory], check=False)
    return result.returncode == 0 and name in result.stdout.split("\n")


def list_names(remote_dir: str, recursive: bool = False, include: str | None = None) -> list[str]:
    """List file paths (relative to ``remote_dir``); raises on failure."""
    args = ["lsf", "--files-only"]
    if recursive:
        args.append("-R")
    if include:
        args += ["--include", include]
    args.append(remote_dir)
    return [line for line in _run(args).stdout.split("\n") if line]


def list_dirs(remote_dir: str) -> list[str]:
    """Names of the sub-folders of ``remote_dir``; raises on failure."""
    lines = _run(["lsf", "--dirs-only", remote_dir]).stdout.split("\n")
    return [line.rstrip("/") for line in lines if line]


def copy_files(remote_root: str, relative_paths: Sequence[str], local_dir: Path) -> None:
    """Download ``relative_paths`` (relative to ``remote_root``) into ``local_dir``; raises."""
    local_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(relative_paths))
        listing = Path(handle.name)
    try:
        args = ["copy", remote_root, str(local_dir), "--files-from", str(listing)]
        _run([*args, "--transfers", "8", *_RETRY_FLAGS])
    finally:
        listing.unlink(missing_ok=True)


def copy_dir(remote_dir: str, local_dir: Path, include: str | None = None) -> None:
    """Download a remote folder (optionally only files matching ``include``); raises."""
    local_dir.mkdir(parents=True, exist_ok=True)
    args = ["copy", remote_dir, str(local_dir)]
    if include:
        args += ["--include", include]
    _run([*args, *_RETRY_FLAGS])
    
def delete_file(remote_file: str) -> bool:
    """Delete one remote file; returns False if it was not there (never raises)."""
    return _run(["deletefile", remote_file], check=False).returncode == 0
