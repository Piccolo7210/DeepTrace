"""
Filesystem locations for the desktop app, plus the `src/`/`Speech/`/`Video/` import shim.

This module is intentionally dependency-free (stdlib only) so it can be imported
first, before anything that pulls in torch/diffusers, to make PROJECT_ROOT and
sys.path available immediately.
"""

import sys
import threading
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
SPEECH_DIR = PROJECT_ROOT / "Speech"
VIDEO_DIR = PROJECT_ROOT / "Video"
AIGVDET_DIR = PROJECT_ROOT / "AIGVDET"
APP_RESULTS_DIR = PROJECT_ROOT / "results" / "app"

_IMPORT_LOCK = threading.Lock()


class isolated_import:
    """
    Context manager: makes `module_dir` the only self-contained module tree on
    sys.path for the duration of the block, then undoes whatever sys.path/sys.modules
    changes it caused.

    src/, Speech/ and Video/ are each internally self-contained: their own *.py files
    import each other with bare names (`from config import ...`, not `from src.config
    import ...` / `from Speech.config import ...`), since none is a real package.
    That's fine in isolation, but the trees happen to define same-named modules
    (all three have a config.py; src/ and Speech/ both have a detect.py) — and Python
    caches imports globally, by bare name, in sys.modules for the life of the process.
    app/engine.py, app/speech_engine.py and app/video_engine.py each load their models
    on a background thread started at app startup, so without this, whichever tree's
    config.py/detect.py gets imported first "wins" sys.modules and the other engines
    silently receive the WRONG module for the rest of the process (an ImportError for
    a missing attribute, not an obvious crash at the call site — this exact failure
    mode showed up during integration testing).

    Fix: only one of {src/, Speech/, Video/} is ever "live" in sys.modules at a time —
    a lock serializes the engines' _load() calls (a one-time app-startup cost, not
    a per-analysis one), and on exit, only the newly-added modules that actually
    live under `module_dir` are purged from sys.modules — third-party packages
    (torch, numpy, cv2, ...) that happened to get imported for the first time inside
    the block are deliberately left cached, both because they're expensive to
    re-init and because re-importing torch as a second, independent instance in the
    same process is its own source of bugs. Whatever functions/objects/values a
    caller pulls out of the `with` block (store them as locals/self attributes
    before it exits) keep working fine afterward — only the module-registry
    bookkeeping is undone, not the already-constructed Python objects.
    """

    def __init__(self, module_dir):
        self.module_dir = str(Path(module_dir).resolve())

    def __enter__(self):
        _IMPORT_LOCK.acquire()
        self._before = set(sys.modules)
        if self.module_dir not in sys.path:
            sys.path.insert(0, self.module_dir)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.module_dir in sys.path:
                sys.path.remove(self.module_dir)
            for name in set(sys.modules) - self._before:
                mod = sys.modules.get(name)
                mod_file = getattr(mod, "__file__", None)
                if not mod_file:
                    continue
                try:
                    is_local = Path(mod_file).resolve().is_relative_to(self.module_dir)
                except Exception:
                    # Path(mod_file) can be malformed for oddball loaders (frozen/zip
                    # imports, namespace packages); safest default is "not ours".
                    is_local = False
                if is_local:
                    del sys.modules[name]
        finally:
            _IMPORT_LOCK.release()
        return False


def new_run_dir(source_filename):
    """
    One directory per analysis run: results/app/<timestamp>_<stem>/. Timestamped so
    two uploads named 1.png don't collide or overwrite each other, and namespaced
    under results/app/ so GUI runs never mix into the results/ the CLI and paper
    figures use.
    """
    stem = Path(source_filename).stem or "image"
    safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:60]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = APP_RESULTS_DIR / f"{timestamp}_{safe_stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
