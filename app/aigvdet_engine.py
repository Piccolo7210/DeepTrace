"""
Loads the AIGVDet detector (two ResNet-50 branches + RAFT) once, off the UI thread, and
runs each analysis on a worker thread. Same shape as app/video_engine.py and
app/speech_engine.py (see app/engine.py's docstring for the tkinter-thread-safety
reasoning all of them share), plus one extra event while an analysis runs:

    ("analysis_progress", (done, total), request_id)   after every frame pair

Cost note: loading is a few seconds (checksum verification of ~230 MB, then ~0.2 GB of
weights); one analysis is ~0.5-2 minutes of CPU work (optical flow dominates), so
analyses are serialized: a second request while one is running is rejected, not queued.
"""

import json
import queue
import threading
import traceback

# torch MUST be imported before pandas on this Windows setup: with pandas first, torch's
# c10.dll fails with WinError 1114 (DLL initialization). aigvdet.detector imports pandas at
# module level, so load torch here — in the main thread, at app start — before any worker
# thread can import aigvdet (or anything else that pulls pandas in first).
import torch  # noqa: F401, E402, I001

from paths import AIGVDET_DIR, isolated_import, new_run_dir  # noqa: E402

DEFAULT_MODE = "accurate"  # standard/quick are unvalidated (see AIGVDET_LOCAL_HANDOFF.md 6.2)


class AIGVDetEngine:
    def __init__(self, detector_factory=None, report_function=None):
        """``detector_factory``/``report_function`` exist so tests can inject fakes."""
        self.events = queue.Queue()
        self.ready = False
        self.load_error = None
        self.missing_models = False  # True when the failure is "run pull_export.ps1"
        self.modes = {}  # mode name -> description, filled once loaded
        self._detector = None
        self._plot_report = report_function
        self._detector_factory = detector_factory
        self._busy = threading.Lock()

    def load_async(self):
        """
        Starts model loading on a background thread. Posts exactly one of
        ("models_ready", None, None) or ("models_failed", message, None) to
        self.events.
        """
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            if self._detector_factory is not None:
                self._detector = self._detector_factory()
            else:
                # The aigvdet package lives under AIGVDET/; isolated_import makes that the
                # only module tree on sys.path while it is imported (see paths.py). Only the
                # imports happen under that lock — other engines hold it for their whole
                # model load, and the weights here need nothing from sys.path.
                with isolated_import(AIGVDET_DIR):
                    from aigvdet.detector import MODES, ExportError, load_detector
                    from aigvdet.report import plot_report

                    self._plot_report = plot_report
                    self.modes = {name: mode.description for name, mode in MODES.items()}
                try:
                    self._detector = load_detector(device="cpu")
                except ExportError:
                    self.missing_models = True
                    raise
            self.ready = True
            self.events.put(("models_ready", None, None))
        except Exception as exc:
            traceback.print_exc()
            message = str(exc) or type(exc).__name__
            if self.missing_models and "pull_export" not in message:
                message = (
                    f"{message}\n\nFetch the model export with "
                    "AIGVDET\\tools\\pull_export.ps1."
                )
            self.load_error = message
            self.events.put(("models_failed", message, None))

    def analyze_async(self, video_path, mode=DEFAULT_MODE):
        """
        Starts one analysis on a background thread. Posts any number of
        ("analysis_progress", (done, total), request_id) followed by exactly one of
        ("analysis_done", result_dict, request_id) or ("analysis_failed", message,
        request_id).

        Returns request_id — the caller should remember it and ignore any event whose
        request_id doesn't match (the worker thread isn't cancellable, so a stray
        result from an abandoned earlier request is still possible).
        """
        request_id = object()
        if not self.ready:
            message = (
                f"AIGVDet models failed to load: {self.load_error}"
                if self.load_error
                else "AIGVDet models are not loaded yet."
            )
            self.events.put(("analysis_failed", message, request_id))
            return request_id
        if self._busy.locked():
            self.events.put((
                "analysis_failed",
                "Another video is still being analyzed. This detector runs one "
                "video at a time — wait for the current one to finish.",
                request_id,
            ))
            return request_id
        threading.Thread(
            target=self._analyze, args=(video_path, mode, request_id), daemon=True
        ).start()
        return request_id

    def _analyze(self, video_path, mode, request_id):
        with self._busy:
            try:
                run_dir = new_run_dir(video_path)

                def progress(done, total):
                    self.events.put(("analysis_progress", (done, total), request_id))

                result = self._detector.analyze(
                    video_path, mode=mode, window="whole", progress=progress, keep_crops=True
                )
                result["file"] = str(video_path)

                figure_path = run_dir / "analysis.png"
                self._plot_report(result, figure_path)
                result.pop("crops", None)  # uint8 arrays: used by the figure, not kept
                result["figure"] = str(figure_path)
                result["original_file"] = str(video_path)
                result["run_dir"] = str(run_dir)
                (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))

                self.events.put(("analysis_done", result, request_id))
            except Exception as exc:
                traceback.print_exc()
                # ValueError carries the user-facing reason ("Clip too short: 3 frames",
                # "... frames are too small for RAFT"), so it is shown as-is.
                message = str(exc) or type(exc).__name__
                if isinstance(exc, ValueError):
                    message = f"This video cannot be analyzed: {message}."
                self.events.put(("analysis_failed", message, request_id))
