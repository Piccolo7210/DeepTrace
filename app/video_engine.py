"""
Loads the NSG video detector once, off the UI thread, and runs each analysis on a
worker thread. Mirrors app/speech_engine.py's shape exactly (see app/engine.py's
docstring for the tkinter-thread-safety reasoning all three share), and is a
separate class for the same reason: "ready" here means the 552 M-parameter
guided-diffusion score network and the gradient-boosting head are both loaded,
which has nothing to do with the other two modules' model stacks.

Cost note: loading takes ~20-40 s and ~2.2 GB of RAM, and one analysis is
0.5-1.5 min of CPU work, so analyses are serialized (concurrency 1) — a second
request while one is running is rejected rather than queued, since two
concurrent UNet passes would thrash a 16 GB machine.
"""

import queue
import threading
import traceback

from paths import VIDEO_DIR, isolated_import, new_run_dir

DEVICE = "cpu"  # no CUDA locally; Video/config.py's DEVICE note applies


class VideoForensicsEngine:
    def __init__(self):
        self.events = queue.Queue()
        self.ready = False
        self.load_error = None
        self._detector = None
        self._detect_module = None
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
            # src/, Speech/ and Video/ all define a bare config.py, and Video/ also
            # has a detect_nsgvd.py that imports `config` and `predictor` by bare
            # name — see paths.isolated_import for why each tree must be the only
            # one live on sys.path while it is imported.
            with isolated_import(VIDEO_DIR):
                import detect_nsgvd

                self._detector = detect_nsgvd.load_detector(device=DEVICE)
                self._detect_module = detect_nsgvd
            self.ready = True
            self.events.put(("models_ready", None, None))
        except Exception as exc:
            traceback.print_exc()
            self.load_error = str(exc) or type(exc).__name__
            self.events.put(("models_failed", self.load_error, None))

    def analyze_async(self, video_path):
        """
        Starts one analysis on a background thread. Posts exactly one of
        ("analysis_done", result_dict, request_id) or
        ("analysis_failed", message, request_id) to self.events.

        Returns request_id — the caller should remember it and ignore any event
        whose request_id doesn't match (see app/engine.py's module docstring: the
        worker thread isn't cancellable, so a stray result from an abandoned
        earlier request is still possible).
        """
        request_id = object()
        if not self.ready:
            message = (
                f"Video models failed to load: {self.load_error}"
                if self.load_error
                else "Video models are not loaded yet."
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
            target=self._analyze, args=(video_path, request_id), daemon=True
        ).start()
        return request_id

    def _analyze(self, video_path, request_id):
        with self._busy:
            try:
                run_dir = new_run_dir(video_path)
                result = self._detector.analyze(video_path)

                figure_path = run_dir / "analysis.png"
                self._detect_module.plot_report(result, figure_path)
                result["figure"] = str(figure_path)
                result["original_file"] = str(video_path)
                result["run_dir"] = str(run_dir)

                self.events.put(("analysis_done", result, request_id))
            except Exception as exc:
                traceback.print_exc()
                # VideoDecodeError carries the user-facing reason ("only 3
                # decodable frames (< 8)"), so its message is shown as-is.
                message = str(exc) or type(exc).__name__
                if type(exc).__name__ == "VideoDecodeError":
                    message = f"This video cannot be analyzed: {message}."
                self.events.put(("analysis_failed", message, request_id))
