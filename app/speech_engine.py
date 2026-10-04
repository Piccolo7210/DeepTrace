"""
Loads the pretrained-SyncNet speech-forensics models once, off the UI thread, and
runs each analysis on a worker thread. Mirrors app/engine.py's shape exactly (see
that module's docstring for the tkinter-thread-safety reasoning it shares), but is a
deliberately separate class rather than a shared one: this engine's "ready"/loading
semantics are specific to the SyncNet+S3FD+AR-model stack and unrelated to the
AEROBLADE autoencoders app/engine.py manages -- merging them would only create
ambiguous shared state (what would `ready` mean if one domain's models loaded and
the other's didn't?) for no benefit. app/ui/app_window.py owns and polls both.
"""

import queue
import threading
import traceback

from paths import SPEECH_DIR, isolated_import, new_run_dir

DEVICE = "cpu"  # no CUDA locally; see Speech/config.py's own DEVICE note


class SpeechForensicsEngine:
    def __init__(self):
        self.events = queue.Queue()
        self.ready = False
        self._syncnet = None
        self._ar_model = None
        self._ckpt = None
        self._detect_module = None

    def load_async(self):
        """
        Starts model loading on a background thread. Posts exactly one of
        ("models_ready", None, None) or ("models_failed", message, None) to
        self.events.
        """
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            # See paths.isolated_import's docstring: Speech/ and src/ both define a
            # bare-named config.py/detect.py, and this app also loads a
            # ForensicsEngine whose _load() runs on its own thread at the same
            # startup moment — importing both trees without this would race on
            # sys.modules and silently hand one engine the other's module.
            with isolated_import(SPEECH_DIR):
                import config
                import detect_pretrained
                from models.ar_model import load_pretrained_ar_model
                from syncnet_wrapper import load_syncnet

                self._syncnet = load_syncnet(DEVICE)
                self._ar_model, self._ckpt = load_pretrained_ar_model(
                    config.PRETRAINED_MODEL_PATH, device=DEVICE
                )
                self._detect_module = detect_pretrained
            self.ready = True
            self.events.put(("models_ready", None, None))
        except Exception as exc:
            traceback.print_exc()
            self.events.put(("models_failed", str(exc) or type(exc).__name__, None))

    def analyze_async(self, video_path):
        """
        Starts one analysis on a background thread. Posts exactly one of
        ("analysis_done", result_dict, request_id) or
        ("analysis_failed", message, request_id) to self.events.

        Returns request_id — the caller should remember it and ignore any event
        whose request_id doesn't match (see app/engine.py's module docstring for why:
        the worker thread isn't cancellable, so a stray result from an abandoned
        earlier request is still possible).
        """
        request_id = object()
        if not self.ready:
            self.events.put(("analysis_failed", "Models are not loaded yet.", request_id))
            return request_id
        threading.Thread(
            target=self._analyze, args=(video_path, request_id), daemon=True
        ).start()
        return request_id

    def _analyze(self, video_path, request_id):
        try:
            run_dir = new_run_dir(video_path)
            work_dir = run_dir / "work"
            work_dir.mkdir(parents=True, exist_ok=True)

            result = self._detect_module.detect_pretrained(
                video_path, self._ar_model, self._ckpt, self._syncnet, work_dir,
                device=DEVICE, cleanup=True,  # cleans up work_dir, not run_dir itself
            )

            figure_path = run_dir / "analysis.png"
            self._detect_module.plot_report(result, figure_path)
            result["figure"] = str(figure_path)
            result["original_file"] = str(video_path)
            result["run_dir"] = str(run_dir)

            self.events.put(("analysis_done", result, request_id))
        except Exception as exc:
            traceback.print_exc()
            self.events.put(("analysis_failed", str(exc) or type(exc).__name__, request_id))
