"""
Loads the AEROBLADE models once, off the UI thread, and runs each analysis on a
worker thread so the ~40s-per-image CPU pipeline never blocks Tk's event loop.

tkinter widgets may only be touched from the main thread, so results cross from
worker threads to the UI via a queue.Queue that app_window.py drains with
root.after(...). Nothing in this module imports tkinter or touches a widget —
that separation is what makes analyze_async() safe to call from a background thread.
"""

import queue
import threading
import traceback

from paths import SRC_DIR, isolated_import, new_run_dir
from preprocess import prepare_input


class ForensicsEngine:
    def __init__(self):
        self.events = queue.Queue()
        self.ready = False
        self._aes = None
        self._threshold = None
        self._calibrated = None
        self._image_size = None
        self._detect = None

    def load_async(self):
        """
        Starts model loading on a background thread. Posts exactly one of
        ("models_ready", None) or ("models_failed", message) to self.events.
        """
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            # See paths.isolated_import's docstring: src/ and Speech/ both define a
            # bare-named config.py/detect.py, and this app also loads a
            # SpeechForensicsEngine whose _load() runs on its own thread at the same
            # startup moment — importing both trees without this would race on
            # sys.modules and silently hand one engine the other's module.
            with isolated_import(SRC_DIR):
                import config
                from autoencoders import load_autoencoders
                import detect as detect_module

                self._image_size = config.IMAGE_SIZE
                aes = load_autoencoders()
                if not aes:
                    raise RuntimeError(
                        "No autoencoder could be loaded. Check your internet connection "
                        "and Hugging Face Hub access, then restart the app."
                    )
                self._aes = aes
                self._threshold, self._calibrated = detect_module.load_threshold()
                self._detect = detect_module.detect
            self.ready = True
            self.events.put(("models_ready", None, None))
        except Exception as exc:
            traceback.print_exc()
            self.events.put(("models_failed", str(exc) or type(exc).__name__, None))

    def analyze_async(self, image_path):
        """
        Starts one analysis on a background thread. Posts exactly one of
        ("analysis_done", result_dict, request_id) or
        ("analysis_failed", message, request_id) to self.events.

        Returns request_id — the caller should remember it and ignore any event
        whose request_id doesn't match (see module docstring).
        """
        request_id = object()  # identity-unique per call; nothing needs its value
        if not self.ready:
            self.events.put(("analysis_failed", "Models are not loaded yet.", request_id))
            return request_id
        threading.Thread(
            target=self._analyze, args=(image_path, request_id), daemon=True
        ).start()
        return request_id

    def _analyze(self, image_path, request_id):
        try:
            run_dir = new_run_dir(image_path)
            prep = prepare_input(image_path, run_dir, size=self._image_size)
            result = self._detect(
                prep.prepared_path,
                aes=self._aes,
                threshold=self._threshold,
                calibrated=self._calibrated,
                out_dir=run_dir,
            )
            result["low_resolution_warning"] = prep.was_upscaled
            result["original_size"] = list(prep.original_size)
            result["original_file"] = str(image_path)
            result["run_dir"] = str(run_dir)
            self.events.put(("analysis_done", result, request_id))
        except Exception as exc:
            traceback.print_exc()
            self.events.put(("analysis_failed", str(exc) or type(exc).__name__, request_id))
