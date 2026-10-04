"""
Speech Forensics screen: instructions -> pick a file -> analyze -> results.
Mirrors image_view.py's state machine (self.state, one _render_<state>() per state),
adapted for a video-file input (metadata probe + thumbnail instead of an image
preview) and the pretrained-SyncNet pipeline's result shape.
"""

from pathlib import Path
from tkinter import filedialog, ttk

import cv2
from PIL import Image

from ui.widgets import (
    ScrollableFrame,
    load_scaled_photo,
    make_kv_table,
    make_verdict_banner,
    make_warning_banner,
    scale_photo_image,
)

SUPPORTED_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm", "*.mpg")

# The bundled checkpoint's window size (Speech/config.py's checkpoint has
# seq_len=50) -- used here only for the UI's own short-clip hint, not for scoring.
MIN_FRAMES_HINT = 50

INSTRUCTIONS_TEXT = (
    "• A short clip (a few seconds) with one clearly visible, front-facing "
    "speaking face and its original audio track.\n"
    "• mp4 / avi / mov / mkv / webm / mpg.\n"
    "• This detector was trained on a small set of speakers (GRID corpus). An "
    "unfamiliar voice or face may occasionally score higher than expected — "
    "treat results as indicative, not definitive.\n"
)


class SpeechView(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        self.state = "instructions"
        self.selected_path = None
        self.result = None
        self.error_message = None
        self._pending_request_id = None
        self._photo_refs = []  # must outlive display or Tk shows a blank image

        self._build_chrome()
        self.content = ttk.Frame(self)
        self.content.pack(fill="both", expand=True)
        self._render()

    def _build_chrome(self):
        """Header bar that stays put across every state — see image_view.py's
        identical chrome for why (back-to-home must always work, including
        mid-analysis, since the worker thread isn't cancellable)."""
        bar = ttk.Frame(self, padding=(16, 12))
        bar.pack(fill="x")
        ttk.Button(bar, text="← Back to Home", command=self.app.show_home).pack(side="left")
        ttk.Label(bar, text="Speech Forensics", style="Header.TLabel").pack(side="left", padx=20)
        ttk.Separator(self).pack(fill="x")

    def _clear_content(self):
        for widget in self.content.winfo_children():
            widget.destroy()

    def _render(self):
        self._clear_content()
        {
            "instructions": self._render_instructions,
            "selected": self._render_selected,
            "running": self._render_running,
            "results": self._render_results,
            "error": self._render_error,
        }[self.state]()

    # ---- instructions ------------------------------------------------------

    def _render_instructions(self):
        frame = ttk.Frame(self.content, padding=30)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Video requirements", font=("Segoe UI", 13, "bold")).pack(
            anchor="w"
        )
        ttk.Label(
            frame, text=INSTRUCTIONS_TEXT, wraplength=820, justify="left", font=("Segoe UI", 10)
        ).pack(anchor="w", pady=(10, 24))

        if not self.app.speech_engine.ready:
            ttk.Label(
                frame,
                text="Detection models are still loading — you can choose a video "
                "now; analysis will start once they're ready.",
                foreground="#7a5b00",
                font=("Segoe UI", 9, "italic"),
            ).pack(anchor="w", pady=(0, 12))

        ttk.Button(
            frame, text="Choose Video…", style="Primary.TButton", command=self._choose_file
        ).pack(anchor="w")

    def _choose_file(self):
        filetypes = [("Videos", " ".join(SUPPORTED_EXTS)), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Choose a video", filetypes=filetypes)
        if not path:
            return
        self.selected_path = path
        self.state = "selected"
        self._render()

    # ---- selected ------------------------------------------------------------

    def _probe_video(self, path):
        """Reads metadata + a middle-frame thumbnail via cv2.VideoCapture. There's
        no video-preview widget in stock Tk, so a static thumbnail is the best cheap
        option — full playback isn't worth the extra dependency for this screen."""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("Could not open this file as a video.")
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frame_count > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            ok, bgr = cap.read()
        finally:
            cap.release()

        thumb = None
        if ok:
            thumb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        duration = frame_count / fps if fps > 0 else None
        return {
            "frame_count": frame_count, "fps": fps,
            "width": width, "height": height, "duration": duration, "thumb": thumb,
        }

    def _render_selected(self):
        frame = ttk.Frame(self.content, padding=30)
        frame.pack(fill="both", expand=True)

        try:
            meta = self._probe_video(self.selected_path)
        except Exception as exc:
            self.error_message = f"Could not open this video:\n{exc}"
            self.state = "error"
            self._render()
            return

        if meta["thumb"] is not None:
            photo = scale_photo_image(meta["thumb"], max_width=380)
            self._photo_refs.append(photo)
            ttk.Label(frame, image=photo).pack(pady=(0, 12))

        ttk.Label(
            frame, text=Path(self.selected_path).name, font=("Segoe UI", 10, "bold")
        ).pack()

        rows = [
            ("Resolution", f"{meta['width']} × {meta['height']} px" if meta["width"] else "unknown"),
            ("Frames", str(meta["frame_count"]) if meta["frame_count"] > 0 else "unknown"),
            ("Frame rate", f"{meta['fps']:.1f} fps" if meta["fps"] else "unknown"),
            ("Duration", f"{meta['duration']:.1f} s" if meta["duration"] else "unknown"),
        ]
        make_kv_table(frame, rows).pack(pady=(12, 12))

        if 0 < meta["frame_count"] < MIN_FRAMES_HINT:
            make_warning_banner(
                frame,
                f"Only {meta['frame_count']} frames, below the model's "
                f"~{MIN_FRAMES_HINT}-frame window size. A result will still be "
                "produced, from a single undersized window rather than several "
                "averaged ones — treat it as lower-confidence.",
            ).pack(fill="x", pady=(0, 12))

        btn_row = ttk.Frame(frame)
        btn_row.pack(pady=10)
        ttk.Button(
            btn_row, text="Choose Different Video…", command=self._choose_file
        ).pack(side="left", padx=6)
        ttk.Button(
            btn_row, text="Analyze", style="Primary.TButton", command=self._start_analysis
        ).pack(side="left", padx=6)

    def _start_analysis(self):
        self.state = "running"
        self._render()
        self._pending_request_id = self.app.speech_engine.analyze_async(self.selected_path)

    # ---- running ---------------------------------------------------------------

    def _render_running(self):
        frame = ttk.Frame(self.content, padding=30)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Analyzing…", font=("Segoe UI", 14, "bold")).pack(
            pady=(80, 10)
        )
        ttk.Label(
            frame,
            text="Face detection, lip-sync feature extraction, then anomaly "
            "scoring — time depends on clip length.",
            font=("Segoe UI", 9),
        ).pack(pady=(0, 20))
        progress = ttk.Progressbar(frame, mode="indeterminate", length=320)
        progress.pack()
        progress.start(12)
        self._progress = progress  # keep alive; also stops implicitly when destroyed

    # ---- results ----------------------------------------------------------------

    def _render_results(self):
        result = self.result
        scroll = ScrollableFrame(self.content)
        scroll.pack(fill="both", expand=True)
        frame = ttk.Frame(scroll.body, padding=24)
        frame.pack(fill="both", expand=True)

        make_verdict_banner(
            frame, result["verdict"], level=result.get("verdict_level")
        ).pack(fill="x", pady=(0, 16))

        # Unlike the image module (which deliberately shows no numbers — see
        # widgets.make_verdict_banner — because Delta_Min is an uncalibrated score),
        # these two ARE directly interpretable physical measurements: the offset is
        # "audio leads/lags video by N frames" and the confidence is SyncNet's own
        # separation margin. They corroborate the verdict independently of the AR
        # model's threshold, which matters here because that threshold is the
        # weakest link (see detect_pretrained.DEFAULT_THRESHOLD_KEY).
        offset = result["syncnet_offset"]
        offset_text = (
            "0 frames (in sync)" if offset == 0
            else f"{offset:+d} frames ({abs(offset) / 25:.2f}s)"
        )
        make_kv_table(
            frame,
            [
                ("Audio-visual offset", offset_text),
                ("Sync confidence", f"{result['syncnet_confidence']:.2f}"),
                ("Frames analyzed", f"{result['num_frames']} ({result['num_windows']} windows)"),
            ],
            title="Measurements",
        ).pack(anchor="w", pady=(0, 16))

        if result.get("threshold_suspect"):
            make_warning_banner(
                frame,
                "The threshold in use is lower than a lower percentile in the "
                "model's own calibration table, so it is not a true percentile and "
                "will flag a large share of genuine video. Treat a flagged verdict "
                "with suspicion.",
            ).pack(fill="x", pady=(0, 12))

        if result.get("short_clip_warning"):
            make_warning_banner(
                frame,
                f"Clip had only {result['num_frames']} scored frames, below the "
                "model's window size — scored as a single undersized window rather "
                "than several averaged ones. Treat this verdict as lower-confidence.",
            ).pack(fill="x", pady=(0, 12))

        for region in result.get("regions", []):
            make_warning_banner(
                frame,
                f"Sustained desync detected from {region['start_sec']:.1f}s to "
                f"{region['end_sec']:.1f}s (frames {region['start_frame']}–"
                f"{region['end_frame']}, peak score {region['peak_score']:.2f}). "
                "Review this section specifically — the rest of the clip scored "
                "within the normal range.",
            ).pack(fill="x", pady=(0, 12))

        if result.get("num_face_tracks", 1) > 1:
            make_warning_banner(
                frame,
                f"{result['num_face_tracks']} separate faces were detected in this "
                "clip; the most-visible one was analyzed. Other faces were not "
                "checked.",
            ).pack(fill="x", pady=(0, 12))

        if not result.get("threshold_calibrated", True):
            make_warning_banner(
                frame,
                "The bundled model's threshold was not validated against a desync "
                "test — treat this verdict as provisional.",
            ).pack(fill="x", pady=(0, 12))

        fig_path = result.get("figure")
        if fig_path and Path(fig_path).exists():
            ttk.Label(frame, text="Analysis figure", font=("Segoe UI", 11, "bold")).pack(
                anchor="w", pady=(6, 6)
            )
            photo = load_scaled_photo(fig_path, max_width=900)
            self._photo_refs.append(photo)
            ttk.Label(frame, image=photo).pack()

        btn_row = ttk.Frame(frame)
        btn_row.pack(pady=(24, 10))
        ttk.Button(btn_row, text="Analyze Another", command=self._reset).pack(
            side="left", padx=6
        )
        ttk.Button(btn_row, text="← Back to Home", command=self.app.show_home).pack(
            side="left", padx=6
        )

    def _reset(self):
        self.selected_path = None
        self.result = None
        self._pending_request_id = None
        self._photo_refs.clear()
        self.state = "instructions"
        self._render()

    # ---- error --------------------------------------------------------------------

    def _render_error(self):
        frame = ttk.Frame(self.content, padding=30)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame, text="Something went wrong", font=("Segoe UI", 14, "bold")
        ).pack(pady=(50, 10))
        ttk.Label(
            frame,
            text=self.error_message or "Unknown error.",
            wraplength=820,
            justify="left",
        ).pack(pady=(0, 20))
        ttk.Button(frame, text="Try Again", command=self._reset).pack()

    # ---- engine events --------------------------------------------------------------

    def on_engine_event(self, event, payload, request_id=None):
        if event in ("analysis_done", "analysis_failed"):
            if request_id != self._pending_request_id:
                return  # stray result from an abandoned earlier request
            if event == "analysis_done":
                self.result = payload
                self.state = "results"
            else:
                self.error_message = payload
                self.state = "error"
            self._render()
