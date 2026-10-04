"""
Video detector screen: instructions -> pick a file -> analyze -> results.
Mirrors speech_view.py's state machine (self.state, one _render_<state>() per
state) and reuses its metadata-probe/thumbnail approach, since both screens take
a video file; what differs is the result shape (one calibrated probability from
the NSG detector instead of lip-sync measurements) and the much longer runtime.

One class serves both video detectors, configured by a *profile* (NSG_PROFILE, the
default, or AIGVDET_PROFILE): which engine to call, the texts, the minimum clip length
and the results layout. The NSG profile reproduces the original NSG-only screen; the
AIGVDet profile adds a mode choice, a determinate progress bar (driven by the engine's
``analysis_progress`` events), per-branch scores, caveats and limits.
"""

import tkinter as tk
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

SUPPORTED_EXTS = (
    "*.mp4", "*.mov", "*.m4v", "*.webm", "*.mkv", "*.avi",
    "*.mpg", "*.mpeg", "*.wmv", "*.flv", "*.3gp", "*.ts", "*.gif",
)

# The detector scores 8 frames sampled from the first 95 (Video/config.py's
# FRAME_WINDOW) — used here only for the UI's own hints, never for scoring.
MIN_FRAMES = 8
FRAME_WINDOW = 95

INSTRUCTIONS_TEXT = (
    "• Any video with at least 8 frames — mp4 / mov / webm / mkv / avi / gif.\n"
    "• The detector samples 8 frames from the first 95 of the clip, so a long "
    "video is judged on its opening seconds. For a specific moment, trim the clip "
    "to that part first.\n"
    "• Analysis runs entirely on this computer's CPU and takes roughly half a "
    "minute to a minute and a half per video.\n"
    "• It looks for physically implausible spatiotemporal dynamics (a diffusion "
    "model's score field), not for faces or voices — any subject matter is fine.\n"
)


NSG_PROFILE = {
    "key": "nsg",
    "header": "Video Forensics · NSG-based",
    "engine_attr": "video_engine",
    "min_frames": 8,
    "instructions": INSTRUCTIONS_TEXT,
    "limitations": None,  # the NSG screen shows no limitations banner
    "loading_note": (
        "Detection models are still loading (about 2 GB of weights) — "
        "you can choose a video now; analysis will start once they're ready."
    ),
    "running_text": (
        "Sampling 8 frames, then running each through the diffusion score "
        "network on the CPU — typically half a minute to a minute and a half. "
        "The window stays responsive; you can go back to Home."
    ),
}

# UI hints only (never used for scoring): the detector's crop size and RAFT's minimum side.
AIGVDET_CROP = 448
AIGVDET_RAFT_MIN_SIDE = 128

AIGVDET_MODES = [
    ("accurate", "Accurate (default) — 8 frame pairs, native optical flow; about 1–2 minutes"),
    ("benchmark", "Benchmark — 16 pairs, the validated protocol; about 2–4 minutes"),
    ("standard", "Standard — faster flow, 12 pairs; accuracy NOT measured"),
    ("quick", "Quick — fastest, 8 pairs; accuracy NOT measured"),
]

AIGVDET_PROFILE = {
    "key": "aigvdet",
    "header": "Video Forensics · AIGVDet-based",
    "engine_attr": "aigvdet_engine",
    "min_frames": 5,
    "instructions": (
        "• Any video with at least 5 frames — mp4 / mov / webm / mkv / avi / gif.\n"
        "• The detector takes pairs of neighbouring frames spread over the whole clip "
        "(8 by default). Each pair gets an RGB score (single-frame generation artefacts) and "
        "an optical-flow score (unnatural motion); the two are fused and averaged.\n"
        "• Analysis runs entirely on this computer's CPU. Optical flow dominates the time: "
        "roughly 1–2 minutes for a normal clip, less for small ones.\n"
        "• The networks look at a 448×448 crop from the centre of each frame. Clips smaller "
        "than that are zero-padded, which the detector handles worse.\n"
    ),
    "limitations": None,  # no banner on the instructions screen (limits show with each result)
    "loading_note": (
        "AIGVDet models are still loading — you can choose a video now; analysis will "
        "start once they're ready."
    ),
    "running_text": (
        "Computing optical flow for each frame pair and scoring both networks on the CPU. "
        "The window stays responsive; you can go back."
    ),
}


class VideoView(ttk.Frame):
    def __init__(self, parent, app, profile=None):
        super().__init__(parent)
        self.app = app
        self.profile = profile or NSG_PROFILE

        self.state = "instructions"
        self.selected_path = None
        self.result = None
        self.error_message = None
        self._pending_request_id = None
        self._photo_refs = []  # must outlive display or Tk shows a blank image
        self._progress = None
        self._progress_label = None
        self.mode_var = tk.StringVar(value="accurate")

        self._build_chrome()
        self.content = ttk.Frame(self)
        self.content.pack(fill="both", expand=True)
        self._render()

    @property
    def engine(self):
        return getattr(self.app, self.profile["engine_attr"])

    def _build_chrome(self):
        """Header bar that stays put across every state — going back must always
        work, including mid-analysis, since the worker thread isn't cancellable."""
        bar = ttk.Frame(self, padding=(16, 12))
        bar.pack(fill="x")
        ttk.Button(bar, text="← Back", command=self.app.show_video_menu).pack(side="left")
        ttk.Label(bar, text=self.profile["header"], style="Header.TLabel").pack(
            side="left", padx=20
        )
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
            frame,
            text=self.profile["instructions"],
            wraplength=820,
            justify="left",
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(10, 16))

        if self.profile["limitations"]:
            make_warning_banner(frame, self.profile["limitations"]).pack(fill="x", pady=(0, 20))

        engine = self.engine
        if engine.load_error:
            ttk.Label(
                frame,
                text=f"Detection models failed to load: {engine.load_error}",
                foreground="#b02a2a",
                wraplength=820,
                justify="left",
                font=("Segoe UI", 9, "bold"),
            ).pack(anchor="w", pady=(0, 12))
        elif not engine.ready:
            ttk.Label(
                frame,
                text=self.profile["loading_note"],
                foreground="#7a5b00",
                wraplength=820,
                justify="left",
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

    # ---- selected ----------------------------------------------------------

    def _probe_video(self, path):
        """Metadata + a middle-frame thumbnail via cv2.VideoCapture (same approach
        as speech_view.py: stock Tk has no video widget, and a still is enough to
        confirm the right file was picked)."""
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

        thumb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) if ok else None
        duration = frame_count / fps if fps > 0 else None
        return {"frame_count": frame_count, "fps": fps, "width": width,
                "height": height, "duration": duration, "thumb": thumb}

    def _render_selected(self):
        # A plain (non-scrolling) frame here would clip the Analyze button off
        # the bottom of the window for any portrait video: thumbnail (up to
        # 380 wide, so up to ~675 tall for a 9:16 phone clip) + title + the
        # 4-row table + the >FRAME_WINDOW warning easily exceeds the window
        # height, and unlike _render_results there is no other content above
        # it to shrink. ScrollableFrame is the same fix _render_results uses.
        scroll = ScrollableFrame(self.content)
        scroll.pack(fill="both", expand=True)
        frame = ttk.Frame(scroll.body, padding=30)
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

        min_frames = self.profile["min_frames"]
        blocked = 0 < meta["frame_count"] < min_frames
        if blocked:
            make_warning_banner(
                frame,
                f"Only {meta['frame_count']} frames. The detector needs at least "
                f"{min_frames}, so this clip cannot be analyzed.",
            ).pack(fill="x", pady=(0, 12))
        elif self.profile["key"] == "aigvdet":
            blocked = self._aigvdet_selection_notes(frame, meta)
        elif meta["frame_count"] > FRAME_WINDOW and meta["fps"]:
            analysed = FRAME_WINDOW / meta["fps"]
            make_warning_banner(
                frame,
                f"Only the first {FRAME_WINDOW} frames (~{analysed:.1f}s of "
                f"{meta['duration']:.1f}s) will be analyzed — that is the window the "
                "detector was calibrated on. Trim the clip if a later moment matters.",
            ).pack(fill="x", pady=(0, 12))

        btn_row = ttk.Frame(frame)
        btn_row.pack(pady=10)
        ttk.Button(
            btn_row, text="Choose Different Video…", command=self._choose_file
        ).pack(side="left", padx=6)
        analyze = ttk.Button(
            btn_row, text="Analyze", style="Primary.TButton", command=self._start_analysis
        )
        analyze.pack(side="left", padx=6)
        if blocked:
            analyze.state(["disabled"])

    def _aigvdet_selection_notes(self, frame, meta):
        """AIGVDet-only parts of the 'selected' screen: size banners and the mode choice.

        Returns True when the clip cannot be analyzed (frames smaller than RAFT allows).
        """
        short_side = min(meta["width"], meta["height"]) if meta["width"] else 0
        blocked = 0 < short_side < AIGVDET_RAFT_MIN_SIDE
        if blocked:
            make_warning_banner(
                frame,
                f"The frames are only {meta['width']} × {meta['height']} px; optical flow needs "
                f"at least {AIGVDET_RAFT_MIN_SIDE} px on each side, so this clip is unsupported.",
            ).pack(fill="x", pady=(0, 12))
        elif 0 < short_side < AIGVDET_CROP:
            make_warning_banner(
                frame,
                f"Small clip: the short side is {short_side} px, below the {AIGVDET_CROP} px "
                "crop the networks were trained on, so they will see zero-padded crops. Real "
                "clips this small were wrongly flagged more often in testing — treat a "
                "'flagged' result with extra caution.",
            ).pack(fill="x", pady=(0, 12))

        box = ttk.LabelFrame(frame, text="  Analysis mode  ", padding=10)
        box.pack(fill="x", pady=(0, 12))
        for value, label in AIGVDET_MODES:
            ttk.Radiobutton(box, text=label, value=value, variable=self.mode_var).pack(anchor="w")
        return blocked

    def _start_analysis(self):
        self.state = "running"
        self._render()
        if self.profile["key"] == "aigvdet":
            self._pending_request_id = self.engine.analyze_async(
                self.selected_path, mode=self.mode_var.get()
            )
        else:
            self._pending_request_id = self.engine.analyze_async(self.selected_path)

    # ---- running -----------------------------------------------------------

    def _render_running(self):
        frame = ttk.Frame(self.content, padding=30)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Analyzing…", font=("Segoe UI", 14, "bold")).pack(pady=(80, 10))
        ttk.Label(
            frame,
            text=self.profile["running_text"],
            wraplength=620,
            justify="center",
            font=("Segoe UI", 9),
        ).pack(pady=(0, 20))
        progress = ttk.Progressbar(frame, mode="indeterminate", length=320)
        progress.pack()
        progress.start(12)
        self._progress = progress  # keep alive; stops implicitly when destroyed
        self._progress_label = None
        if self.profile["key"] == "aigvdet":
            # Becomes a determinate bar once the first analysis_progress event arrives.
            self._progress_label = ttk.Label(frame, text="Starting…", font=("Segoe UI", 9))
            self._progress_label.pack(pady=(8, 0))

    def _update_progress(self, done, total):
        """Switch the bar to determinate and show 'pair done of total'."""
        progress, label = self._progress, self._progress_label
        if progress is None or label is None or not progress.winfo_exists():
            return
        progress.stop()
        progress.configure(mode="determinate", maximum=total, value=done)
        label.configure(text=f"Frame pair {done} of {total}")

    # ---- results -----------------------------------------------------------

    def _render_results(self):
        if self.profile["key"] == "aigvdet":
            self._render_results_aigvdet()
        else:
            self._render_results_nsg()

    def _render_results_aigvdet(self):
        result = self.result
        scroll = ScrollableFrame(self.content)
        scroll.pack(fill="both", expand=True)
        frame = ttk.Frame(scroll.body, padding=24)
        frame.pack(fill="both", expand=True)

        level = {"flagged": "generated", "uncertain": "suspicious", "clear": "real"}.get(
            result["verdict_level"], "suspicious"
        )
        make_verdict_banner(frame, result["verdict"], level=level).pack(fill="x", pady=(0, 16))

        fig_path = result.get("figure")
        if fig_path and Path(fig_path).exists():
            ttk.Label(frame, text="AIGVDet evidence", font=("Segoe UI", 11, "bold")).pack(
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
        ttk.Button(btn_row, text="← Back", command=self.app.show_video_menu).pack(
            side="left", padx=6
        )

    def _render_results_nsg(self):
        result = self.result
        scroll = ScrollableFrame(self.content)
        scroll.pack(fill="both", expand=True)
        frame = ttk.Frame(scroll.body, padding=24)
        frame.pack(fill="both", expand=True)

        make_verdict_banner(
            frame, result["verdict"], level=result.get("verdict_level")
        ).pack(fill="x", pady=(0, 16))

        # Unlike the image module (which shows no numbers because Delta_Min is
        # uncalibrated), p_fake here IS a calibrated probability: the threshold was
        # set for 5% false positives on validation reals and held at 4.8% on unseen
        # test reals, so showing both the value and the threshold is honest and lets
        # a reader see how marginal a call was.
        rows = [
            ("Probability of AI generation", f"{result['p_fake'] * 100:.1f}%"),
            ("Decision threshold", f"{result['threshold'] * 100:.1f}% (5% false-positive calibration)"),
            ("Frames scored", f"8 of the first {result.get('frame_window', FRAME_WINDOW)}"),
        ]
        if result.get("analysed_seconds") is not None:
            span = f"{result['analysed_seconds']:.1f} s"
            if result.get("duration_seconds"):
                span += f" of {result['duration_seconds']:.1f} s"
            rows.append(("Portion analyzed", span))
        rows.append(("Source", f"{result.get('resolution', '?')} · {result.get('codec', '?')} · "
                               f"{result.get('fps', 0):.1f} fps"))
        rows.append(("Analysis time", f"{result['seconds']:.0f} s"))
        make_kv_table(frame, rows, title="Measurements").pack(anchor="w", pady=(0, 16))

        if result.get("verdict_level") == "suspicious":
            make_warning_banner(
                frame,
                f"This probability ({result['p_fake'] * 100:.1f}%) sits within 10 "
                f"points of the {result['threshold'] * 100:.1f}% threshold, so the "
                "call could go either way. Treat it as inconclusive rather than as "
                "a weak 'real' or a weak 'fake'.",
            ).pack(fill="x", pady=(0, 12))

        fig_path = result.get("figure")
        if fig_path and Path(fig_path).exists():
            ttk.Label(frame, text="NSG evidence", font=("Segoe UI", 11, "bold")).pack(
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

    # ---- error -------------------------------------------------------------

    def _render_error(self):
        frame = ttk.Frame(self.content, padding=30)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Something went wrong", font=("Segoe UI", 14, "bold")).pack(
            pady=(50, 10)
        )
        ttk.Label(
            frame,
            text=self.error_message or "Unknown error.",
            wraplength=820,
            justify="left",
        ).pack(pady=(0, 20))
        ttk.Button(frame, text="Try Again", command=self._reset).pack()

    # ---- engine events -----------------------------------------------------

    def on_engine_event(self, event, payload, request_id=None):
        if event == "analysis_progress":
            if request_id == self._pending_request_id and self.state == "running":
                self._update_progress(*payload)
            return
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
        elif event in ("models_ready", "models_failed") and self.state == "instructions":
            self._render()  # refresh the "still loading" / failure notice
