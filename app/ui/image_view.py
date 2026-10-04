"""
Image Forensics screen: instructions -> pick a file -> analyze -> results.

State machine (self.state), each with its own _render_<state>():
  instructions -> selected -> running -> results
                                       -> error  (bad file, or analysis raised)
"""

from pathlib import Path
from tkinter import filedialog, ttk

from PIL import Image

from ui.widgets import (
    ScrollableFrame,
    load_scaled_photo,
    make_verdict_banner,
    make_warning_banner,
)

SUPPORTED_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")

INSTRUCTIONS_TEXT = (
    "• Shorter side of at least 512 px. Larger images are fine — they are "
    "scaled down and center-cropped to 512×512.\n"
    "• Non-square images are center-cropped, so keep the subject near the middle.\n"
)


class ImageView(ttk.Frame):
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
        """Header bar that stays put across every state, so 'back to home' always
        works — including mid-analysis, since the worker thread isn't cancellable
        and simply keeps running; a stray result is discarded via request-id
        matching in on_engine_event if the user has since started another one."""
        bar = ttk.Frame(self, padding=(16, 12))
        bar.pack(fill="x")
        ttk.Button(bar, text="← Back to Home", command=self.app.show_home).pack(side="left")
        ttk.Label(bar, text="Image Forensics", style="Header.TLabel").pack(side="left", padx=20)
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

        ttk.Label(frame, text="Image requirements", font=("Segoe UI", 13, "bold")).pack(
            anchor="w"
        )
        ttk.Label(
            frame, text=INSTRUCTIONS_TEXT, wraplength=820, justify="left", font=("Segoe UI", 10)
        ).pack(anchor="w", pady=(10, 24))

        if not self.app.engine.ready:
            ttk.Label(
                frame,
                text="Detection models are still loading — you can choose an image "
                "now; analysis will start once they're ready.",
                foreground="#7a5b00",
                font=("Segoe UI", 9, "italic"),
            ).pack(anchor="w", pady=(0, 12))

        ttk.Button(
            frame, text="Choose Image…", style="Primary.TButton", command=self._choose_file
        ).pack(anchor="w")

    def _choose_file(self):
        filetypes = [("Images", " ".join(SUPPORTED_EXTS)), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Choose an image", filetypes=filetypes)
        if not path:
            return
        self.selected_path = path
        self.state = "selected"
        self._render()

    # ---- selected ------------------------------------------------------------

    def _render_selected(self):
        frame = ttk.Frame(self.content, padding=30)
        frame.pack(fill="both", expand=True)

        try:
            with Image.open(self.selected_path) as probe:
                w, h = probe.size
        except Exception as exc:
            self.error_message = f"Could not open this image:\n{exc}"
            self.state = "error"
            self._render()
            return

        photo = load_scaled_photo(self.selected_path, max_width=380)
        self._photo_refs.append(photo)
        ttk.Label(frame, image=photo).pack(pady=(0, 12))

        ttk.Label(
            frame, text=Path(self.selected_path).name, font=("Segoe UI", 10, "bold")
        ).pack()
        ttk.Label(frame, text=f"{w} × {h} px", font=("Segoe UI", 9)).pack(pady=(0, 12))

        if min(w, h) < 512:
            make_warning_banner(
                frame,
                f"Shorter side is {min(w, h)} px, below the 512 px recommendation. "
                "The result will be flagged low-confidence.",
            ).pack(fill="x", pady=(0, 12))

        btn_row = ttk.Frame(frame)
        btn_row.pack(pady=10)
        ttk.Button(
            btn_row, text="Choose Different Image…", command=self._choose_file
        ).pack(side="left", padx=6)
        ttk.Button(
            btn_row, text="Analyze", style="Primary.TButton", command=self._start_analysis
        ).pack(side="left", padx=6)

    def _start_analysis(self):
        self.state = "running"
        self._render()
        self._pending_request_id = self.app.engine.analyze_async(self.selected_path)

    # ---- running ---------------------------------------------------------------

    def _render_running(self):
        frame = ttk.Frame(self.content, padding=30)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Analyzing…", font=("Segoe UI", 14, "bold")).pack(
            pady=(80, 10)
        )
        ttk.Label(
            frame,
            text="Roughly 150-180 s on CPU — three autoencoder round-trips plus LPIPS.",
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

        make_verdict_banner(frame, result["verdict"]).pack(fill="x", pady=(0, 16))

        if result.get("low_resolution_warning"):
            ow, oh = result.get("original_size", (None, None))
            size_text = f"{ow}×{oh}" if ow else "below the minimum"
            make_warning_banner(
                frame,
                f"Source image was {size_text}, smaller than 512×512. Upscaling "
                "blur biases the score toward “AI-generated” — treat this "
                "verdict as low-confidence.",
            ).pack(fill="x", pady=(0, 12))

        if not result.get("threshold_calibrated", True):
            make_warning_banner(
                frame,
                "No calibrated threshold was found on disk; this verdict used the "
                f"uncalibrated default ({result['threshold']:.4f}) rather than a fitted "
                "one. Results may be less reliable.",
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
