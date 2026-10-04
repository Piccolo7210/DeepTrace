"""Small reusable Tk/ttk widgets shared by the app's views."""

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

COLOR_REAL = "#1e7e34"
COLOR_GENERATED = "#b02a2a"
COLOR_SUSPICIOUS = "#b8860b"
COLOR_WARNING_BG = "#fff3cd"
COLOR_WARNING_FG = "#7a5b00"


class ScrollableFrame(ttk.Frame):
    """
    A vertically scrollable frame (Canvas + Scrollbar + inner content Frame). Use
    `.body` as the parent for content — the results view (figure + tables) routinely
    grows taller than the window.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.body = ttk.Frame(canvas)

        self.body.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        window_id = canvas.create_window((0, 0), window=self.body, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        def _on_wheel(event):
            if canvas.winfo_exists():
                canvas.yview_scroll(int(-event.delta / 120), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))


def make_verdict_banner(parent, verdict, is_flagged=None, level=None):
    """
    Colour-coded verdict strip. No confidence percentage or score breakdown is
    shown — Delta_Min vs. threshold is a decision rule, not a calibrated
    probability, so surfacing the raw numbers here would invite over-reading them.

    is_flagged picks red vs. green explicitly. Defaults to substring-matching
    "AI-generated" in verdict (the image module's convention) when omitted, so that
    call site needs no changes; the speech module's verdict strings ("Anomalous
    (audio-visual desync)" / "Consistent with real") don't contain that substring,
    so it must pass is_flagged explicitly or every result would render green.

    level overrides both with a named band ("real"/"suspicious"/"anomalous"). The
    middle band exists because a single threshold on a weakly-calibrated anomaly
    score can't honestly support a binary call — a localized finding is reported as
    its own amber level rather than being rounded to either "clean" or "fake".
    """
    if level is not None:
        color = {"real": COLOR_REAL, "suspicious": COLOR_SUSPICIOUS}.get(level, COLOR_GENERATED)
        display_text = verdict
    else:
        is_generated = "AI-generated" in verdict if is_flagged is None else is_flagged
        color = COLOR_GENERATED if is_generated else COLOR_REAL
        display_text = "AI Generated" if is_generated and "AI-generated" in verdict else verdict
    frame = tk.Frame(parent, bg=color)

    tk.Label(
        frame,
        text=display_text,
        bg=color,
        fg="white",
        font=("Segoe UI", 20, "bold"),
        pady=14,
    ).pack(fill="x")

    return frame


def make_warning_banner(parent, text):
    frame = tk.Frame(parent, bg=COLOR_WARNING_BG)
    tk.Label(
        frame,
        text=f"⚠ {text}",
        bg=COLOR_WARNING_BG,
        fg=COLOR_WARNING_FG,
        font=("Segoe UI", 10, "bold"),
        wraplength=820,
        justify="left",
        padx=12,
        pady=8,
    ).pack(fill="x")
    return frame


def make_kv_table(parent, rows, title=None):
    """rows: list of (label, value) string pairs, rendered as a two-column grid."""
    frame = ttk.Frame(parent)
    start_row = 0
    if title:
        ttk.Label(frame, text=title, font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        start_row = 1
    for i, (key, value) in enumerate(rows):
        ttk.Label(frame, text=key, font=("Segoe UI", 9)).grid(
            row=start_row + i, column=0, sticky="w", padx=(0, 16), pady=2
        )
        ttk.Label(frame, text=value, font=("Segoe UI", 9, "bold")).grid(
            row=start_row + i, column=1, sticky="w", pady=2
        )
    return frame


def scale_photo_image(img, max_width=900):
    """
    Scales an already-loaded PIL Image to max_width (preserving aspect ratio; images
    already narrower are left alone) and returns an ImageTk.PhotoImage. Factored out
    of load_scaled_photo so callers with an in-memory frame (e.g. speech_view.py's
    video-thumbnail grab via cv2.VideoCapture, no file on disk) can share the same
    resize logic instead of duplicating it.

    The caller MUST keep a reference to the returned object for as long as it's
    displayed — Tk does not hold one itself, and a garbage-collected PhotoImage
    renders as a blank widget with no error.
    """
    w, h = img.size
    if w > max_width:
        scale = max_width / w
        img = img.resize((max_width, round(h * scale)), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


def load_scaled_photo(path, max_width=900):
    """Loads an image file and returns it scaled via scale_photo_image(). Same
    caller-must-keep-a-reference caveat applies."""
    return scale_photo_image(Image.open(path), max_width)
