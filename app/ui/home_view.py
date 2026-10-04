"""Home screen: entry point for choosing a forensics module."""

import tkinter as tk
from tkinter import ttk


class HomeView(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=40)
        self.app = app

        ttk.Label(self, text="Forensics Toolkit", style="Header.TLabel").pack(pady=(30, 4))
        ttk.Label(
            self,
            text="Choose a module to analyze media for signs of AI generation.",
            style="Sub.TLabel",
        ).pack(pady=(0, 40))

        button_area = ttk.Frame(self)
        button_area.pack(pady=10)

        ttk.Button(
            button_area,
            text="🖼   Image Forensics",
            style="Primary.TButton",
            width=34,
            command=self.app.show_image_forensics,
        ).pack(pady=8)

        ttk.Button(
            button_area,
            text="🎙   Speech Forensics",
            style="Primary.TButton",
            width=34,
            command=self.app.show_speech_forensics,
        ).pack(pady=8)

        ttk.Button(
            button_area,
            text="🎬   Video Forensics",
            style="Primary.TButton",
            width=34,
            command=self.app.show_video_menu,
        ).pack(pady=8)

        self.status_var = tk.StringVar(value=self._status_text())
        ttk.Label(self, textvariable=self.status_var, style="Status.TLabel").pack(
            side="bottom", pady=(30, 0)
        )

    def _status_text(self):
        """Composed from all three engines' .ready flags rather than the triggering
        event's payload, so any engine's models_ready/models_failed — whichever
        fires — just triggers a recompute of the combined line; no per-event source
        tracking needed (see app_window.py's _poll_events for the matching reasoning
        on why cross-engine events never need to be tagged)."""
        image_text = (
            "Image models ready" if self.app.engine.ready else "loading image models…"
        )
        speech_text = (
            "speech models ready"
            if self.app.speech_engine.ready
            else "loading speech models…"
        )
        video_engine = self.app.video_engine
        if video_engine.load_error:
            video_text = "video models unavailable"
        elif video_engine.ready:
            video_text = "video models ready"
        else:
            video_text = "loading video models…"
        aigvdet_engine = self.app.aigvdet_engine
        if aigvdet_engine.load_error:
            aigvdet_text = (
                "AIGVDet models missing (run pull_export.ps1)"
                if aigvdet_engine.missing_models
                else "AIGVDet models unavailable"
            )
        elif aigvdet_engine.ready:
            aigvdet_text = "AIGVDet models ready"
        else:
            aigvdet_text = "loading AIGVDet models…"
        return (
            f"{image_text} (first run may download several GB of weights); "
            f"{speech_text}; {video_text}; {aigvdet_text}."
        )

    def on_engine_event(self, event, payload, request_id=None):
        if event in ("models_ready", "models_failed"):
            self.status_var.set(self._status_text())
