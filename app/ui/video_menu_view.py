"""
Video Forensics chooser: two separate detectors, opened from the Home screen.

Each card opens its own screen (the same state machine in video_view.py, configured by a
detector profile), so the two methods are never mixed in one result.
"""

import tkinter as tk
from tkinter import ttk


class VideoMenuView(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=40)
        self.app = app

        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Button(bar, text="← Back to Home", command=self.app.show_home).pack(side="left")

        ttk.Label(self, text="Video Forensics", style="Header.TLabel").pack(pady=(10, 4))
        ttk.Label(
            self,
            text="Two independent detectors — pick one. They use different evidence, so "
            "their verdicts on the same clip can differ.",
            style="Sub.TLabel",
            wraplength=760,
            justify="center",
        ).pack(pady=(0, 24))

        cards = ttk.Frame(self)
        cards.pack(fill="x")
        cards.columnconfigure((0, 1), weight=1, uniform="card")

        self.nsg_status = tk.StringVar()
        self.aigvdet_status = tk.StringVar()
        self._card(
            cards, 0, "NSG-based", "diffusion physics",
            self.nsg_status, self.app.show_video_forensics,
        )  # fmt: skip
        self._card(
            cards, 1, "AIGVDet-based", "RGB frames + optical flow",
            self.aigvdet_status, self.app.show_aigvdet,
        )  # fmt: skip
        self._refresh_status()

    def _card(self, parent, column, title, subtitle, status_var, command):
        card = ttk.LabelFrame(parent, text=f"  {title}  ", padding=18)
        card.grid(row=0, column=column, sticky="nsew", padx=10, pady=6)
        ttk.Label(card, text=subtitle, font=("Segoe UI", 11, "bold")).pack(
            anchor="w", pady=(0, 8)
        )
        ttk.Label(
            card, textvariable=status_var, wraplength=330, justify="left", style="Status.TLabel"
        ).pack(anchor="w", pady=(0, 12))
        ttk.Button(card, text=f"Open {title}", style="Primary.TButton", command=command).pack(
            anchor="w"
        )

    def _refresh_status(self):
        nsg = self.app.video_engine
        if nsg.load_error:
            self.nsg_status.set("Models unavailable — see the NSG screen for details.")
        elif nsg.ready:
            self.nsg_status.set("Models ready.")
        else:
            self.nsg_status.set("Loading models… (about 2 GB of weights)")

        engine = self.app.aigvdet_engine
        if engine.load_error and engine.missing_models:
            self.aigvdet_status.set("Models not found — run AIGVDET\\tools\\pull_export.ps1.")
        elif engine.load_error:
            self.aigvdet_status.set("Models unavailable — see the AIGVDet screen for details.")
        elif engine.ready:
            self.aigvdet_status.set("Models ready.")
        else:
            self.aigvdet_status.set("Loading models…")

    def on_engine_event(self, event, payload, request_id=None):
        if event in ("models_ready", "models_failed"):
            self._refresh_status()
