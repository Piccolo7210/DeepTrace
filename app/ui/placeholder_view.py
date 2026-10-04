"""Reusable 'coming soon' screen for modules not yet implemented (Speech, Video)."""

from tkinter import ttk


class PlaceholderView(ttk.Frame):
    def __init__(self, parent, app, module_name="This module"):
        super().__init__(parent, padding=40)
        self.app = app

        ttk.Label(self, text=module_name, style="Header.TLabel").pack(pady=(80, 10))
        ttk.Label(
            self,
            text="Not implemented yet — this is a placeholder for a future module.",
            style="Sub.TLabel",
        ).pack(pady=(0, 40))
        ttk.Button(self, text="← Back to Home", command=self.app.show_home).pack()
