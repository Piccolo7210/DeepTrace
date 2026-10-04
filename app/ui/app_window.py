"""Application shell: Tk root window, shared ttk style, and view routing."""

import queue
import tkinter as tk
from tkinter import ttk

from aigvdet_engine import AIGVDetEngine
from engine import ForensicsEngine
from speech_engine import SpeechForensicsEngine
from video_engine import VideoForensicsEngine
from ui.home_view import HomeView
from ui.image_view import ImageView
from ui.placeholder_view import PlaceholderView
from ui.speech_view import SpeechView
from ui.video_menu_view import VideoMenuView
from ui.video_view import AIGVDET_PROFILE, VideoView

APP_TITLE = "Forensics Toolkit"


class AppWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1000x720")
        self.minsize(900, 640)

        self._configure_style()

        self.engine = ForensicsEngine()
        self.speech_engine = SpeechForensicsEngine()
        self.video_engine = VideoForensicsEngine()
        self.aigvdet_engine = AIGVDetEngine()
        self._current_view = None

        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)

        self.show_home()
        self.engine.load_async()
        self.speech_engine.load_async()
        self.video_engine.load_async()
        self.aigvdet_engine.load_async()
        self.after(100, self._poll_events)

    def _configure_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", font=("Segoe UI", 10), padding=8)
        style.configure("Primary.TButton", font=("Segoe UI", 12, "bold"), padding=12)
        style.configure("Muted.TButton", font=("Segoe UI", 12), padding=12)
        style.configure("Header.TLabel", font=("Segoe UI", 22, "bold"))
        style.configure("Sub.TLabel", font=("Segoe UI", 10))
        style.configure("Status.TLabel", font=("Segoe UI", 9))

    # ---- view routing ----------------------------------------------------

    def _swap(self, view_cls, **kwargs):
        if self._current_view is not None:
            self._current_view.destroy()
        self._current_view = view_cls(self.container, self, **kwargs)
        self._current_view.pack(fill="both", expand=True)

    def show_home(self):
        self._swap(HomeView)

    def show_image_forensics(self):
        self._swap(ImageView)

    def show_speech_forensics(self):
        self._swap(SpeechView)

    def show_video_menu(self):
        """Chooser between the two video detectors (NSG-based / AIGVDet-based)."""
        self._swap(VideoMenuView)

    def show_video_forensics(self):
        """The NSG-based detector (VideoView's default profile)."""
        self._swap(VideoView)

    def show_aigvdet(self):
        """The AIGVDet-based detector (same view class, AIGVDet profile)."""
        self._swap(VideoView, profile=AIGVDET_PROFILE)

    def show_placeholder(self, module_name):
        self._swap(PlaceholderView, module_name=module_name)

    # ---- engine event pump -------------------------------------------------

    def _poll_events(self):
        # Drains all four engines every tick. No "which engine" tag is threaded through
        # to on_engine_event: request_id = object() is identity-unique per call (see
        # engine.py / speech_engine.py), so a stray event from the *other* engine can
        # never spuriously match a view's _pending_request_id — the existing
        # request-id guard already gives perfect cross-engine isolation for free.
        for engine in (self.engine, self.speech_engine, self.video_engine, self.aigvdet_engine):
            try:
                while True:
                    event, payload, request_id = engine.events.get_nowait()
                    handler = getattr(self._current_view, "on_engine_event", None)
                    if handler is not None:
                        handler(event, payload, request_id)
            except queue.Empty:
                pass
        self.after(100, self._poll_events)
