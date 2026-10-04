"""Tests for the AIGVDet package. They need no trained weights.

Run from the AIGVDET folder::

    python tests/test_aigvdet_module.py        # or: python -m pytest tests
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image as torchvision_flow_to_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aigvdet.detector import ExportError, load_detector  # noqa: E402
from aigvdet.flow import clamp_scale, load_raft, pad_to_multiple_of_8, resize_flow  # noqa: E402
from aigvdet.flow_viz import flow_to_image  # noqa: E402
from aigvdet.fuse import fuse, verdict  # noqa: E402
from aigvdet.models import build_branch, load_branch  # noqa: E402
from aigvdet.preprocess import center_crop_or_pad  # noqa: E402
from aigvdet.scoring import PairScorer  # noqa: E402
from aigvdet.verification import (  # noqa: E402
    sha256_of,
    verify_checksums,
    write_checksums,
)
from aigvdet.video_io import (  # noqa: E402
    evenly_spaced,
    pick_pair_indices,
    read_selected_frames,
    resolution_bucket,
)

FRAME_SIZE = (160, 128)  # (width, height): RAFT needs at least 128 px per side
UPDATES = 2  # keeps the CPU tests fast


def write_clip(path: Path, frames: int = 12, size: tuple[int, int] = FRAME_SIZE) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 16.0, size)
    base = np.random.default_rng(0).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    for step in range(frames):
        writer.write(np.roll(base, 3 * step, axis=1))
    writer.release()


def make_bundle(directory: Path) -> None:
    """A complete export directory with random weights (what pull_export.ps1 produces)."""
    torch.manual_seed(0)
    for branch in ("rgb", "flow"):
        torch.save(build_branch(pretrained=False).state_dict(), directory / f"aigvdet_{branch}.pth")
    torch.save(raft_large(weights=None).state_dict(), directory / "raft_large_C_T_V1.pth")
    meta = {
        "fusion_alpha": 0.5,
        "flow": {"raft_updates": UPDATES, "max_flow_pixels": 1_500_000},
        "threshold": {"value": 0.45, "source": "calibrated", "uncertain_band": 0.1},
        "calibration": {"fpr_on_report": 0.08},
        "known_limits": ["test bundle"],
    }
    (directory / "aigvdet_meta.json").write_text(json.dumps(meta))
    clip = directory.parent / "bundle_clip.mp4"
    write_clip(clip)
    frames = read_selected_frames(clip, {0, 1, 5, 6})
    scorer = PairScorer(
        load_branch(directory / "aigvdet_rgb.pth"),
        load_branch(directory / "aigvdet_flow.pth"),
        load_raft("cpu", weights_path=directory / "raft_large_C_T_V1.pth"),
        "cpu",
        updates=UPDATES,
        conditioned=False,
    )
    rows = [scorer.score_pair(frames[i], frames[i + 1]) for i in (0, 5)]
    np.savez_compressed(
        directory / "parity_samples.npz",
        s0_key=np.array("test/clip"),
        s0_pair_idx=np.array([0, 5]),
        s0_frames=np.stack([np.stack([frames[0], frames[1]]), np.stack([frames[5], frames[6]])]),
        s0_p_rgb=np.array([r["p_rgb_raw"] for r in rows]),
        s0_p_flow=np.array([r["p_flow_raw"] for r in rows]),
    )
    write_checksums(directory)


# --------------------------------------------------------------------------- #
# Pure functions
# --------------------------------------------------------------------------- #
def test_fusion_and_verdict_bands() -> None:
    assert fuse(0.2, 0.6) == 0.4 and fuse(1.0, 0.0, alpha=0.25) == 0.25
    assert verdict(0.70, 0.45, 0.10) == ("AI-generated", "flagged", True)
    assert verdict(0.50, 0.45, 0.10) == ("Uncertain", "uncertain", True)
    assert verdict(0.40, 0.45, 0.10) == ("Uncertain", "uncertain", False)
    assert verdict(0.20, 0.45, 0.10) == ("Consistent with real", "clear", False)


def test_pair_selection_and_buckets() -> None:
    assert pick_pair_indices(95, 32)[0] == 0 and pick_pair_indices(95, 32)[-1] == 93
    assert pick_pair_indices(8, 32) == list(range(7))  # short clips use every pair
    assert pick_pair_indices(1, 5) == [] and evenly_spaced(93, 16)[-1] == 92
    sizes = [(320, 568), (640, 1088), (1280, 720)]
    assert [resolution_bucket(*s) for s in sizes] == ["<448", "448-719", ">=720"]


def test_centre_crop_matches_torchvision() -> None:
    rng = np.random.default_rng(0)
    for height, width in [(636, 1184), (637, 1001), (300, 500), (447, 449)]:
        image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        expected = np.asarray(transforms.CenterCrop(448)(Image.fromarray(image)))
        assert np.array_equal(center_crop_or_pad(image), expected), (height, width)


def test_flow_colour_map_matches_torchvision() -> None:
    flow = np.random.default_rng(0).normal(0, 5, (60, 80, 2)).astype(np.float32)
    reference = torchvision_flow_to_image(torch.from_numpy(flow).permute(2, 0, 1))
    difference = np.abs(flow_to_image(flow).astype(int) - reference.permute(1, 2, 0).numpy())
    assert difference.max() <= 1
    assert (flow_to_image(np.zeros((8, 8, 2), np.float32)) == 255).all()  # no motion = white


def test_padding_matches_raft_input_padder() -> None:
    for height, width in [(636, 1184), (333, 501), (8, 8)]:
        pad_h = (((height // 8) + 1) * 8 - height) % 8
        pad_w = (((width // 8) + 1) * 8 - width) % 8
        padding = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
        batch = torch.rand(1, 3, height, width)
        expected = functional.pad(batch, padding, mode="replicate")
        assert torch.equal(pad_to_multiple_of_8(batch), expected)


def test_small_frames_keep_raft_minimum_size() -> None:
    assert clamp_scale(0.5, 128, 160) == 1.0  # already at the minimum: no shrinking
    assert abs(clamp_scale(0.5, 240, 320) - 128 / 240) < 1e-12  # raised just enough
    assert clamp_scale(0.75, 240, 320) == 0.75 and clamp_scale(0.5, 1080, 1920) == 0.5
    assert clamp_scale(0.5, 100, 400) == 1.0  # below the minimum natively: left to raise


def test_resize_flow_rescales_vectors() -> None:
    field = torch.zeros(1, 2, 40, 60)
    field[:, 0], field[:, 1] = 2.0, 1.0
    bigger = resize_flow(field, (80, 180))  # width x3, height x2
    assert torch.allclose(bigger[:, 0], torch.full((80, 180), 6.0))
    assert torch.allclose(bigger[:, 1], torch.full((80, 180), 2.0))


# --------------------------------------------------------------------------- #
# Files and video
# --------------------------------------------------------------------------- #
def test_checksums_detect_corruption() -> None:
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder)
        (directory / "a.bin").write_bytes(b"alpha" * 1000)
        (directory / "sub").mkdir()
        (directory / "sub" / "b.bin").write_bytes(b"beta" * 1000)
        write_checksums(directory)
        assert verify_checksums(directory) == []
        (directory / "a.bin").write_bytes(b"alpha" * 999)  # truncated download
        (directory / "sub" / "b.bin").unlink()
        problems = verify_checksums(directory)
        assert any("a.bin" in p and "mismatch" in p for p in problems)
        assert any("b.bin" in p and "missing" in p for p in problems)
        assert sha256_of(directory / "a.bin") != ""


def test_read_selected_frames() -> None:
    with tempfile.TemporaryDirectory() as folder:
        clip = Path(folder) / "clip.mp4"
        write_clip(clip, frames=10)
        frames = read_selected_frames(clip, {0, 4, 9, 40})  # 40 does not exist
        assert sorted(frames) == [0, 4, 9] and frames[4].shape == (FRAME_SIZE[1], FRAME_SIZE[0], 3)
        assert read_selected_frames(clip, set()) == {}


# --------------------------------------------------------------------------- #
# Detector (random weights: checks plumbing, not accuracy)
# --------------------------------------------------------------------------- #
def test_detector_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        bundle = root / "models"
        bundle.mkdir()
        make_bundle(bundle)
        detector = load_detector(bundle)

        parity = detector.parity()
        assert float(parity[["d_rgb", "d_flow", "d_fused"]].to_numpy().max()) < 1e-6

        clip = root / "clip.mp4"
        write_clip(clip, frames=12)
        seen = []
        result = detector.analyze(
            clip, mode="quick", pairs=3, updates=UPDATES, progress=lambda d, t: seen.append((d, t))
        )
        assert seen == [(1, 3), (2, 3), (3, 3)] and result["pairs_used"] == 3
        assert 0.0 <= result["p_fake"] <= 1.0 and result["verdict"] in {
            "AI-generated", "Uncertain", "Consistent with real"}  # fmt: skip
        assert abs(result["p_fake"] - np.mean(result["frame_probs"]["fused"])) < 1e-12
        assert result["resolution"] == list(FRAME_SIZE) and result["threshold"] == 0.45
        text = " ".join(result["caveats"])
        assert "Small clip" in text and "Only 3 frame pairs" in text and "whole clip" in text
        assert "flagged 8%" in text

        assert "crops" not in result  # opt-in only
        kept = detector.analyze(clip, mode="quick", pairs=4, updates=UPDATES, keep_crops=True)
        extremes = kept["crops"]
        assert set(extremes) == {"most_suspicious", "least_suspicious"}
        assert extremes["most_suspicious"]["score"] >= extremes["least_suspicious"]["score"]
        assert extremes["most_suspicious"]["rgb"].shape == (448, 448, 3)
        assert extremes["most_suspicious"]["flow"].dtype == np.uint8

        window = detector.analyze(
            clip, mode="benchmark", window="first95", pairs=2, updates=UPDATES
        )
        assert "whole clip" not in " ".join(window["caveats"])

        for bad in ({"mode": "turbo"}, {"window": "middle"}):
            try:
                detector.analyze(clip, **bad)
                raise AssertionError("invalid argument accepted")
            except ValueError:
                pass
        short = root / "short.mp4"
        write_clip(short, frames=3)
        try:
            detector.analyze(short, updates=UPDATES)
            raise AssertionError("3-frame clip accepted")
        except ValueError as error:
            assert "too short" in str(error)

        (bundle / "aigvdet_rgb.pth").write_bytes(b"truncated")
        try:
            load_detector(bundle)
            raise AssertionError("corrupt bundle accepted")
        except ExportError as error:
            assert "aigvdet_rgb.pth" in str(error)


def synthetic_result(with_crops: bool) -> dict:
    rng = np.random.default_rng(1)
    result = {
        "file": "synthetic.mp4", "p_fake": 0.62, "p_spatial": 0.7, "p_temporal": 0.54,
        "threshold": 0.43, "uncertain_band": 0.1, "verdict": "AI-generated",
        "verdict_level": "flagged", "pairs_used": 4, "pair_indices": [0, 10, 20, 30], "fps": 16.0,
        "frame_probs": {"spatial": [0.9, 0.7, 0.6, 0.6], "temporal": [0.3, 0.5, 0.6, 0.7],
                        "fused": [0.6, 0.6, 0.6, 0.65]},
    }  # fmt: skip
    if with_crops:
        crop = lambda: rng.integers(0, 255, (448, 448, 3), dtype=np.uint8)  # noqa: E731
        result["crops"] = {
            "most_suspicious": {"pair_idx": 30, "score": 0.65, "rgb": crop(), "flow": crop()},
            "least_suspicious": {"pair_idx": 0, "score": 0.6, "rgb": crop(), "flow": crop()},
        }
    return result


def test_report_renders_with_and_without_crops() -> None:
    from aigvdet.report import plot_report

    with tempfile.TemporaryDirectory() as folder:
        for with_crops in (True, False):
            out = plot_report(synthetic_result(with_crops), Path(folder) / f"r{with_crops}.png")
            assert out.exists() and out.stat().st_size > 10_000
        flagged_only = synthetic_result(False)
        flagged_only["fps"] = 0.0  # unknown frame rate falls back to frame indices
        assert plot_report(flagged_only, Path(folder) / "nofps.png").exists()


def test_parity_gate_fails_on_wrong_weights() -> None:
    with tempfile.TemporaryDirectory() as folder:
        bundle = Path(folder) / "models"
        bundle.mkdir()
        make_bundle(bundle)
        torch.manual_seed(99)  # different flow-branch weights than the ones the samples came from
        torch.save(build_branch(pretrained=False).state_dict(), bundle / "aigvdet_flow.pth")
        detector = load_detector(bundle, verify=False)
        worst = float(detector.parity()[["d_rgb", "d_flow", "d_fused"]].to_numpy().max())
        assert worst > 0.0


def test_cli_json_and_parity() -> None:
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        bundle = root / "models"
        bundle.mkdir()
        make_bundle(bundle)
        clip = root / "clip.mp4"
        write_clip(clip, frames=12)
        script = str(ROOT / "detect_aigvdet.py")
        parity = subprocess.run(
            [sys.executable, script, "--parity", "--models-dir", str(bundle)],
            capture_output=True, text=True,
        )  # fmt: skip
        assert parity.returncode == 0 and "PASSED" in parity.stdout, parity.stdout + parity.stderr
        run = subprocess.run(
            [sys.executable, script, str(clip), "--models-dir", str(bundle), "--pairs", "2",
             "--mode", "quick", "--json"],
            capture_output=True, text=True,
        )  # fmt: skip
        assert run.returncode == 0, run.stderr
        assert json.loads(run.stdout)["pairs_used"] == 2
        missing = subprocess.run(
            [sys.executable, script, str(clip), "--models-dir", str(root / "nowhere")],
            capture_output=True, text=True,
        )  # fmt: skip
        assert missing.returncode == 1 and "pull_export" in missing.stderr


# --------------------------------------------------------------------------- #
# App integration (fake detector: no weights needed)
# --------------------------------------------------------------------------- #
APP_DIR = ROOT.parent / "app"


def _app_imports():
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    import aigvdet_engine

    return aigvdet_engine


class FakeDetector:
    """Stands in for AIGVDetDetector: reports progress and returns a plausible result."""

    def __init__(self, error: Exception | None = None, delay: float = 0.0) -> None:
        self.error, self.delay, self.calls = error, delay, []

    def analyze(self, path, mode="accurate", window="whole", progress=None, keep_crops=False,
                **kwargs):  # fmt: skip
        import time

        self.calls.append((str(path), mode, window, keep_crops))
        if self.error:
            raise self.error
        for done in (1, 2):
            time.sleep(self.delay)
            if progress:
                progress(done, 2)
        result = synthetic_result(with_crops=keep_crops)
        result.update({"mode": mode, "window": window, "alpha": 0.5, "seconds": 3.0,
                       "resolution": [320, 240], "caveats": ["note one"], "limits": ["limit one"],
                       "calibration": {"mean_fake_recall": 0.74,
                                       "fake_recall_by_set": {"T2V sora": 0.33, "T2V emu": 1.0}},
                       "duration_seconds": 2.0, "verdict_level": "clear",
                       "verdict": "Consistent with real"})  # fmt: skip
        return result


def _drain(engine, request_id, timeout: float = 20.0) -> list[tuple]:
    """Collect this request's events up to and including its terminal event."""
    import queue

    events = []
    while True:
        event, payload, rid = engine.events.get(timeout=timeout)
        if rid is not request_id and event.startswith("analysis"):
            continue
        events.append((event, payload))
        if event in ("analysis_done", "analysis_failed"):
            return events
        if event == "models_failed":
            raise queue.Empty


def test_engine_runs_analysis_and_reports_progress() -> None:
    module = _app_imports()
    with tempfile.TemporaryDirectory() as folder:
        run_dir = Path(folder) / "run"
        run_dir.mkdir()
        module.new_run_dir = lambda path: run_dir  # keep results/app/ clean
        detector = FakeDetector()
        plot = lambda result, out: Path(out).write_bytes(b"png")  # noqa: E731
        engine = module.AIGVDetEngine(lambda: detector, plot)
        engine.load_async()
        assert engine.events.get(timeout=10)[0] == "models_ready" and engine.ready
        request = engine.analyze_async("clip.mp4", mode="benchmark")
        events = _drain(engine, request)
        assert [e for e, _ in events] == ["analysis_progress", "analysis_progress", "analysis_done"]
        assert events[0][1] == (1, 2) and events[1][1] == (2, 2)
        result = events[-1][1]
        assert detector.calls == [("clip.mp4", "benchmark", "whole", True)]
        assert "crops" not in result and result["figure"] == str(run_dir / "analysis.png")
        assert (run_dir / "analysis.png").exists()
        assert json.loads((run_dir / "result.json").read_text())["pairs_used"] == 4


def test_engine_failure_paths() -> None:
    module = _app_imports()
    with tempfile.TemporaryDirectory() as folder:
        module.new_run_dir = lambda path: Path(folder)
        engine = module.AIGVDetEngine(lambda: FakeDetector(ValueError("Clip too short: 3 frames")),
                                      lambda result, out: None)  # fmt: skip
        early = engine.analyze_async("clip.mp4")  # before models are loaded
        assert "not loaded" in engine.events.get(timeout=5)[1] and early is not None
        engine.load_async()
        engine.events.get(timeout=10)
        request = engine.analyze_async("clip.mp4")
        message = _drain(engine, request)[-1]
        assert message[0] == "analysis_failed" and "cannot be analyzed" in message[1]
        assert "too short" in message[1]

        busy = module.AIGVDetEngine(lambda: FakeDetector(delay=0.4), lambda result, out: None)
        busy.load_async()
        busy.events.get(timeout=10)
        first = busy.analyze_async("a.mp4")
        second = busy.analyze_async("b.mp4")  # rejected immediately, not queued
        rejected = _drain(busy, second)[-1]
        assert rejected[0] == "analysis_failed" and "one video at a time" in rejected[1]
        assert _drain(busy, first)[-1][0] == "analysis_done"

        broken = module.AIGVDetEngine(lambda: (_ for _ in ()).throw(RuntimeError("no weights")))
        broken.load_async()
        event = broken.events.get(timeout=10)
        assert event[0] == "models_failed" and "no weights" in event[1] and not broken.ready


def _tk_root():
    import tkinter

    try:
        root = tkinter.Tk()
    except tkinter.TclError:
        return None  # no display: the UI smoke test is skipped
    root.withdraw()
    return root


def test_video_views_render_every_state() -> None:
    _app_imports()
    from types import SimpleNamespace

    root = _tk_root()
    if root is None:
        print("      (no display - skipped)")
        return
    try:
        from ui.video_menu_view import VideoMenuView
        from ui.video_view import AIGVDET_PROFILE, VideoView

        engine = SimpleNamespace(ready=True, load_error=None, missing_models=False)
        calls = []
        app = SimpleNamespace(
            video_engine=engine, aigvdet_engine=engine,
            show_home=lambda: calls.append("home"), show_video_menu=lambda: calls.append("menu"),
            show_video_forensics=lambda: calls.append("nsg"), show_aigvdet=lambda: calls.append("ai"),
        )  # fmt: skip
        menu = VideoMenuView(root, app)
        assert "ready" in menu.aigvdet_status.get()
        app.aigvdet_engine = SimpleNamespace(ready=False, load_error="x", missing_models=True)
        menu._refresh_status()
        assert "pull_export" in menu.aigvdet_status.get()

        with tempfile.TemporaryDirectory() as folder:
            clip = Path(folder) / "clip.mp4"
            write_clip(clip, frames=12)
            figure = Path(folder) / "fig.png"
            figure.write_bytes(_tiny_png())
            for profile in (None, AIGVDET_PROFILE):
                view = VideoView(root, app, profile=profile)
                view.selected_path = str(clip)
                view.state = "selected"
                view._render()  # probes the clip, shows the small-clip banner / mode choice
                view.state = "running"
                view._render()
                request = object()
                view._pending_request_id = request
                view.on_engine_event("analysis_progress", (1, 8), request)
                view.on_engine_event("analysis_progress", (2, 8), object())  # stale: ignored
                if profile is AIGVDET_PROFILE:
                    assert view._progress_label.cget("text") == "Frame pair 1 of 8"
                    result = FakeDetector().analyze("c.mp4")
                    result.update(figure=str(figure))
                    view.on_engine_event("analysis_done", result, request)
                    assert view.state == "results"
                    view.result["verdict_level"] = "uncertain"
                    view._render()
                    view.result["verdict_level"] = "flagged"
                    view._render()
                else:
                    view.on_engine_event("analysis_failed", "boom", request)
                    assert view.state == "error"
                    view._render()
    finally:
        root.destroy()


def _tiny_png() -> bytes:
    ok, buffer = cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))
    assert ok
    return buffer.tobytes()


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, function in tests:
        try:
            function()
            print(f"PASS  {name}")
        except Exception:  # noqa: BLE001 - report every failure, then exit non-zero
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
