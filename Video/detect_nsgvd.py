"""Local video forensics: video -> 8 canonical frames -> NSG -> statistics -> verdict.

    python detect_nsgvd.py VIDEO [VIDEO ...] [--plot] [--json]
    python detect_nsgvd.py --parity          # check this PC reproduces Kaggle's features

This is the app-facing wrapper around the Kaggle bundle in Video/nsgvd_local/.
The bundle is used unmodified for everything that defines the features (frame
sampling, the guided-diffusion score, the NSG denominator, the 62 statistics) —
NSGVD_KAGGLE_HANDOFF.md §10 is explicit that changing any of it invalidates the
trained model. Two things are deliberately done differently here:

* the classifier is evaluated by Video/predictor.py (numpy) instead of the
  scikit-learn pickle, so this module does not pin the project's scikit-learn
  version for the image and speech modules. Identical probabilities — see
  Video/tools/export_tree_dump.py --verify.
* the result dict is enriched for the UI (three-level verdict, how much of the
  clip was actually analysed, caveats), without touching the decision rule:
  flagged iff p_fake > tau from the trained bundle.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import config as C
from predictor import TreeDumpPredictor

# Frame window each trained feature set was built with (handoff §7: detect.py
# forces this from the bundle rather than trusting a config file).
TRAINED_FRAME_WINDOW = {"nsg/steps5_res224": 95, "nsg/steps5_res224_full": 100_000}


@contextlib.contextmanager
def _sys_path(*directories):
    """Put directories on sys.path for the duration of the block, then remove them.

    Video/nsgvd_local/ contains a top-level detect.py, and third_party/ makes
    `guided_diffusion` importable. Leaving either on sys.path would let a later
    bare `import detect` in src/ or Speech/ resolve to the wrong file — the exact
    failure app/paths.py's isolated_import exists to prevent. Modules imported
    inside the block stay in sys.modules, so already-built objects keep working.
    """
    added = [str(d) for d in directories if str(d) not in sys.path]
    for directory in added:
        sys.path.insert(0, directory)
    try:
        yield
    finally:
        for directory in added:
            with contextlib.suppress(ValueError):
                sys.path.remove(directory)


_FETCH_HINT = (
    "Fetch the model files from Google Drive and rebuild the tree dump:\n"
    "  rclone copy gdrive:SPL3/Video-NSGVD/models/256x256_diffusion_uncond.pt "
    "Video/nsgvd_local/models\n"
    "  rclone copy gdrive:SPL3/Video-NSGVD/export/nsg_stats_gbm.joblib "
    "Video/nsgvd_local/models\n"
    "  python Video/tools/export_tree_dump.py --verify"
)


def unet_status() -> tuple[bool, str]:
    """(usable, human-readable reason) for the UNet checkpoint."""
    if not C.UNET_PATH.exists():
        return False, f"{C.UNET_PATH.name} is not downloaded"
    size = C.UNET_PATH.stat().st_size
    if size != C.UNET_BYTES:
        return False, (
            f"{C.UNET_PATH.name} is {size / 1e9:.2f} GB, expected "
            f"{C.UNET_BYTES / 1e9:.2f} GB — the download is incomplete or truncated"
        )
    return True, "ok"


def verify_unet_hash() -> str:
    """SHA-256 of the checkpoint (handoff §9 wants the prefix a37c32fffd31 once)."""
    import hashlib

    digest = hashlib.sha256()
    with open(C.UNET_PATH, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_model_files():
    if not C.TREE_DUMP_PATH.exists():
        raise FileNotFoundError(f"missing {C.TREE_DUMP_PATH}\n\n{_FETCH_HINT}")
    usable, reason = unet_status()
    if not usable:
        raise FileNotFoundError(f"{reason}\n\n{_FETCH_HINT}")


class VideoDetector:
    """Loads the score network and classifier once; `analyze(path)` scores one video.

    Construction costs ~20-40 s and ~2.2 GB of RAM (the 552 M-parameter UNet), so
    build one per process and keep it. `analyze()` is CPU-bound and must not run
    on a UI thread — app/video_engine.py does that on a worker thread.
    """

    def __init__(self, device: str | None = None, micro_batch: int | None = None):
        _check_model_files()
        self.predictor = TreeDumpPredictor(C.TREE_DUMP_PATH)
        self.tau = self.predictor.tau

        # The bundle's own modules, imported from the bundle directory.
        os.environ.setdefault("NSGVD_WORK", str(C.BUNDLE_DIR / "work"))
        os.environ.setdefault("NSGVD_SCRATCH", str(C.BUNDLE_DIR / "work" / "scratch"))
        with _sys_path(C.BUNDLE_DIR):
            import torch

            from nsgvd import config as NC
            from nsgvd.nsg import NSGEstimator, load_unet
            from nsgvd.stats import clip_stats
            from nsgvd.video import VideoDecodeError, read_clip

            NC.GUIDED_DIFFUSION_DIR = C.GUIDED_DIFFUSION_DIR
            # Frame sampling is taken from what the model was trained with, not
            # from the bundle's config file (handoff §7).
            feature_dir = self.predictor.meta.get("feature_dir")
            NC.FRAME_WINDOW = TRAINED_FRAME_WINDOW.get(feature_dir, NC.FRAME_WINDOW)
            self.frame_window = NC.FRAME_WINDOW

            self._clip_stats = clip_stats
            self._read_clip = read_clip
            self.VideoDecodeError = VideoDecodeError

            self.device = torch.device(device or C.DEVICE)
            if self.device.type == "cpu":
                torch.set_num_threads(os.cpu_count() or 4)
            self._torch = torch
            unet = load_unet(C.UNET_PATH, self.device)
            self.estimator = NSGEstimator(
                unet, self.device, micro_batch=micro_batch or C.UNET_MICRO_BATCH
            )

    # ---- feature extraction -------------------------------------------------

    def features(self, frames: np.ndarray) -> dict:
        """(T, 3, 224, 224) uint8 canonical frames -> the detector's 62 statistics."""
        torch = self._torch
        batch = torch.from_numpy(np.ascontiguousarray(frames)[None])
        score, denom = self.estimator.score_and_denom(batch)
        # fp16 round-trip: Kaggle stored the score maps as fp16, so the statistics
        # must be computed from the same rounded values (handoff §7, "Precision").
        score_np = score[0].half().float().cpu().numpy()
        return self._clip_stats(score_np, denom[0].double().cpu().numpy())

    # ---- decision -----------------------------------------------------------

    def _verdict(self, p_fake: float) -> tuple[str, str]:
        """(verdict text, UI level). The flag itself is p_fake > tau, nothing else."""
        if abs(p_fake - self.tau) < C.UNCERTAIN_BAND:
            return "Uncertain", "suspicious"
        if p_fake > self.tau:
            return "AI-generated", "generated"
        return "Consistent with real", "real"

    def analyze(self, path: str | Path) -> dict:
        started = time.time()
        frames, info = self._read_clip(path)
        stats = self.features(frames)
        p_fake = self.predictor.predict_one(stats)
        verdict, level = self._verdict(p_fake)

        fps = info.get("fps") or 0.0
        n_meta = info.get("n_meta") or 0
        analysed_frames = (info["indices"][-1] + 1) if info.get("indices") else 0
        return {
            "file": str(path),
            "p_fake": round(float(p_fake), 4),
            "threshold": round(float(self.tau), 4),
            "verdict": verdict,
            "verdict_level": level,
            "is_flagged": bool(p_fake > self.tau),
            "seconds": round(time.time() - started, 1),
            "frames_used": list(info["indices"]),
            "frame_window": self.frame_window,
            "frames_spanned": analysed_frames,
            "analysed_seconds": round(analysed_frames / fps, 2) if fps > 0 else None,
            "duration_seconds": round(n_meta / fps, 2) if (fps > 0 and n_meta > 0) else None,
            "fps": fps,
            "resolution": f"{info.get('width', 0)}x{info.get('height', 0)}",
            "codec": info.get("codec", "?"),
            "partial_clip": bool(n_meta > analysed_frames > 0),
            "caveats": list(C.CAVEATS),
            "stats": stats,
        }

    # ---- verification -------------------------------------------------------

    def parity(self, npz_path: str | Path | None = None, tolerance: float = 0.05) -> dict:
        """Recompute Kaggle's own test clips here and compare probabilities.

        The stored clips carry both the frames and the score/denominator Kaggle
        computed on a T4, so this separates "does this PC reproduce the features"
        from "does the classifier agree" — the only expected difference is fp32
        CPU vs fp16 GPU arithmetic.
        """
        npz_path = Path(npz_path or C.PARITY_SAMPLES_PATH)
        if not npz_path.exists():
            raise FileNotFoundError(
                f"missing {npz_path} — rclone copy "
                "gdrive:SPL3/Video-NSGVD/export/parity_samples.npz Video/nsgvd_local/models"
            )
        data = np.load(npz_path)
        rows, worst = [], 0.0
        for clip_id, frames, score_ref, denom_ref in zip(
            data["clip_ids"], data["frames"], data["score"], data["denom"]
        ):
            p_local = self.predictor.predict_one(self.features(frames))
            p_kaggle = self.predictor.predict_one(
                self._clip_stats(score_ref.astype(np.float32), denom_ref.astype(np.float64))
            )
            diff = abs(p_local - p_kaggle)
            worst = max(worst, diff)
            rows.append({"clip_id": str(clip_id), "p_local": p_local,
                         "p_kaggle": p_kaggle, "diff": diff})
        return {"rows": rows, "worst": worst, "passed": worst < tolerance,
                "tolerance": tolerance}


# ---- module-level API (mirrors Speech/detect_pretrained.py) ------------------

_DETECTOR = None


def load_detector(device: str | None = None) -> VideoDetector:
    """Process-wide detector, built on first use (the UNet load is expensive)."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = VideoDetector(device=device)
    return _DETECTOR


def detect_video(path, detector: VideoDetector | None = None, device: str | None = None) -> dict:
    return (detector or load_detector(device)).analyze(path)


def plot_report(result: dict, out_path) -> Path:
    """Per-frame NSG evidence behind one verdict, saved as a PNG.

    Shows what the classifier actually reads: the spatial part (score energy per
    frame) and the temporal part (the NSG denominator, whose sign flips are what
    "violates probability-flow continuity" looks like in practice), plus where
    p_fake sits relative to the calibrated threshold.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = result.get("stats") or {}
    n = len(result.get("frames_used") or []) or C.T_FRAMES
    energy = [stats.get(f"logE{t}", np.nan) for t in range(n)]
    logden = [stats.get(f"logden{t}", np.nan) for t in range(n)]
    signs = [stats.get(f"sign{t}", 0.0) for t in range(n)]
    rough = [stats.get(f"hf{t}", np.nan) for t in range(n)]
    frame_idx = result.get("frames_used") or list(range(n))

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(range(n), energy, "o-", color="#1f4e79", label="score energy  log10 |s|")
    ax.plot(range(n), rough, "s--", color="#7a5b00", label="spatial roughness")
    ax.set_xticks(range(n), [str(i) for i in frame_idx])
    ax.set_xlabel("source frame")
    ax.set_title("Spatial part of NSG — diffusion score per sampled frame")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    colors = ["#b02a2a" if s < 0 else "#1e7e34" for s in signs]
    ax.bar(range(n), logden, color=colors)
    ax.set_xticks(range(n), [str(i) for i in frame_idx])
    ax.set_xlabel("source frame")
    ax.set_ylabel("log10 |denominator|")
    ax.set_title("Temporal part of NSG — magnitude, with sign (green +, red −)")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    p_fake, tau = result["p_fake"], result["threshold"]
    flagged = result.get("is_flagged", p_fake > tau)
    ax.barh([0], [p_fake], color="#b02a2a" if flagged else "#1e7e34", height=0.45)
    ax.axvline(tau, color="black", linestyle="--", linewidth=1.4)
    ax.annotate(f"threshold {tau:.3f}", (tau, 0.32), fontsize=8,
                ha="center", va="bottom")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("P(AI-generated)")
    ax.set_title(f"{result['verdict']} — p_fake = {p_fake:.3f}")

    span = ""
    if result.get("analysed_seconds") is not None:
        span = f" · analysed first {result['analysed_seconds']:.1f}s"
        if result.get("duration_seconds"):
            span += f" of {result['duration_seconds']:.1f}s"
    fig.suptitle(f"{Path(result['file']).name} — NSG video analysis{span}", fontsize=11)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# ---- CLI --------------------------------------------------------------------

def _print_result(result: dict, as_json: bool) -> None:
    if as_json:
        printable = {k: v for k, v in result.items() if k != "stats"}
        print(json.dumps(printable))
        return
    span = ""
    if result.get("analysed_seconds") is not None:
        span = f" analysed {result['analysed_seconds']:.1f}s"
        if result.get("duration_seconds"):
            span += f"/{result['duration_seconds']:.1f}s"
    print(f"{result['verdict']:22s} p_fake={result['p_fake']:.3f} "
          f"(threshold {result['threshold']:.3f}) {result['resolution']}{span} "
          f"{result['seconds']}s  {Path(result['file']).name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("videos", nargs="*", help="video files to analyse")
    parser.add_argument("--parity", nargs="?", const=str(C.PARITY_SAMPLES_PATH),
                        help="verify against Kaggle's stored clips (default: the bundled file)")
    parser.add_argument("--plot", action="store_true", help="also write <stem>_nsg.png")
    parser.add_argument("--json", action="store_true", help="print one JSON object per video")
    parser.add_argument("--out-dir", default=str(C.RESULTS_DIR), help="where --plot writes")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: cpu)")
    args = parser.parse_args(argv)

    if not args.videos and not args.parity:
        parser.error("give at least one video, or --parity")

    detector = load_detector(args.device)
    status = 0

    if args.parity:
        report = detector.parity(args.parity)
        for row in report["rows"]:
            print(f"{row['clip_id'][:44]:44s} local={row['p_local']:.4f} "
                  f"kaggle={row['p_kaggle']:.4f} |diff|={row['diff']:.4f}")
        print(f"PARITY {'OK' if report['passed'] else 'CHECK'}: largest difference "
              f"{report['worst']:.4f} (fp32 CPU vs fp16 GPU, tolerance {report['tolerance']})")
        status = 0 if report["passed"] else 1

    for video in args.videos:
        try:
            result = detector.analyze(video)
        except detector.VideoDecodeError as exc:
            print(f"SKIPPED  {Path(video).name}: {exc}")
            status = 1
            continue
        _print_result(result, args.json)
        if args.plot:
            out = Path(args.out_dir) / f"{Path(video).stem}_nsg.png"
            print(f"  figure -> {plot_report(result, out)}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
