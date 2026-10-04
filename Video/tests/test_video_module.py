"""Tests for the video forensics module.

Run from the project root:
    venv312\\Scripts\\python Video\\tests\\test_video_module.py

Three groups, in increasing cost:
  1. classifier + decode  — fast, no model weights needed
  2. parity               — needs the 2.1 GB UNet; the handoff's acceptance check
  3. end-to-end           — needs the UNet; scores a synthetic clip

Groups 2 and 3 skip (not fail) when the UNet is absent, so the fast tests stay
useful on a machine that has not downloaded the weights yet.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(MODULE_ROOT / "nsgvd_local"))

import config as C  # noqa: E402
from predictor import TreeDumpPredictor  # noqa: E402

# Probabilities for the five clips in parity_samples.npz, computed from Kaggle's
# own stored score maps. Recorded when scikit-learn 1.6.1 was installed and the
# numpy predictor was verified to match its predict_proba to 1.1e-16, so these
# double as a regression test that the tree dump still decodes correctly.
EXPECTED_P_FAKE = [0.451327, 0.102795, 0.098229, 0.042790, 0.027721]

_passed, _failed, _skipped = [], [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (_passed if condition else _failed).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{(' - ' + detail) if detail else ''}")


def skip(name: str, why: str) -> None:
    _skipped.append(name)
    print(f"  SKIP  {name} - {why}")


def make_test_video(path: Path, n_frames: int = 24, size: int = 128, fps: float = 12.0) -> Path:
    """A synthetic clip: a bright square moving over a gradient, so there is real
    inter-frame motion for the NSG temporal term to pick up."""
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open an mp4v writer")
    gradient = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    for i in range(n_frames):
        frame = np.dstack([gradient, np.roll(gradient, i * 3, axis=1), gradient.T])
        x = int((size - 24) * i / max(n_frames - 1, 1))
        frame[size // 2 - 12:size // 2 + 12, x:x + 24] = 255
        writer.write(frame)
    writer.release()
    return path


# ---- 1. classifier + decode (no weights needed) -----------------------------

def test_tree_dump():
    print("\n[1] classifier (tree dump, no sklearn)")
    if not C.TREE_DUMP_PATH.exists():
        skip("tree dump present", f"{C.TREE_DUMP_PATH} missing — run Video/tools/export_tree_dump.py")
        return None
    predictor = TreeDumpPredictor(C.TREE_DUMP_PATH)
    check("62 features in trained order", len(predictor.feature_names) == 62,
          f"{len(predictor.feature_names)} features")
    check("threshold is the calibrated tau", abs(predictor.tau - 0.6915) < 1e-3,
          f"tau={predictor.tau:.4f}")
    check("all 578 trees decoded", predictor.n_trees == 578, f"{predictor.n_trees} trees")

    # sklearn-free: predicting must not require sklearn to be importable at all
    check("no sklearn import needed", "sklearn" not in sys.modules)

    row = np.zeros((1, len(predictor.feature_names)))
    probability = predictor.predict_proba(row)[0]
    check("probability in [0, 1]", 0.0 <= probability <= 1.0, f"p={probability:.4f}")
    return predictor


def test_stored_features(predictor):
    print("\n[2] classifier on Kaggle's stored NSG features")
    if predictor is None:
        skip("stored-feature regression", "no tree dump")
        return
    if not C.PARITY_SAMPLES_PATH.exists():
        skip("stored-feature regression", f"{C.PARITY_SAMPLES_PATH.name} missing")
        return
    from nsgvd.stats import clip_stats

    data = np.load(C.PARITY_SAMPLES_PATH)
    probabilities = [
        predictor.predict_one(clip_stats(score.astype(np.float32), denom.astype(np.float64)))
        for score, denom in zip(data["score"], data["denom"])
    ]
    worst = max(abs(p - e) for p, e in zip(probabilities, EXPECTED_P_FAKE))
    check("probabilities match the recorded values", worst < 1e-5, f"max diff {worst:.2e}")
    check("62 statistics produced per clip",
          len(clip_stats(data["score"][0].astype(np.float32),
                         data["denom"][0].astype(np.float64))) == 62)


def test_decode():
    print("\n[3] canonical frame sampling")
    from nsgvd import config as NC
    from nsgvd.video import VideoDecodeError, read_clip, sample_indices

    NC.FRAME_WINDOW = C.FRAME_WINDOW
    check("stride rule inside the window", sample_indices(95) == [0, 11, 22, 33, 44, 55, 66, 77],
          str(sample_indices(95)))
    check("short clip still yields 8 indices", len(sample_indices(8)) == 8)
    check("long clip is capped at the window", max(sample_indices(10_000)) < C.FRAME_WINDOW)
    try:
        sample_indices(7)
        check("fewer than 8 frames is rejected", False, "no error raised")
    except VideoDecodeError:
        check("fewer than 8 frames is rejected", True)

    with tempfile.TemporaryDirectory() as tmp:
        clip = make_test_video(Path(tmp) / "moving.mp4", n_frames=24)
        frames, info = read_clip(clip)
        check("frames shape is (8, 3, 224, 224)", frames.shape == (8, 3, 224, 224), str(frames.shape))
        check("frames are uint8", frames.dtype == np.uint8)
        check("8 distinct source frames", len(set(info["indices"])) == 8, str(info["indices"]))
        check("frames actually differ", frames.std(axis=0).mean() > 1.0,
              f"mean per-pixel std {frames.std(axis=0).mean():.2f}")

        short = make_test_video(Path(tmp) / "short.mp4", n_frames=4)
        try:
            read_clip(short)
            check("too-short video raises VideoDecodeError", False, "no error raised")
        except VideoDecodeError as exc:
            check("too-short video raises VideoDecodeError", True, str(exc))


# ---- 2. parity (needs the UNet) ---------------------------------------------

def test_parity_and_end_to_end():
    print("\n[4] parity against Kaggle + end-to-end analysis (needs UNet weights)")
    import detect_nsgvd

    usable, reason = detect_nsgvd.unet_status()
    if not usable:
        skip("parity", reason)
        skip("end-to-end analyze()", reason)
        return

    digest = detect_nsgvd.verify_unet_hash()
    check("UNet SHA-256 matches the official checkpoint",
          digest.startswith(C.UNET_SHA256_PREFIX), digest[:16] + "...")

    detector = detect_nsgvd.load_detector()
    report = detector.parity()
    for row in report["rows"]:
        print(f"      {row['clip_id'][:40]:40s} local={row['p_local']:.4f} "
              f"kaggle={row['p_kaggle']:.4f} diff={row['diff']:.4f}")
    check("PARITY OK (fp32 CPU reproduces fp16 GPU features)", report["passed"],
          f"largest difference {report['worst']:.4f} (tolerance {report['tolerance']})")

    with tempfile.TemporaryDirectory() as tmp:
        clip = make_test_video(Path(tmp) / "synthetic.mp4", n_frames=24)
        result = detector.analyze(clip)
        print(f"      synthetic clip: p_fake={result['p_fake']:.4f} "
              f"verdict={result['verdict']} in {result['seconds']}s")
        check("result has the documented keys",
              {"file", "p_fake", "threshold", "verdict", "seconds", "frames_used",
               "fps", "resolution"} <= set(result),
              f"missing {{'file','p_fake',...}} - set(result) = "
              f"{ {'file','p_fake','threshold','verdict','seconds','frames_used','fps','resolution'} - set(result) }")
        check("probability in [0, 1]", 0.0 <= result["p_fake"] <= 1.0)
        check("verdict is consistent with the threshold",
              result["is_flagged"] == (result["p_fake"] > result["threshold"]))
        check("deterministic (same clip, same probability)",
              detector.analyze(clip)["p_fake"] == result["p_fake"])

        figure = detect_nsgvd.plot_report(result, Path(tmp) / "report.png")
        check("report figure written", figure.exists() and figure.stat().st_size > 10_000,
              f"{figure.stat().st_size // 1024} KB" if figure.exists() else "missing")


def main() -> int:
    print("video module tests")
    predictor = test_tree_dump()
    test_stored_features(predictor)
    test_decode()
    test_parity_and_end_to_end()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed, {len(_skipped)} skipped")
    if _failed:
        print("failed: " + ", ".join(_failed))
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
