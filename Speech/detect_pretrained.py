"""
Single-video inference for the pretrained-SyncNet pipeline: runs the full chain
(face crop -> SyncNet S(i,.) features -> AR anomaly model) on one raw video and
reports a verdict against the checkpoint's own validated threshold. This is the
module app/speech_engine.py calls; mirrors detect.py's role for the from-scratch
pipeline (which never got a trained checkpoint to run against).

Usage:
    python detect_pretrained.py --video path/to/clip.mp4 --plot
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import PRETRAINED_MODEL_PATH, RESULTS_DIR
from models.ar_model import load_pretrained_ar_model, sequence_anomaly_score
from syncnet_wrapper import extract_from_video, load_syncnet

# Which key of the checkpoint's `thresholds` table to use as the decision boundary.
#
# That table is a percentile table of the AR model's anomaly score over REAL
# held-out video -- except "p95", which speechcode.ipynb's Cell 14 overwrote with
# (max_real_score + min_fake_score)/2 from a 20-clip desync test, deliberately
# keeping the key name "for compatibility". The tell is visible in the stored
# values themselves: p95 (2.0733) is LOWER than p90 (2.1848), which no real
# percentile sequence can be. That number actually sits between p75 and p90, i.e.
# around the ~80th percentile of genuine real video -- so using it flags roughly
# one in five authentic clips, and it did exactly that during integration testing
# (an unmodified, correctly-synced GRID clip that SyncNet itself scored at
# offset=0 / confidence=5.46 came back "Anomalous").
#
# p99 is an intact, genuine percentile and is the deployed default: it bounds
# false accusations on real video at ~1% of the training distribution while still
# leaving the desync positive control well clear of the line (2.83 vs 2.52). That
# choice -- prefer bounded false-positive rate over raw sensitivity -- is the same
# one this repo already made deliberately for the image module, where
# src/calibrate_target_fpr.py replaced F1-maximization for the same reason
# (see ARCHITECTURE.md §5). Override with --threshold-key / --threshold.
DEFAULT_THRESHOLD_KEY = "p99"

# Localized-manipulation rule (see find_anomalous_regions).
#
# Mean-over-windows alone only catches uniformly-fake video: a clip where just one
# stretch is dubbed or spliced averages its bad windows against its good ones and
# lands under the line. Measured on a real manipulated clip: per-window scores
# 2.16 / 2.16 / 2.30 / 2.50 / 2.64 -> mean 2.35, under a 2.52 threshold, despite the
# last two windows being at or over it and the delay band visibly drifting off
# centre. So the verdict is a disjunction: global mean OR a sustained local region.
#
# "Sustained" matters because taking a max over N windows inflates the false-positive
# rate with clip length (test enough windows at 1% each and a long genuine video
# trips eventually). Rather than pushing the per-window threshold out to an extreme
# quantile the checkpoint's table doesn't even contain, require a RUN of windows --
# persistence is the more robust correction when windows overlap by
# (seq_len - stride) and their scores are therefore strongly correlated.
#
# Hysteresis (trigger high, extend low) is used for the run's extent because a
# window straddling the boundary of a manipulated segment genuinely contains both
# real and fake frames, so its score is diluted by construction -- the same reason
# Canny edge detection extends edges through a lower threshold.
EXTEND_THRESHOLD_KEY = "p90"
MIN_RUN_WINDOWS = 2


def resolve_threshold(ckpt, threshold_key=DEFAULT_THRESHOLD_KEY):
    """Looks up a decision threshold in the checkpoint's percentile table. Returns
    (value, is_suspect) — is_suspect flags the overwritten "p95" entry described in
    DEFAULT_THRESHOLD_KEY's comment, detected structurally (a percentile that sits
    below a lower one) rather than by hardcoding the number, so it stays correct if
    the checkpoint is ever retrained."""
    thresholds = ckpt["thresholds"]
    if threshold_key not in thresholds:
        available = sorted(k for k, v in thresholds.items() if isinstance(v, (int, float)))
        raise KeyError(f"No threshold {threshold_key!r} in checkpoint; available: {available}")

    value = float(thresholds[threshold_key])
    ordered = ["p50", "p75", "p90", "p95", "p99"]
    is_suspect = any(
        threshold_key == key
        and (lower := thresholds.get(ordered[i - 1])) is not None
        and value < float(lower)
        for i, key in enumerate(ordered)
        if i > 0
    )
    return value, is_suspect


def build_windows(num_frames, seq_len, stride):
    """Fixed-size sliding windows over a (T, DELAY_BINS) feature sequence, in the
    same shape as speechcode.ipynb's SyncFeatureDataset (range(0, T-seq_len+1,
    stride)) -- the windowing the checkpoint's threshold was actually calibrated
    against -- plus one final window ending at T, so an ad-hoc clip's trailing
    frames are never silently left unscored (harmless for bulk training-window
    sampling; more noticeable for a single user clip). Falls back to one (0, T)
    window when the clip is shorter than seq_len.
    """
    if num_frames < 2:
        raise ValueError(f"Need at least 2 scored frames to detect anomaly, got {num_frames}.")
    if num_frames <= seq_len:
        return [(0, num_frames)]

    bounds = [(s, s + seq_len) for s in range(0, num_frames - seq_len + 1, stride)]
    if bounds[-1][1] != num_frames:
        bounds.append((num_frames - seq_len, num_frames))
    return bounds


def score_windows(model, features, seq_len, stride, device="cpu"):
    """Scores every window with the AR model's per-frame mean loss (paper Eq. 4,
    normalized like detect.py's own `score = total / len(losses)`). Returns
    (window_scores (W,), window_bounds, first-window per-step loss curve for
    plotting)."""
    num_frames = features.shape[0]
    bounds = build_windows(num_frames, seq_len, stride)
    seq = torch.from_numpy(features).to(device)

    scores = []
    first_curve = None
    for i, (start, end) in enumerate(bounds):
        window = seq[start:end]
        total, losses = sequence_anomaly_score(model, window)
        scores.append(total / max(1, len(losses)))
        if i == 0:
            first_curve = losses.cpu().numpy()

    return np.array(scores, dtype=np.float32), bounds, first_curve


def find_anomalous_regions(window_scores, window_bounds, trigger, extend,
                            min_run_windows=MIN_RUN_WINDOWS):
    """Contiguous runs of windows whose score stays above `extend` and peaks above
    `trigger`, at least `min_run_windows` long. Returns a list of regions with frame
    spans, so a partially-manipulated clip gets localized rather than averaged away.
    See EXTEND_THRESHOLD_KEY's comment for why hysteresis + a run-length floor
    instead of a bare max-over-windows."""
    above_extend = window_scores > extend
    regions = []
    i, n = 0, len(window_scores)

    while i < n:
        if not above_extend[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and above_extend[j + 1]:
            j += 1

        run = window_scores[i:j + 1]
        if len(run) >= min_run_windows and run.max() > trigger:
            regions.append({
                "start_frame": int(window_bounds[i][0]),
                "end_frame": int(window_bounds[j][1]),
                "num_windows": int(len(run)),
                "peak_score": float(run.max()),
            })
        i = j + 1

    return regions


def detect_pretrained(video_path, model, ckpt, syncnet, work_dir, device="cpu", cleanup=True,
                       threshold_key=DEFAULT_THRESHOLD_KEY, threshold=None):
    """End-to-end: raw video -> face crop -> S(i,.) -> sliding-window AR scoring ->
    verdict. Uses the *mean* across windows as the verdict statistic (not max) --
    the checkpoint's threshold was validated on single, unsliced windows
    (speechcode.ipynb's own desync test truncates each clip to one feats[:seq_len]
    window; it never slides or aggregates), so mean-across-windows is the aggregate
    statistic that stays closest in scale/distribution to what was actually
    validated. Max grows with clip length/window count purely from having more
    chances to spike, so using it as the *verdict* would inflate false positives on
    longer real videos; it's surfaced only as a secondary "peak anomaly" diagnostic.
    """
    seq_len = ckpt.get("seq_len", 50)
    stride = ckpt.get("stride", 10)
    threshold_suspect = False
    if threshold is None:
        threshold, threshold_suspect = resolve_threshold(ckpt, threshold_key)
    else:
        threshold_key = None  # explicit override; no percentile key describes it

    extracted = extract_from_video(syncnet, video_path, work_dir)
    features = extracted.features
    num_frames = features.shape[0]

    window_scores, window_bounds, first_curve = score_windows(
        model, features, seq_len, stride, device
    )

    mean_score = float(window_scores.mean())
    max_score = float(window_scores.max())
    peak_idx = int(window_scores.argmax())

    extend_threshold, _ = resolve_threshold(ckpt, EXTEND_THRESHOLD_KEY)
    regions = find_anomalous_regions(window_scores, window_bounds, threshold, extend_threshold)

    fps = ckpt.get("fps", 25) or 25
    for region in regions:
        region["start_sec"] = region["start_frame"] / fps
        region["end_sec"] = region["end_frame"] / fps

    # Disjunction: a clip is flagged if it looks bad overall OR if any sustained
    # stretch of it does. The two are reported as different levels rather than
    # collapsed, because "the whole clip is off" and "2.4s in the middle are off"
    # are different findings and warrant different follow-up.
    global_hit = mean_score > threshold
    if global_hit:
        verdict_level = "anomalous"
        verdict = "Anomalous (audio-visual desync)"
    elif regions:
        verdict_level = "suspicious"
        verdict = "Suspicious (localized desync)"
    else:
        verdict_level = "real"
        verdict = "Consistent with real"

    result = {
        "num_frames": int(num_frames),
        "num_windows": len(window_scores),
        "num_face_tracks": extracted.num_face_tracks,
        "mean_window_score": mean_score,
        "max_window_score": max_score,
        "peak_window_bounds": window_bounds[peak_idx],
        "window_scores": window_scores,
        "window_bounds": window_bounds,
        "features": features,
        "first_window_loss_curve": first_curve,
        "syncnet_offset": extracted.offset,
        "syncnet_confidence": extracted.confidence,
        "threshold": threshold,
        "threshold_key": threshold_key,
        "threshold_suspect": threshold_suspect,
        "extend_threshold": extend_threshold,
        "threshold_calibrated": bool(ckpt["thresholds"].get("validated_on_desync_test", False)),
        "regions": regions,
        "verdict": verdict,
        "verdict_level": verdict_level,
        "short_clip_warning": num_frames < seq_len,
    }

    if cleanup:
        shutil.rmtree(work_dir, ignore_errors=True)

    return result


def plot_report(result, out_path):
    features = result["features"]
    window_scores = result["window_scores"]
    window_bounds = result["window_bounds"]
    threshold = result["threshold"]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [2, 1]})

    axes[0].imshow(features.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1)
    axes[0].set_ylabel("Delay bin (S(i,.))")
    axes[0].set_xlabel("Frame")
    # The centre row is zero delay: on a synced clip the bright band sits on it, so
    # drawing it makes a constant offset readable straight off the heatmap.
    axes[0].axhline(features.shape[1] // 2, color="cyan", linewidth=0.8, alpha=0.6)
    axes[0].set_title(
        f"Synchronization features — SyncNet offset {result['syncnet_offset']} frames, "
        f"confidence {result['syncnet_confidence']:.2f}"
    )

    # Shade any sustained region the localized rule flagged, on both panels, so the
    # heatmap and the score curve can be read against the same span.
    for region in result.get("regions", []):
        for ax in axes:
            ax.axvspan(region["start_frame"], region["end_frame"],
                       color="red", alpha=0.15, zorder=0)

    centers = [(s + e) / 2 for s, e in window_bounds]
    axes[1].plot(centers, window_scores, marker="o")
    axes[1].axhline(threshold, color="black", linestyle="--", label=f"trigger={threshold:.3f}")
    extend = result.get("extend_threshold")
    if extend is not None:
        axes[1].axhline(extend, color="grey", linestyle=":", label=f"extend={extend:.3f}")
    axes[1].set_xlabel("Frame (window center)")
    axes[1].set_ylabel("Per-window mean loss")
    axes[1].set_title(f"Mean score = {result['mean_window_score']:.4f}  |  {result['verdict']}")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pretrained-SyncNet audio-visual anomaly detection on one video."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--ar-checkpoint", type=Path, default=PRETRAINED_MODEL_PATH)
    parser.add_argument("--threshold-key", type=str, default=DEFAULT_THRESHOLD_KEY,
                        help="Percentile key from the checkpoint's threshold table "
                             "(p50/p75/p90/p95/p99). Note p95 is not a real "
                             "percentile — see DEFAULT_THRESHOLD_KEY in this file.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Explicit threshold value, overriding --threshold-key.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading models on {device}...")
    syncnet = load_syncnet(device)
    model, ckpt = load_pretrained_ar_model(args.ar_checkpoint, device=device)

    stem = args.video.stem
    work_dir = args.out_dir / f"{stem}_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    result = detect_pretrained(
        args.video, model, ckpt, syncnet, work_dir, device=device,
        cleanup=not args.keep_work_dir,
        threshold_key=args.threshold_key, threshold=args.threshold,
    )

    print(f"\n{stem}")
    print(f"  frames = {result['num_frames']}  windows = {result['num_windows']}  face tracks = {result['num_face_tracks']}")
    print(f"  SyncNet offset = {result['syncnet_offset']} frames  confidence = {result['syncnet_confidence']:.3f}")
    print(f"  mean window score = {result['mean_window_score']:.4f}  max = {result['max_window_score']:.4f}")
    key_label = f" [{result['threshold_key']}]" if result["threshold_key"] else " [explicit]"
    print(f"  threshold = {result['threshold']:.4f}{key_label}")
    print(f"  verdict = {result['verdict']}")
    if result["threshold_suspect"]:
        print("  warning: the selected threshold is lower than a lower percentile in "
              "the checkpoint's own table — it is not a real percentile (see "
              "DEFAULT_THRESHOLD_KEY) and will flag a large share of genuine video.")
    for region in result["regions"]:
        print(f"  flagged region: frames {region['start_frame']}-{region['end_frame']} "
              f"({region['start_sec']:.2f}s-{region['end_sec']:.2f}s), "
              f"{region['num_windows']} windows, peak {region['peak_score']:.4f}")
    if result["short_clip_warning"]:
        print(f"  warning: clip shorter than the model's window size ({ckpt.get('seq_len', 50)} frames) -- scored as a single, undersized window.")

    if args.plot:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = args.out_dir / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}_analysis.png"
        plot_report(result, plot_path)
        print(f"  saved plot: {plot_path}")


if __name__ == "__main__":
    main()
