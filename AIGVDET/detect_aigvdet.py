"""Command-line front end of the AIGVDet detector (runs on the CPU).

Examples (from the AIGVDET folder)::

    python detect_aigvdet.py clip.mp4 --mode accurate
    python detect_aigvdet.py clip.mp4 --mode benchmark --window first95 --json
    python detect_aigvdet.py --parity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from aigvdet.detector import MODES, WINDOWS, ExportError, load_detector
from aigvdet.verification import PARITY_TOLERANCE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI-generated video detection (CPU).")
    parser.add_argument("video", nargs="?", type=Path, help="video file to analyse")
    parser.add_argument("--mode", choices=sorted(MODES), default="accurate")
    parser.add_argument("--window", choices=WINDOWS, default="whole")
    parser.add_argument("--pairs", type=int, default=None, help="override the mode's pair count")
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument("--parity", action="store_true", help="check this install against Kaggle")
    parser.add_argument("--no-verify", action="store_true", help="skip checksum verification")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--plot", type=Path, default=None, help="write the report figure (PNG)")
    return parser.parse_args(argv)


def run_parity(detector) -> int:
    """Gate G5: every stored Kaggle frame pair must be reproduced within the tolerance."""
    report = detector.parity()
    worst = float(report[["d_rgb", "d_flow", "d_fused"]].to_numpy().max())
    print(report.round(5).to_string(index=False))
    passed = worst <= PARITY_TOLERANCE
    print(
        f"\nlargest |delta p| = {worst:.5f} (tolerance {PARITY_TOLERANCE}): "
        f"{'PASSED' if passed else 'FAILED'}"
    )
    return 0 if passed else 1


def print_summary(result: dict) -> None:
    width, height = result["resolution"]
    print(
        f"Verdict: {result['verdict']}  (P(AI-generated) = {result['p_fake']:.3f}, "
        f"threshold {result['threshold']:.3f} +- {result['uncertain_band']:.2f})"
    )
    print(
        f"  RGB branch {result['p_spatial']:.3f} | flow branch {result['p_temporal']:.3f} "
        f"| alpha {result['alpha']}"
    )
    print(
        f"  {result['pairs_used']} pairs, {width}x{height}, {result['fps']:.1f} fps, "
        f"mode {result['mode']}, {result['seconds']:.0f} s"
    )
    for note in result["caveats"]:
        print(f"  note: {note}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    if not args.parity and args.video is None:
        print("error: give a video file or --parity", file=sys.stderr)
        return 2
    try:
        detector = load_detector(args.models_dir, verify=not args.no_verify)
        if args.parity:
            return run_parity(detector)
        result = detector.analyze(
            args.video,
            mode=args.mode,
            window=args.window,
            pairs=args.pairs,
            progress=lambda done, total: print(f"\rpair {done}/{total}", end="", file=sys.stderr),
            keep_crops=args.plot is not None,
        )
    except (ExportError, ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(file=sys.stderr)
    if args.plot is not None:
        from aigvdet.report import plot_report

        result["file"] = str(args.video)
        print(f"figure: {plot_report(result, args.plot)}", file=sys.stderr)
        result.pop("crops", None)  # arrays are not JSON; the figure already used them
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
