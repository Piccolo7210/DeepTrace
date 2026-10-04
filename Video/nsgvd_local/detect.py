"""Local NSG detector: video -> 8 canonical frames -> NSG (guided-diffusion score) -> statistics -> classifier.

    python detect.py VIDEO [VIDEO ...] [--json]
    python detect.py --parity models/parity_samples.npz     # check this PC reproduces Kaggle's features

Expected layout next to this file:
    nsgvd/                                 __init__ config utils video nsg stats (.py)
    third_party/guided-diffusion/          openai/guided-diffusion @ 22e0df8 (pinned)
    models/256x256_diffusion_uncond.pt     UNet weights      (Drive: models/)
    models/nsg_stats_gbm.joblib            trained detector  (Drive: export/)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NSGVD_WORK", str(ROOT / "work"))
os.environ.setdefault("NSGVD_SCRATCH", str(ROOT / "work" / "scratch"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from nsgvd import config as C  # noqa: E402

C.GUIDED_DIFFUSION_DIR = ROOT / "third_party" / "guided-diffusion"
UNET_PATH = ROOT / "models" / C.UNET_FILENAME
DETECTOR_PATH = ROOT / "models" / "nsg_stats_gbm.joblib"
# frame sampling the detector was trained with (features recorded in the bundle's "feature_dir")
TRAINED_FRAME_WINDOW = {"nsg/steps5_res224": 95, "nsg/steps5_res224_full": 100_000}

from nsgvd.nsg import NSGEstimator, load_unet  # noqa: E402
from nsgvd.stats import clip_stats  # noqa: E402
from nsgvd.video import read_clip  # noqa: E402


class Detector:
    """Loads the score network + classifier once; `analyze(path)` scores one video."""

    def __init__(self, device: str | None = None):
        for path in (UNET_PATH, DETECTOR_PATH):
            if not path.exists():
                raise FileNotFoundError(f"missing {path} -- download it from Google Drive (see README)")
        self.bundle = joblib.load(DETECTOR_PATH)
        C.FRAME_WINDOW = TRAINED_FRAME_WINDOW.get(self.bundle.get("feature_dir"), C.FRAME_WINDOW)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)
        self.est = NSGEstimator(load_unet(UNET_PATH, self.device), self.device, micro_batch=8)

    def features(self, frames: np.ndarray) -> dict:
        """(T, 3, 224, 224) uint8 canonical frames -> the detector's NSG statistics."""
        score, denom = self.est.score_and_denom(torch.from_numpy(np.ascontiguousarray(frames)[None]))
        s = score[0].half().float().cpu().numpy()          # same fp16 rounding as the stored training features
        return clip_stats(s, denom[0].double().cpu().numpy())

    def predict(self, stats: dict) -> float:
        x = pd.DataFrame([stats])[self.bundle["features"]]
        return float(self.bundle["model"].predict_proba(x)[0, 1])

    def analyze(self, path: str | Path) -> dict:
        t0 = time.time()
        frames, info = read_clip(path)
        p_fake = self.predict(self.features(frames))
        tau = float(self.bundle["tau_fpr05"])
        return {"file": str(path), "p_fake": round(p_fake, 4), "threshold": round(tau, 4),
                "verdict": "AI-generated" if p_fake > tau else "real", "seconds": round(time.time() - t0, 1),
                "frames_used": info["indices"], "fps": info["fps"], "resolution": f"{info['width']}x{info['height']}"}

    def parity(self, npz_path: str | Path) -> float:
        """Recompute 5 Kaggle test clips locally; returns the largest probability difference."""
        data = np.load(npz_path)
        worst = 0.0
        for cid, frames, s_ref, d_ref in zip(data["clip_ids"], data["frames"], data["score"], data["denom"]):
            p_local = self.predict(self.features(frames))
            p_kaggle = self.predict(clip_stats(s_ref.astype(np.float32), d_ref.astype(np.float64)))
            worst = max(worst, abs(p_local - p_kaggle))
            print(f"{str(cid)[:44]:44s} p_fake local={p_local:.4f} kaggle={p_kaggle:.4f} |diff|={abs(p_local - p_kaggle):.4f}")
        print(f"PARITY {'OK' if worst < 0.05 else 'CHECK'}: largest difference {worst:.4f} (fp32 CPU vs fp16 GPU)")
        return worst


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="*", help="video files (.mp4, .mov, .webm, .gif, ...)")
    parser.add_argument("--parity", help="parity_samples.npz from Drive export/")
    parser.add_argument("--json", action="store_true", help="print one JSON object per video")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    args = parser.parse_args(argv)
    if not args.videos and not args.parity:
        parser.error("give at least one video, or --parity FILE")
    detector = Detector(args.device)
    if args.parity:
        detector.parity(args.parity)
    for video in args.videos:
        result = detector.analyze(video)
        if args.json:
            print(json.dumps(result))
        else:
            print(f"{result['verdict']:13s} p_fake={result['p_fake']:.3f} (threshold {result['threshold']:.3f}) "
                  f"{result['resolution']} {result['seconds']}s  {result['file']}")


if __name__ == "__main__":
    main()
