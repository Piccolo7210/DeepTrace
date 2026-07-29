"""
Recalibrates the AEROBLADE threshold to target a specific false-positive rate (FPR)
on real images, instead of blindly maximizing F1 on the (imbalanced) real/generated
pool the way compute_threshold.py does. With far more generated images (7000) than
real (~554-1024), F1-maximization implicitly tolerates a higher real-image
false-positive rate than is acceptable in practice, since a chunk of misclassified
real images barely dents precision when true positives number in the thousands.
This mirrors the paper's own TPR@5%FPR evaluation metric, used here as the actual
deployment criterion instead of just a reported number.

Reads real_scores.csv / gen_scores.csv from the project root (same as
compute_threshold.py) and writes models/threshold.json (same file src/detect.py
reads) — this overwrites whatever threshold is currently calibrated, be it from
compute_threshold.py or a previous run of this script.

Usage (from the project root):
    python src/calibrate_target_fpr.py --target-fpr 0.05
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score

from config import LPIPS_LAYER_NAME, MODELS_DIR, RESULTS_DIR, THRESHOLD_PATH

REAL_SCORES_PATH = "real_scores.csv"
GEN_SCORES_PATH = "gen_scores.csv"


def _load_scores_csv(path):
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Missing {path} (expected in the project root).")
    scores = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row["delta_min"]))
    scores = np.array(scores)
    print(f"{path.name}: {len(scores)} scores")
    return scores


def choose_threshold_at_fpr(real_scores, gen_scores, target_fpr):
    """
    Picks the Delta_Min threshold below which at most `target_fpr` fraction of real
    images fall (i.e. would be false-positively called "generated"). Since lower
    Delta_Min means more generated-like, this is simply the target_fpr-th percentile
    of the real score distribution.
    """
    threshold = float(np.percentile(real_scores, target_fpr * 100))

    # matches detect.py's actual decision rule: is_generated = delta_min < threshold
    achieved_fpr = float(np.mean(real_scores < threshold))
    tpr = float(np.mean(gen_scores < threshold))  # recall on generated images

    y_true = np.concatenate([np.zeros_like(real_scores), np.ones_like(gen_scores)])
    y_score = -np.concatenate([real_scores, gen_scores])
    ap = average_precision_score(y_true, y_score)  # threshold-independent, unchanged

    return threshold, achieved_fpr, tpr, ap


def plot_histogram(real_scores, gen_scores, threshold, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    upper = max(real_scores.max(), gen_scores.max()) * 1.05
    bins = np.linspace(0, upper, 40)
    ax.hist(real_scores, bins=bins, alpha=0.6, label="Real")
    ax.hist(gen_scores, bins=bins, alpha=0.6, label="Generated")
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.4f}")
    ax.set_xlabel(rf"$\Delta_{{Min}}$ (LPIPS{LPIPS_LAYER_NAME})")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recalibrate the AEROBLADE threshold to target a false-positive rate on real images."
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.05,
        help="Target fraction of real images allowed to be misclassified as generated (default 0.05 = 5%%).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    real_scores = _load_scores_csv(REAL_SCORES_PATH)
    gen_scores = _load_scores_csv(GEN_SCORES_PATH)
    print(f"\nTotal: {len(real_scores)} real, {len(gen_scores)} generated")

    threshold, achieved_fpr, tpr, ap = choose_threshold_at_fpr(real_scores, gen_scores, args.target_fpr)

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    hist_path = Path(RESULTS_DIR) / "calibration_histogram_fpr.png"
    plot_histogram(real_scores, gen_scores, threshold, hist_path)

    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    with open(THRESHOLD_PATH, "w") as f:
        json.dump(
            {
                "threshold": float(threshold),
                "distance_metric": f"lpips_{LPIPS_LAYER_NAME}_min",
                "calibration_method": f"target_fpr_{args.target_fpr}",
                "average_precision": float(ap),
                "achieved_fpr_on_real": achieved_fpr,
                "tpr_on_generated": tpr,
                "num_real": int(len(real_scores)),
                "num_generated": int(len(gen_scores)),
                "calibrated_at": datetime.now().isoformat(timespec="seconds"),
                "computed_on": "kaggle dual T4 (scores) + local (FPR-targeted threshold)",
            },
            f,
            indent=2,
        )

    print(f"\nTarget FPR = {args.target_fpr:.2%}")
    print(f"Achieved FPR on real images = {achieved_fpr:.2%}")
    print(f"TPR on generated images (recall) = {tpr:.2%}")
    print(f"AP (threshold-independent) = {ap:.4f}")
    print(f"Chosen threshold (Delta_Min) = {threshold:.4f}")
    print(f"Saved threshold to {THRESHOLD_PATH}")
    print(f"Saved score histogram to {hist_path}")


if __name__ == "__main__":
    main()
