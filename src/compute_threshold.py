"""
Computes the AEROBLADE calibration threshold from real_scores.csv and gen_scores.csv
(pre-computed elsewhere, e.g. on Kaggle via kaggle/generated_scores_dual_gpu.py and
its real-image equivalent) placed in the project root. Mirrors the threshold-
selection logic in calibrate.py exactly, just skipping the score-computation step
since that's already done.

Run from the project root:
    python src/compute_threshold.py
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

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


def choose_threshold(real_scores, gen_scores):
    """Picks the Delta_Min threshold that maximizes F1 for the 'generated' class."""
    y_true = np.concatenate([np.zeros_like(real_scores), np.ones_like(gen_scores)])
    # paper convention: higher score = more likely generated, so negate distance
    y_score = -np.concatenate([real_scores, gen_scores])

    ap = average_precision_score(y_true, y_score)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best_idx = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    best_score_threshold = thresholds[best_idx] if len(thresholds) else 0.0
    delta_threshold = -best_score_threshold  # back to Delta_Min space

    return delta_threshold, ap, float(f1[best_idx]) if len(thresholds) else 0.0


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


def main():
    real_scores = _load_scores_csv(REAL_SCORES_PATH)
    gen_scores = _load_scores_csv(GEN_SCORES_PATH)
    print(
        f"\nTotal: {len(real_scores)} real, {len(gen_scores)} generated "
        f"({len(real_scores) + len(gen_scores)} images)"
    )

    threshold, ap, best_f1 = choose_threshold(real_scores, gen_scores)

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    hist_path = Path(RESULTS_DIR) / "calibration_histogram.png"
    plot_histogram(real_scores, gen_scores, threshold, hist_path)

    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    with open(THRESHOLD_PATH, "w") as f:
        json.dump(
            {
                "threshold": float(threshold),
                "distance_metric": f"lpips_{LPIPS_LAYER_NAME}_min",
                "average_precision": float(ap),
                "best_f1": best_f1,
                "num_real": int(len(real_scores)),
                "num_generated": int(len(gen_scores)),
                "calibrated_at": datetime.now().isoformat(timespec="seconds"),
                "computed_on": "kaggle dual T4 (scores) + local (threshold)",
            },
            f,
            indent=2,
        )

    print(f"\nAP = {ap:.4f}  best F1 = {best_f1:.4f}")
    print(f"Chosen threshold (Delta_Min) = {threshold:.4f}")
    print(f"Saved threshold to {THRESHOLD_PATH}")
    print(f"Saved score histogram to {hist_path}")


if __name__ == "__main__":
    main()
