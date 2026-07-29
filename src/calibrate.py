"""
Calibrate the AEROBLADE detection threshold on a local dataset of real and
generated images: computes Delta_Min (LPIPS2) for every image, reproduces a
small version of the paper's Fig. 3 score-separation histogram, reports AP
as a health check, and persists the chosen threshold to models/threshold.json.
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from autoencoders import load_autoencoders
from config import (
    AE_CONFIGS,
    DATA_DIR,
    LPIPS_LAYER_INDEX,
    LPIPS_LAYER_NAME,
    MODELS_DIR,
    RESULTS_DIR,
    THRESHOLD_PATH,
)
from distance import reconstruction_error
from reconstruct import load_image_tensor, reconstruct

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _iter_images(dir_path):
    dir_path = Path(dir_path)
    return sorted(p for p in dir_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def delta_min(image_path, aes):
    """Delta_Min (LPIPS2) for one image, without saving any artifacts."""
    x = load_image_tensor(str(image_path))
    distances = []
    for name, vae in aes.items():
        ae_type = AE_CONFIGS[name]["type"]
        x_tilde = reconstruct(vae, x, ae_type=ae_type)
        distances.append(reconstruction_error(x, x_tilde, layer=LPIPS_LAYER_INDEX))
    return min(distances)


def collect_scores(aes):
    real_dir = Path(DATA_DIR) / "real"
    generated_root = Path(DATA_DIR) / "generated"

    real_images = _iter_images(real_dir)
    generated_images = [
        p for sub in generated_root.iterdir() if sub.is_dir() for p in _iter_images(sub)
    ]

    if not real_images:
        raise RuntimeError(f"No real images found in {real_dir}.")
    if not generated_images:
        raise RuntimeError(f"No generated images found under {generated_root}.")

    real_scores = []
    for i, path in enumerate(real_images, 1):
        real_scores.append(delta_min(path, aes))
        print(f"[real {i}/{len(real_images)}] {path.name}: {real_scores[-1]:.4f}")

    gen_scores = []
    for i, path in enumerate(generated_images, 1):
        gen_scores.append(delta_min(path, aes))
        print(f"[generated {i}/{len(generated_images)}] {path.name}: {gen_scores[-1]:.4f}")

    return np.array(real_scores), np.array(gen_scores)


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
    aes = load_autoencoders()
    real_scores, gen_scores = collect_scores(aes)

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
