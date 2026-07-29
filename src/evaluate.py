"""
Reproduces the paper's evaluation tables from precomputed score CSVs, so our numbers
can be compared against the paper's directly.

The paper always evaluates PER GENERATOR — 1000 real vs 1000 images from one model at
a time (Tab. 1 for AP, Tab. 3 for TPR@5%FPR). It never reports a pooled "real vs all
generators at once" figure. Pooling is a strictly harder problem, because one global
threshold has to work for whichever generator separates worst, so a pooled number
looks worse than the paper's even when the method is working correctly. Both are
printed below: per-generator for comparability, pooled for the deployed detector.

Expects (in the project root):
    real_scores.csv   columns: file, delta_min
    gen_scores.csv    columns: model, file, delta_min

Run from the project root:
    python src/evaluate.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from config import LPIPS_LAYER_NAME

REAL_SCORES_PATH = "real_scores.csv"
GEN_SCORES_PATH = "gen_scores.csv"
TARGET_FPR = 0.05

# Paper Tab. 1 (LPIPS2, "Min" row) and Tab. 3, keyed by the substring that identifies
# each generator in our data/generated/ folder names.
PAPER_REFERENCE = {
    "stable-diffusion-v1-1": ("SD1.1", 0.979, 0.981),
    "stable-diffusion-v1-5": ("SD1.5", 0.978, 0.963),
    "stable-diffusion-2-1": ("SD2.1", 0.994, 0.995),
    "kandinsky-2-1": ("KD2.1", 0.999, 0.999),
    "midjourney-v4": ("MJ4", 0.999, 0.997),
    "midjourney-v5-1": ("MJ5.1", 0.996, 0.989),
    "midjourney-v5": ("MJ5", 0.997, 0.993),
}


def _paper_entry(model_name):
    """Longest matching key wins, so 'midjourney-v5-1' isn't matched by 'midjourney-v5'."""
    best = None
    for key, entry in PAPER_REFERENCE.items():
        if key in model_name and (best is None or len(key) > len(best[0])):
            best = (key, entry)
    return best[1] if best else (model_name[:20], None, None)


def load_real_scores(path=REAL_SCORES_PATH):
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Missing {path} (expected in the project root).")
    with open(path, newline="") as f:
        return np.array([float(row["delta_min"]) for row in csv.DictReader(f)])


def load_generated_scores(path=GEN_SCORES_PATH):
    """Returns {model_name: np.array(scores)}."""
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Missing {path} (expected in the project root).")
    by_model = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "model" not in reader.fieldnames:
            raise RuntimeError(
                f"{path} has no 'model' column — it was produced by an older script. "
                f"Re-score with kaggle/score_dual_gpu.py to get per-generator attribution."
            )
        for row in reader:
            by_model[row["model"]].append(float(row["delta_min"]))
    return {m: np.array(v) for m, v in by_model.items()}


def evaluate(real_scores, gen_scores, target_fpr=TARGET_FPR):
    """AP and TPR at a fixed FPR, for one real-vs-one-generated-set comparison."""
    y_true = np.concatenate([np.zeros_like(real_scores), np.ones_like(gen_scores)])
    # paper convention: higher score = more likely generated, so negate the distance
    y_score = -np.concatenate([real_scores, gen_scores])
    ap = float(average_precision_score(y_true, y_score))

    # threshold below which `target_fpr` of real images fall; matches detect.py's
    # decision rule (is_generated = delta_min < threshold)
    threshold = float(np.percentile(real_scores, target_fpr * 100))
    tpr = float(np.mean(gen_scores < threshold))
    return ap, tpr, threshold


def main():
    real_scores = load_real_scores()
    by_model = load_generated_scores()

    print(f"Metric: Delta_Min with LPIPS{LPIPS_LAYER_NAME}")
    print(f"Real images: {len(real_scores)}")
    print(
        f"Real Delta_Min: mean={real_scores.mean():.4f} "
        f"range=[{real_scores.min():.4f}, {real_scores.max():.4f}]  "
        f"(paper Fig. 3 spans roughly 0.02-0.065)"
    )
    print()

    header = f"{'Generator':<10} {'N':>5} {'mean':>7} {'AP':>7} {'paper':>7} {'TPR@5%':>8} {'paper':>7}"
    print(header)
    print("-" * len(header))

    for model_name in sorted(by_model):
        scores = by_model[model_name]
        label, paper_ap, paper_tpr = _paper_entry(model_name)
        ap, tpr, _ = evaluate(real_scores, scores)
        paper_ap_s = f"{paper_ap:.3f}" if paper_ap else "-"
        paper_tpr_s = f"{paper_tpr:.3f}" if paper_tpr else "-"
        print(
            f"{label:<10} {len(scores):>5} {scores.mean():>7.4f} "
            f"{ap:>7.3f} {paper_ap_s:>7} {tpr:>8.3f} {paper_tpr_s:>7}"
        )

    all_gen = np.concatenate(list(by_model.values()))
    pooled_ap, pooled_tpr, pooled_threshold = evaluate(real_scores, all_gen)
    print("-" * len(header))
    print(
        f"{'POOLED':<10} {len(all_gen):>5} {all_gen.mean():>7.4f} "
        f"{pooled_ap:>7.3f} {'n/a':>7} {pooled_tpr:>8.3f} {'n/a':>7}"
    )
    print(
        "\nPooled is what the deployed single-threshold detector actually faces; the "
        "paper reports only the per-generator columns, so compare those against 'paper'."
    )
    print(f"Pooled threshold at {TARGET_FPR:.0%} FPR = {pooled_threshold:.4f}")

    mean_ap = float(np.mean([evaluate(real_scores, s)[0] for s in by_model.values()]))
    print(f"Mean per-generator AP = {mean_ap:.4f}  (paper's mean AP = 0.992)")


if __name__ == "__main__":
    main()
