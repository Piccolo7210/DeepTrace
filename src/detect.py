"""
AEROBLADE-style single-image detector.

For an input image, reconstructs it through each candidate LDM autoencoder,
computes the LPIPS2 reconstruction error for each, takes the minimum
(Delta_Min, per Eq. 2 of the paper) as the detection score, and localizes that
error using whichever AE gave the best reconstruction: the un-averaged LPIPS2
error map is alpha-blended straight onto the input, and averaged within SLIC
superpixels so each region carries a quotable number instead of just a colour.
"""

import argparse
import json
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from PIL import Image
from skimage.segmentation import slic

from autoencoders import load_autoencoders
from config import (
    AE_CONFIGS,
    IMAGE_SIZE,
    LPIPS_LAYER_INDEX,
    LPIPS_LAYER_NAME,
    RESULTS_DIR,
    THRESHOLD_PATH,
)
from distance import reconstruction_error, reconstruction_heatmap
from reconstruct import load_image_tensor, reconstruct, tensor_to_pil

# Rough heuristic cutoff (LPIPS2, Delta_Min) used only until models/threshold.json
# exists. Based on the separation shown in Fig. 3 of the paper (generated images
# cluster near 0, real images spread out with mass above ~0.01). Run
# src/calibrate.py to replace this with a properly calibrated value.
DEFAULT_THRESHOLD = 0.01

# --- Error-localization / visualization settings -------------------------------
HEATMAP_CMAP = "inferno"
OVERLAY_ALPHA = 0.55       # blend weight of the error wash over the input
# The error map is long-tailed: a couple of extreme pixels would compress every
# other value into the dark end of the colormap. Saturating the wash at a high
# percentile keeps the mid-range readable; the colorbar is marked extend="max"
# so the clipping is visible rather than implied.
OVERLAY_PERCENTILE = 99
SLIC_N_SEGMENTS = 160      # SLIC target region count (actual count varies slightly)
SLIC_COMPACTNESS = 10.0    # lower = regions follow colour/texture edges more closely
N_LABELED_REGIONS = 6      # highest-error regions annotated on the figure
LABEL_MIN_DISTANCE = 70    # px between annotated centroids, so labels don't collide
N_REPORTED_REGIONS = 10    # highest-error regions written into the JSON report


def load_threshold():
    path = Path(THRESHOLD_PATH)
    if path.exists():
        with open(path) as f:
            return json.load(f)["threshold"], True
    return DEFAULT_THRESHOLD, False


def detect(image_path, aes=None, threshold=None, save_report=True, calibrated=None):
    """
    Runs the AEROBLADE detection pipeline on a single image.

    Returns a dict with per-AE distances, the Delta_Min score, the
    attributed (winning) AE, the verdict, and paths to any saved artifacts.
    """
    image_path = Path(image_path)
    if aes is None:
        aes = load_autoencoders()
    if threshold is None:
        threshold, calibrated = load_threshold()
    elif calibrated is None:
        # A caller that passes a threshold but no flag (e.g. main() forwarding the
        # uncalibrated default) must not have it recorded as calibrated.
        calibrated = Path(THRESHOLD_PATH).exists()

    # The paper (Sec. 7.1) only ever evaluates images whose short side is at
    # least IMAGE_SIZE, deliberately avoiding resizing. Below that, the Resize
    # in load_image_tensor becomes an *upscale*, and its interpolation blur
    # suppresses the fine detail that makes real images hard to reconstruct —
    # biasing Delta_Min downwards, i.e. towards an "AI-generated" verdict.
    with Image.open(image_path) as probe:
        source_size = probe.size
    low_resolution = min(source_size) < IMAGE_SIZE

    x = load_image_tensor(str(image_path))

    distances = {}
    reconstructions = {}
    for name, vae in aes.items():
        ae_type = AE_CONFIGS[name]["type"]
        x_tilde = reconstruct(vae, x, ae_type=ae_type)
        distances[name] = reconstruction_error(x, x_tilde, layer=LPIPS_LAYER_INDEX)
        reconstructions[name] = x_tilde

    winner = min(distances, key=distances.get)
    delta_min = distances[winner]
    score = -delta_min  # paper convention: higher score = more likely generated
    is_generated = delta_min < threshold

    heatmap = reconstruction_heatmap(x, reconstructions[winner], layer=LPIPS_LAYER_INDEX)
    input_rgb = np.asarray(tensor_to_pil(x), dtype=float) / 255.0
    segments, regions = region_attribution(input_rgb, heatmap)

    result = {
        "file": str(image_path),
        "distances": distances,
        "delta_min": delta_min,
        "attributed_ae": winner,
        "score": score,
        "threshold": threshold,
        "threshold_calibrated": calibrated,
        "verdict": "AI-generated (LDM)" if is_generated else "Real",
        "source_size": list(source_size),
        "low_resolution_warning": low_resolution,
        # Highest-error regions of the winning AE's reconstruction, so a write-up can
        # cite where the error concentrated instead of describing colours.
        "region_attribution": regions[:N_REPORTED_REGIONS],
    }

    if low_resolution:
        print(
            f"Warning: {image_path.name} is {source_size[0]}x{source_size[1]}, "
            f"smaller than {IMAGE_SIZE}x{IMAGE_SIZE}. Upscaling blur biases the "
            f"score towards 'AI-generated'; treat this verdict as low-confidence."
        )

    if save_report:
        out_dir = Path(RESULTS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem

        fig_path = out_dir / f"{stem}_analysis.png"
        _save_analysis_figure(
            x, reconstructions[winner], heatmap, segments, regions, result, fig_path
        )
        result["figure"] = str(fig_path)

        json_path = out_dir / f"{stem}_report.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        result["report"] = str(json_path)

    if not calibrated:
        print(
            f"Warning: no calibrated threshold found at {THRESHOLD_PATH}, "
            f"using uncalibrated default ({DEFAULT_THRESHOLD}). "
            f"Run src/calibrate.py to calibrate one."
        )

    return result


def region_attribution(image_rgb, heatmap, n_segments=SLIC_N_SEGMENTS):
    """
    Segments the input into SLIC superpixels — regions that follow colour/texture
    boundaries rather than an arbitrary grid, so they tend to land on coherent
    things (a face, a patterned fabric, a flat sky) — and averages the LPIPS error
    inside each. This converts "the bright parts reconstruct badly" into per-region
    numbers that can be quoted directly in a write-up.

    image_rgb is a (H, W, 3) float array in [0, 1]; heatmap is (H, W).

    Returns (segments, regions), regions sorted highest mean error first, each entry
    holding its label, mean error, area fraction, and centroid in (x, y) pixels.
    """
    segments = slic(
        image_rgb,
        n_segments=n_segments,
        compactness=SLIC_COMPACTNESS,
        start_label=0,
        channel_axis=-1,
    )

    flat = segments.ravel()
    n_labels = int(segments.max()) + 1
    counts = np.bincount(flat, minlength=n_labels).astype(float)
    # SLIC can leave a label unused; guard the division rather than filtering first,
    # so every array below stays indexable by label.
    safe_counts = np.maximum(counts, 1.0)

    means = np.bincount(flat, weights=heatmap.ravel(), minlength=n_labels) / safe_counts
    ys, xs = np.indices(segments.shape)
    cy = np.bincount(flat, weights=ys.ravel(), minlength=n_labels) / safe_counts
    cx = np.bincount(flat, weights=xs.ravel(), minlength=n_labels) / safe_counts

    regions = [
        {
            "region": int(label),
            "mean_error": float(means[label]),
            "area_fraction": float(counts[label] / heatmap.size),
            "centroid_xy": [float(cx[label]), float(cy[label])],
        }
        for label in range(n_labels)
        if counts[label] > 0
    ]
    regions.sort(key=lambda r: r["mean_error"], reverse=True)
    return segments, regions


def _spread_out(regions, k, min_distance):
    """
    Takes the k highest-error regions, skipping any whose centroid sits within
    min_distance of one already taken. The top regions are usually neighbours in
    the same hot area, so labelling them blindly would stack unreadable text in
    one corner and say nothing about the rest of the image.
    """
    chosen = []
    for region in regions:
        x, y = region["centroid_xy"]
        if all(
            (x - cx) ** 2 + (y - cy) ** 2 >= min_distance**2
            for cx, cy in (c["centroid_xy"] for c in chosen)
        ):
            chosen.append(region)
            if len(chosen) == k:
                break
    return chosen


def _save_analysis_figure(x, x_tilde, heatmap, segments, regions, result, out_path):
    input_rgb = np.asarray(tensor_to_pil(x), dtype=float) / 255.0

    # Blend the error map onto the input so error intensity is read at the pixel it
    # belongs to — no mental alignment between a separate image and heatmap panel.
    vmax = float(np.percentile(heatmap, OVERLAY_PERCENTILE))
    if vmax <= 0:
        vmax = max(float(heatmap.max()), 1e-6)
    norm = plt.Normalize(vmin=0.0, vmax=vmax)
    cmap = colormaps[HEATMAP_CMAP]
    wash = cmap(norm(heatmap))[..., :3]
    overlay = input_rgb * (1 - OVERLAY_ALPHA) + wash * OVERLAY_ALPHA

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))

    axes[0].imshow(input_rgb)
    axes[0].set_title("Input")

    axes[1].imshow(tensor_to_pil(x_tilde))
    axes[1].set_title(f"Reconstruction ({result['attributed_ae']})")

    axes[2].imshow(overlay)
    axes[2].set_title(
        f"LPIPS{LPIPS_LAYER_NAME} error overlay\n"
        f"labels = mean error per superpixel region"
    )

    for region in _spread_out(regions, N_LABELED_REGIONS, LABEL_MIN_DISTANCE):
        mask = segments == region["region"]
        axes[2].contour(mask, levels=[0.5], colors="white", linewidths=0.9)
        x_c, y_c = region["centroid_xy"]
        label = axes[2].text(
            x_c,
            y_c,
            f"{region['mean_error']:.3f}",
            color="white",
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
        )
        # Dark stroke keeps the value legible over both the bright and dark ends
        # of the wash, without needing a background box that hides the image.
        label.set_path_effects(
            [path_effects.withStroke(linewidth=2.4, foreground="black")]
        )

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(mappable, ax=axes[2], fraction=0.046, pad=0.04, extend="max")

    for ax in axes:
        ax.axis("off")

    title = (
        f"{result['verdict']}  (ΔMin={result['delta_min']:.4f}, "
        f"threshold={result['threshold']:.4f})"
    )
    if result.get("low_resolution_warning"):
        w, h = result["source_size"]
        title += f"\nLOW CONFIDENCE: source {w}x{h} upscaled to {IMAGE_SIZE}"
    # Reserve the top strip for the suptitle explicitly: the overlay panel's title is
    # two lines tall, and tight_layout alone lets the suptitle sit on top of the
    # shorter single-line titles next to it.
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect whether an image was produced by a latent diffusion model."
    )
    parser.add_argument(
        "images", type=Path, nargs="+", help="Path(s) to image file(s) to analyze."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    aes = load_autoencoders()
    threshold, calibrated = load_threshold()

    for image_path in args.images:
        result = detect(image_path, aes=aes, threshold=threshold, calibrated=calibrated)
        print(f"\n{image_path}")
        for name, d in result["distances"].items():
            marker = " <- min" if name == result["attributed_ae"] else ""
            print(f"  {name}: {d:.4f}{marker}")
        print(f"  Delta_Min = {result['delta_min']:.4f}  Verdict = {result['verdict']}")
        print("  Highest-error regions (mean LPIPS, % of image, centroid x,y):")
        for region in result["region_attribution"][:3]:
            cx, cy = region["centroid_xy"]
            print(
                f"    {region['mean_error']:.4f}  "
                f"{region['area_fraction'] * 100:4.1f}%  "
                f"({cx:.0f}, {cy:.0f})"
            )
        print(f"  Saved: {result['figure']}, {result['report']}")

    if not calibrated:
        print(
            f"\nNote: using uncalibrated default threshold ({DEFAULT_THRESHOLD}). "
            f"Run src/calibrate.py once you have real + generated sample data."
        )


if __name__ == "__main__":
    main()
