"""Report figure for one AIGVDet result (plan section 9, ``plot_report``).

Three panels: (a) per-pair RGB and flow probabilities along the clip, (b) the crops the
networks saw for the most and least suspicious pair, (c) the clip score against the
threshold with the uncertain band shaded. Matplotlib is forced to the Agg backend so the
function is safe to call from a worker thread.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLOR_RGB = "#1f4e79"
COLOR_FLOW = "#c26a00"
COLOR_FLAGGED = "#b02a2a"
COLOR_CLEAR = "#1e7e34"
COLOR_UNCERTAIN = "#b8860b"
LEVEL_COLORS = {"flagged": COLOR_FLAGGED, "uncertain": COLOR_UNCERTAIN, "clear": COLOR_CLEAR}


def _time_axis(result: dict) -> tuple[np.ndarray, str]:
    """Pair start positions in seconds when the frame rate is known, else frame indices."""
    indices = np.asarray(result["pair_indices"], dtype=float)
    fps = result.get("fps") or 0.0
    if fps > 0:
        return indices / fps, "position in the clip (s) - start of each frame pair"
    return indices, "start frame of each frame pair"


def _plot_pairs(ax, result: dict) -> None:
    x, label = _time_axis(result)
    probs = result["frame_probs"]
    ax.plot(x, probs["spatial"], "o-", color=COLOR_RGB, label="RGB branch")
    ax.plot(x, probs["temporal"], "s-", color=COLOR_FLOW, label="optical-flow branch")
    ax.plot(x, probs["fused"], "-", color="black", linewidth=1.6, alpha=0.7, label="fused")
    ax.axhline(result["threshold"], color="black", linestyle="--", linewidth=1)
    ax.annotate(
        f"threshold {result['threshold']:.3f}", (x[0], result["threshold"]),
        textcoords="offset points", xytext=(2, 3), fontsize=8,
    )  # fmt: skip
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("score per frame pair")
    ax.set_xlabel(label)
    ax.set_title("(a) Per-pair scores along the clip", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper right", ncol=3)
    ax.grid(alpha=0.3)


def _plot_crops(axes, result: dict) -> None:
    crops = result.get("crops")
    titles = [
        ("most_suspicious", "rgb", "most suspicious pair - RGB"),
        ("most_suspicious", "flow", "most suspicious pair - flow"),
        ("least_suspicious", "rgb", "least suspicious pair - RGB"),
        ("least_suspicious", "flow", "least suspicious pair - flow"),
    ]
    for ax, (which, kind, title) in zip(axes, titles):
        ax.set_xticks([])
        ax.set_yticks([])
        if crops and which in crops:
            entry = crops[which]
            ax.imshow(entry[kind])
            detail = f"pair {entry['pair_idx']}, score {entry['score']:.2f}"
            ax.set_title(f"{title}\n({detail})", fontsize=8)
        else:
            ax.set_title(title, fontsize=8)
            ax.text(0.5, 0.5, "crops not kept", ha="center", va="center", fontsize=8,
                    transform=ax.transAxes)  # fmt: skip


def _plot_score(ax, result: dict) -> None:
    score, threshold, band = result["p_fake"], result["threshold"], result["uncertain_band"]
    color = LEVEL_COLORS.get(result["verdict_level"], "black")
    ax.axvspan(threshold - band, threshold + band, color=COLOR_UNCERTAIN, alpha=0.18,
               label=f"uncertain band (+-{band:.2f})")  # fmt: skip
    ax.barh([0], [score], color=color, height=0.45)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("clip score (mean over frame pairs) - compared with a threshold, not a probability")
    ax.set_title(
        f"(c) {result['verdict']} - score {score:.3f}, threshold {threshold:.3f}",
        fontsize=10, loc="left",
    )  # fmt: skip
    ax.legend(fontsize=8, loc="upper right")


def plot_report(result: dict, out_png: str | Path) -> Path:
    """Render the three-panel figure for ``result`` to ``out_png`` and return the path."""
    figure = plt.figure(figsize=(10, 9.6), layout="constrained")
    top, middle, bottom = figure.subfigures(3, 1, height_ratios=[2.2, 2.9, 1.3], hspace=0.02)
    _plot_pairs(top.subplots(), result)
    middle.suptitle("(b) What the networks saw (448x448 centre crops)", fontsize=10, x=0.01,
                    ha="left")  # fmt: skip
    _plot_crops(middle.subplots(1, 4), result)
    _plot_score(bottom.subplots(), result)
    name = Path(result.get("file", "clip")).name
    figure.suptitle(f"{name} - AIGVDet analysis ({result['pairs_used']} frame pairs)", fontsize=11)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_png, dpi=100)
    plt.close(figure)
    return out_png
