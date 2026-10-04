"""Decision-threshold calibration on held-out real videos (plan section 8.4).

The threshold is chosen from *real* videos only: the smallest score that at most
``TARGET_FPR`` of the calibration reals exceed. It is then checked on a disjoint
half of the Drive reals and its fake recall is measured per generator set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_FPR = 0.05
MIN_MEAN_RECALL = 0.60  # plan rule: ship the calibrated threshold only above this fake recall
UNCERTAIN_BAND = 0.10  # verdict is "uncertain" within +-0.10 of the threshold
HF_REAL_POOL = "R1 HF-val"


def threshold_at_fpr(real_scores: np.ndarray | pd.Series, fpr: float = TARGET_FPR) -> float:
    """Smallest threshold such that at most ``fpr`` of ``real_scores`` are strictly above it."""
    return float(np.quantile(np.asarray(real_scores, dtype=float), 1 - fpr, method="higher"))


def rate_above(scores: np.ndarray | pd.Series, threshold: float) -> float:
    """Share of ``scores`` strictly above ``threshold`` (NaN for an empty input)."""
    values = np.asarray(scores, dtype=float)
    return float((values > threshold).mean()) if len(values) else float("nan")


def split_reals(videos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``calib`` = R1 + the Drive pools' calibration halves; ``report`` = their other halves."""
    reals = videos[videos["kind"] == "real"]
    calib = reals[reals["half"].isin(["calib", ""])]
    report = reals[reals["half"] == "report"]
    return calib, report


def bootstrap_threshold(
    real_scores: np.ndarray, fpr: float = TARGET_FPR, rounds: int = 500, seed: int = 0
) -> tuple[float, float]:
    """95% percentile interval of the calibrated threshold (it hinges on a few extreme reals)."""
    rng = np.random.default_rng(seed)
    scores = np.asarray(real_scores, dtype=float)
    draws = [threshold_at_fpr(rng.choice(scores, len(scores)), fpr) for _ in range(rounds)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def recall_by_set(
    videos: pd.DataFrame, thresholds: dict[str, float], score: str = "fused"
) -> pd.DataFrame:
    """Fake recall (share of videos above each threshold) per generator set."""
    fakes = videos[videos["kind"] == "fake"]
    columns = {name: fakes[score] > value for name, value in thresholds.items()}
    return pd.DataFrame(columns).groupby(fakes["set"]).mean()


def fpr_by_pool(
    videos: pd.DataFrame, thresholds: dict[str, float], score: str = "fused"
) -> pd.DataFrame:
    """False-positive rate on the real videos per pool and calibration half."""
    reals = videos[videos["kind"] == "real"].copy()
    reals["half"] = reals["half"].replace("", "all")
    columns = {name: reals[score] > value for name, value in thresholds.items()}
    return pd.DataFrame(columns).groupby([reals["set"], reals["half"]]).mean()


def calibrate(videos: pd.DataFrame, score: str = "fused", fpr: float = TARGET_FPR) -> dict:
    """Pick the shipped threshold following the plan's rule and describe how it behaves."""
    calib, report = split_reals(videos)
    tau_cal = threshold_at_fpr(calib[score], fpr)
    r1 = calib[calib["set"] == HF_REAL_POOL]
    tau_r1 = threshold_at_fpr(r1[score], fpr) if len(r1) else float("nan")
    recall = recall_by_set(videos, {"cal": tau_cal, "r1": tau_r1}, score)
    mean_recall = float(recall["cal"].mean())
    use_calibrated = mean_recall >= MIN_MEAN_RECALL or np.isnan(tau_r1)
    shipped = tau_cal if use_calibrated else tau_r1
    low, high = bootstrap_threshold(calib[score].to_numpy(), fpr)
    return {
        "score": score,
        "target_fpr": fpr,
        "tau_cal": tau_cal,
        "tau_cal_ci95": [low, high],
        "tau_r1": tau_r1,
        "shipped_threshold": float(shipped),
        "shipped_source": "calibrated" if use_calibrated else "r1_only",
        "rule": f"ship calibrated threshold if mean fake recall >= {MIN_MEAN_RECALL}",
        "uncertain_band": UNCERTAIN_BAND,
        "n_calib_reals": len(calib),
        "n_report_reals": len(report),
        "fpr_on_calib": rate_above(calib[score], tau_cal),
        "fpr_on_report": rate_above(report[score], tau_cal),
        "mean_fake_recall_at_cal": mean_recall,
        "fake_recall_by_set": {k: float(v) for k, v in recall["cal"].items()},
    }
