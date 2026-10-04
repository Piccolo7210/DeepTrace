"""Dependency-free evaluation of the NSG gradient-boosting detector.

Why this exists: the trained detector arrives as a scikit-learn pickle that only
loads under scikit-learn 1.6.1, while the image and speech modules were built
against 1.9.0. Rather than let one module pin the whole project, Video/tools/
export_tree_dump.py re-encodes the trees into a .npz and this module evaluates
them with numpy alone — verified to reproduce scikit-learn's probabilities
exactly (max |dp| = 0, see the exporter's --verify).

The traversal mirrors scikit-learn's `_predict_one_from_raw_data`: at each node
go left when `x[feature] <= num_threshold`, and when the value is NaN follow
`missing_go_to_left`. Summing the leaf values of every tree plus the baseline
gives the raw score; the sigmoid of that is P(AI-generated).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TreeDumpPredictor:
    """Evaluates the exported gradient-boosting trees with numpy only."""

    def __init__(self, dump_path: str | Path):
        self.path = Path(dump_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"missing {self.path} — create it with: python Video/tools/export_tree_dump.py --verify"
            )
        with np.load(self.path, allow_pickle=True) as data:
            self.feature_names = [str(name) for name in data["feature_names"]]
            self.baseline = float(data["baseline"][0])
            self.tree_offsets = data["tree_offsets"]
            self.value = data["value"]
            self.feature_idx = data["feature_idx"]
            self.num_threshold = data["num_threshold"]
            self.missing_go_to_left = data["missing_go_to_left"].astype(bool)
            self.left = data["left"]
            self.right = data["right"]
            self.is_leaf = data["is_leaf"].astype(bool)
            self.tau = float(data["tau_fpr05"][0])
            self.meta = json.loads(str(data["meta"][0]))

    @property
    def n_trees(self) -> int:
        return len(self.tree_offsets) - 1

    def raw_score(self, x: np.ndarray) -> np.ndarray:
        """(n_rows, n_features) float -> (n_rows,) raw (pre-sigmoid) scores."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if x.shape[1] != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {x.shape[1]}")
        total = np.full(x.shape[0], self.baseline, dtype=np.float64)
        for tree in range(self.n_trees):
            base = int(self.tree_offsets[tree])
            node = np.full(x.shape[0], base, dtype=np.int64)          # every row starts at this tree's root
            active = ~self.is_leaf[node]
            while active.any():
                idx = node[active]
                values = x[active, self.feature_idx[idx]]
                go_left = np.where(np.isnan(values), self.missing_go_to_left[idx],
                                   values <= self.num_threshold[idx])
                node[active] = np.where(go_left, base + self.left[idx], base + self.right[idx])
                active = ~self.is_leaf[node]
            total += self.value[node]
        return total

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """(n_rows, n_features) -> (n_rows,) P(class 1 = AI-generated)."""
        raw = self.raw_score(x)
        # expit, written to avoid overflow for large |raw|
        return np.where(raw >= 0, 1.0 / (1.0 + np.exp(-raw)), np.exp(raw) / (1.0 + np.exp(raw)))

    def predict_one(self, stats: dict) -> float:
        """NSG statistics dict -> P(AI-generated). Columns are taken by name, in trained order."""
        missing = [name for name in self.feature_names if name not in stats]
        if missing:
            raise ValueError(f"missing {len(missing)} NSG statistics, e.g. {missing[:4]}")
        row = np.array([[float(stats[name]) for name in self.feature_names]], dtype=np.float64)
        return float(self.predict_proba(row)[0])
