"""One-off: convert the Kaggle detector bundle into a dependency-free tree dump.

The bundle ``nsg_stats_gbm.joblib`` is a pickled scikit-learn
``HistGradientBoostingClassifier`` and only unpickles under the version it was
written with (scikit-learn 1.6.1; 1.9.0 raises ``No module named '_loss'``).
Pinning the whole project to 1.6.1 for one module's pickle would make the video
module dictate versions for the image and speech modules, so instead we export
the trees once and predict them with numpy (see Video/predictor.py).

This is a re-encoding, not a retrain: the same thresholds, the same leaf values,
the same decision function. ``--verify`` proves it by comparing probabilities
against scikit-learn's own ``predict_proba`` on random and real feature rows.

Run (needs scikit-learn 1.6.1 available, once):
    venv312\\Scripts\\python Video\\tools\\export_tree_dump.py --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PATH = MODULE_ROOT / "nsgvd_local" / "models" / "nsg_stats_gbm.joblib"
DUMP_PATH = MODULE_ROOT / "models" / "nsg_gbm_trees.npz"

# Fields of sklearn's predictor node dtype that the raw-data traversal needs.
_NODE_FIELDS = ("value", "feature_idx", "num_threshold", "missing_go_to_left", "left", "right", "is_leaf")


def export(bundle_path: Path = BUNDLE_PATH, dump_path: Path = DUMP_PATH) -> dict:
    import joblib  # imported here so --help works without the heavy deps

    bundle = joblib.load(bundle_path)
    model = bundle["model"]

    if model.n_trees_per_iteration_ != 1:
        raise SystemExit(f"expected a binary classifier, got {model.n_trees_per_iteration_} trees/iteration")
    if list(model.classes_) != [0, 1]:
        raise SystemExit(f"expected classes [0, 1], got {list(model.classes_)}")
    if type(model._loss).__name__ != "HalfBinomialLoss":
        raise SystemExit(f"unexpected loss {type(model._loss).__name__}: sigmoid link may not apply")

    trees = [iteration[0].nodes for iteration in model._predictors]
    if any(nodes["is_categorical"].any() for nodes in trees):
        raise SystemExit("categorical splits present: the numpy traversal below only handles numeric splits")

    # One flat array per field, with an offset table marking each tree's slice.
    offsets = np.cumsum([0] + [len(nodes) for nodes in trees]).astype(np.int64)
    flat = {name: np.concatenate([nodes[name] for nodes in trees]) for name in _NODE_FIELDS}

    payload = {
        "feature_names": np.array(bundle["features"], dtype=object),
        "baseline": np.asarray(model._baseline_prediction, dtype=np.float64).reshape(-1)[:1],
        "tree_offsets": offsets,
        "value": flat["value"].astype(np.float64),
        "feature_idx": flat["feature_idx"].astype(np.int32),
        "num_threshold": flat["num_threshold"].astype(np.float64),
        "missing_go_to_left": flat["missing_go_to_left"].astype(np.uint8),
        "left": flat["left"].astype(np.int32),
        "right": flat["right"].astype(np.int32),
        "is_leaf": flat["is_leaf"].astype(np.uint8),
        "tau_fpr05": np.array([float(bundle["tau_fpr05"])], dtype=np.float64),
        "meta": np.array([json.dumps({
            "source_bundle": bundle_path.name,
            "feature_dir": bundle.get("feature_dir"),
            "n_trees": len(trees),
            "n_nodes": int(offsets[-1]),
            "n_features": len(bundle["features"]),
            "sklearn_version_exported_from": __import__("sklearn").__version__,
            "note": "re-encoding of HistGradientBoostingClassifier for sklearn-free inference",
        })], dtype=object),
    }

    dump_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dump_path, **payload)
    print(f"wrote {dump_path}  ({dump_path.stat().st_size / 1e6:.1f} MB, "
          f"{len(trees)} trees, {int(offsets[-1])} nodes, tau={float(bundle['tau_fpr05']):.4f})")
    return bundle


def verify(bundle: dict, n_random: int = 2000, seed: int = 0) -> None:
    """Compare the numpy predictor against scikit-learn on random and real rows."""
    import pandas as pd

    sys.path.insert(0, str(MODULE_ROOT))
    from predictor import TreeDumpPredictor  # noqa: E402  (needs MODULE_ROOT on the path)

    predictor = TreeDumpPredictor(DUMP_PATH)
    names = bundle["features"]
    if list(predictor.feature_names) != list(names):
        raise SystemExit("feature name/order mismatch between dump and bundle")

    rng = np.random.default_rng(seed)
    rows = [rng.normal(0.0, 5.0, size=len(names)) for _ in range(n_random)]
    rows += [rng.normal(-8.0, 1.0, size=len(names)) for _ in range(200)]   # logden-like magnitudes
    rows.append(np.zeros(len(names)))
    rows.append(np.full(len(names), np.nan))                                # missing-value path

    parity = MODULE_ROOT / "nsgvd_local" / "models" / "parity_samples.npz"
    real_rows = 0
    if parity.exists():                                                     # real NSG statistics
        from nsgvd_local.nsgvd.stats import clip_stats                      # noqa: E402

        data = np.load(parity)
        for score, denom in zip(data["score"], data["denom"]):
            stats = clip_stats(score.astype(np.float32), denom.astype(np.float64))
            rows.append(np.array([stats[name] for name in names], dtype=np.float64))
            real_rows += 1

    x = np.vstack(rows)
    mine = predictor.predict_proba(x)
    theirs = bundle["model"].predict_proba(pd.DataFrame(x, columns=names))[:, 1]
    worst = float(np.max(np.abs(mine - theirs)))
    print(f"verified {len(x)} rows ({real_rows} real NSG clips, 1 all-NaN, 1 all-zero): "
          f"max |dp| = {worst:.3e}")
    if worst > 1e-9:
        raise SystemExit(f"MISMATCH: numpy predictor differs from scikit-learn by {worst:.3e} (max |dp|)")
    print("EXACT MATCH: the tree dump reproduces scikit-learn's probabilities")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="check the dump against scikit-learn")
    args = parser.parse_args()
    bundle = export()
    if args.verify:
        verify(bundle)


if __name__ == "__main__":
    main()
