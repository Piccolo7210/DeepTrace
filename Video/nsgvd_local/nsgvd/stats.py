"""Per-clip NSG statistics used by the gradient-boosting detector.

Must stay byte-for-byte identical in behaviour to the function the detector was trained with
(notebook cell "NSG-statistics detector"), otherwise the classifier sees different features.
"""
from __future__ import annotations

import numpy as np


def clip_stats(s: np.ndarray, d: np.ndarray) -> dict:
    """s: (T,3,H,W) score maps (float32), d: (T,) NSG denominators -> scale-aware NSG statistics."""
    a = np.abs(s)
    e = a.mean(axis=(1, 2, 3))                                   # score energy per frame
    ech = a.mean(axis=(2, 3))                                    # ... per colour channel
    hf = (np.abs(np.diff(s, axis=-1)).mean(axis=(1, 2, 3)) +
          np.abs(np.diff(s, axis=-2)).mean(axis=(1, 2, 3))) / (2 * e + 1e-12)   # spatial roughness
    f = {}
    for t in range(s.shape[0]):
        f[f"logden{t}"] = np.log10(abs(d[t]) + 1e-12)
        f[f"sign{t}"] = float(np.sign(d[t]))
        f[f"logE{t}"] = np.log10(e[t] + 1e-12)
        f[f"hf{t}"] = float(hf[t])
        for c in range(s.shape[1]):
            f[f"logE{t}c{c}"] = np.log10(ech[t, c] + 1e-12)
    for k in ("logden", "logE", "hf"):
        v = np.array([f[f"{k}{t}"] for t in range(s.shape[0])])
        f[f"{k}_mean"], f[f"{k}_std"] = float(v.mean()), float(v.std())
    return f
