"""Small numpy replacements for hot calls whose Python-level overhead dominates on small arrays (Pyodide / wasm pays it most).
Every function returns exactly what the numpy call it replaces returns (same dtype, same bits)."""
from __future__ import annotations

import numpy as np


def pct_rows(a: np.ndarray, q: float) -> np.ndarray:
    """`np.percentile(a, q, axis=1)` of a 2-D array without NaNs (method 'linear', as numpy computes it for a scalar q)."""
    n = a.shape[1]
    vi = (n - 1) * np.true_divide(q, 100)
    lo = int(vi)
    hi = lo + 1 if lo + 1 < n else lo
    t = float(vi - lo)
    p = np.partition(a, [lo, hi] if hi != lo else [lo], axis=1)
    x, y = p[:, lo], p[:, hi]
    d = y - x
    out = x + d * t
    if t >= 0.5:
        out = y - d * (1 - t)
    return out
