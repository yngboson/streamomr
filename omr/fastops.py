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


def nanmedian0(a: np.ndarray) -> np.ndarray:
    """`np.nanmedian(a, axis=0)` (all-NaN columns give NaN, no warning).  numpy routes small arrays holding NaNs through
    np.ma.median, which costs ~40 us per call; a sort along the frame axis and the middle one / two values is the same number."""
    a = np.asarray(a)
    if not np.isnan(a).any():
        return np.median(a, axis=0)
    s = np.sort(a, axis=0)                                  # NaN last
    c = (~np.isnan(a)).sum(axis=0)
    lo, hi = np.maximum(c - 1, 0) // 2, c // 2
    x = np.take_along_axis(s, lo[None], axis=0)[0]
    y = np.take_along_axis(s, np.minimum(hi, a.shape[0] - 1)[None], axis=0)[0]
    out = np.where(lo == hi, x, (x + y) / 2).astype(a.dtype)       # even count: mean of the two middle values
    return np.where(c == 0, np.nan, out).astype(a.dtype)


def pct1(a: np.ndarray, q: float) -> np.floating:
    """`np.percentile(a, q)` of an array without NaNs (any shape, flattened; method 'linear'): one partition instead of numpy's ~100 us of
    Python-level quantile machinery per call."""
    a = np.asarray(a).ravel()
    n = a.size
    vi = (n - 1) * np.true_divide(q, 100)
    lo = int(vi)
    hi = lo + 1 if lo + 1 < n else lo
    t = float(vi - lo)
    p = np.partition(a, [lo, hi] if hi != lo else [lo])
    x, y = p[lo], p[hi]
    d = y - x
    out = x + d * t
    if t >= 0.5:
        out = y - d * (1 - t)
    return out


def pct_axis0(a: np.ndarray, q: float) -> np.ndarray:
    """`np.percentile(a, q, axis=0)` of a 2-D array without NaNs."""
    n = a.shape[0]
    vi = (n - 1) * np.true_divide(q, 100)
    lo = int(vi)
    hi = lo + 1 if lo + 1 < n else lo
    t = float(vi - lo)
    p = np.partition(a, [lo, hi] if hi != lo else [lo], axis=0)
    x, y = p[lo], p[hi]
    d = y - x
    out = x + d * t
    if t >= 0.5:
        out = y - d * (1 - t)
    return out
