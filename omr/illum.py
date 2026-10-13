"""Lighting-robust bubble darkness.

Why this module exists
----------------------
``omr.recognize._paper_level`` estimates the local paper brightness with a 10 mm max filter plus a 3 mm blur.  A
max filter reaches across a hard shadow edge, so an *empty* bubble inside a shadow is divided by the brightness of
the lit paper 5-10 mm away and reads as 30-50 % dark: a false mark ("multi" with two ~1.0 fills, silently wrong).
Soft gradients are fine, step edges (a hand, a phone, a window frame) are not.  The same window also averages the
sheet's printed grey bands into the paper level and cannot undo glare (additive light).

What it does instead
--------------------
1. **Layout-guided paper sampling.**  The layout says exactly where bubbles are, so the nearest *known blank paper*
   to a bubble is the thin annulus just outside its own printed ring (0.25-0.75 mm beyond the ring line, inside the
   gap to the neighbouring bubble).  Eight sectors of that annulus give eight paper samples 2.5-3.5 mm from the
   bubble centre, all around it.  A sample is a robust high percentile of the sector's pixels, so a ring touching it
   or a bit of a neighbour's mark does not matter, and it can never be a mark of the bubble itself (a mark is inside
   the ring).  This is local enough to respect a shadow edge and works under bands (it measures the band, not
   the white outside it).
2. **Edge-aware per-pixel paper level.**  The eight samples are clustered (values within ``CLUSTER_TOL`` of each
   other).  Each disc pixel takes the paper level of the cluster with the largest spatial support *at that pixel*
   (Gaussian weights from the pixel to each sample), so a bubble beside a shadow edge is normalised by the
   paper on its own side only; a gradient or smooth shading is interpolated.  An isolated odd sector (a glyph, a
   streak) is repaired from its two neighbours first.
3. **Doubt is flagged, never guessed.**  If the per-pixel paper level still differs across the disc by more than
   ``EDGE_TOL`` the shadow edge runs through the bubble: its darkness cannot be known, ``reliable`` is False and its
   darkness pixels are returned as 0 (so it neither fakes a mark nor pollutes the per-sheet ink level).  The caller
   must flag the question / digit.  Same for a paper level that is nearly black (no signal).
4. **Local contrast (glare / additive light).**  A multiplicative normalisation cannot undo additive glare: it lifts
   paper and ink alike so ``1 - pix/paper`` shrinks for *every* mark.  The printed rings are a black of known
   strength all over the sheet.  Their darkness, relative to the page's median, is a local contrast factor ``c``
   (smooth field, median over ~25 mm).  Darkness is divided by ``c`` (affine normalisation
   ``(paper - pix) / (paper - local_black)``); where ``c`` falls below ``C_FLAG`` a mark could vanish into the
   glare, so the bubble is flagged instead of trusted.

Contract
--------
``bubble_darkness(smooth, centers_mm, diameter_mm, ppm, method="final")`` returns ``(D, reliable)``:
``D`` is the (n_bubbles, n_disc_pixels) array that ``omr.recognize._bubble_darkness`` returns (0 = paper,
1 = black, same disc pixels), ``reliable`` is a bool vector.  ``omr.ink`` and ``recognize._fill`` consume ``D``
unchanged.  Alternatives B, D, F are kept for the benchmark (``scripts/illum_bench.py``).

The module is pure numpy / OpenCV, no I/O and no state; cost is a few ms per page.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .fastops import pct_rows

INNER_DISC_FRAC = 0.72        # scored disc: same as omr.recognize.INNER_DISC_FRAC
ANN_IN_MM = 0.25              # annulus starts this far beyond the printed ring's centre line radius
ANN_OUT_MM = 0.75             # ... and ends here (the neighbour's ring is >= 1.3 mm - ring width away)
N_SECT = 8                    # paper samples per bubble
SECT_PCT = 97                 # robust high percentile of a sector (rings / glyphs only ever darken it; blur bleeds the ring into
                              # the annulus, and a lower percentile of its few pixels then reads the paper 4-5 % dark)
OUTLIER_TOL = 0.08            # an isolated sector this far from the mean of its neighbours is repaired
PLANE_TOL = 0.07              # samples on one plane within this (relative) = smooth illumination
EDGE_TOL = 0.05               # step regime: paper level may differ this much across the disc, else the edge crosses it
CLUSTER_TOL = 0.06            # sector values within this (relative) belong to the same illumination
SIGMA_MM = 1.3                # spatial support of one sector sample over the disc
PAPER_GAIN = 1.06             # multiplies the paper level: matches the shipped envelope's slight upward bias (1.10 recovers a little ID read-rate but turns ghosts into marks: rejected)
MIN_PAPER = 20.0              # gray levels: darker paper carries no signal
NEIGHBOUR_MM = 9.0            # inked neighbours this close can darken the annulus
HOT_DARKNESS = 0.2            # mean disc darkness (vs the median paper level) above which a bubble counts as inked
HALO_MAX = 1.12               # a mark's own blur halo lifts the paper level by at most this (measured 4-8 % at sigma ~0.4 mm)
HALO_NEIGHBOURS = 4           # nearest un-inked bubbles whose paper level stands in for an inked bubble's own
HALO_REACH_MM = 12.0          # ... within this distance (a gradient or edge further away says nothing about this bubble)
C_REF_PCT = 60                # page reference of the ring darkness: this percentile over all bubbles
C_NEIGHBOURS = 14             # ring-darkness median over this many nearest bubbles
C_APPLY = 0.88                # contrast below this (relative to page) is corrected ...
C_NOISE_SIGMAS = 3.0          # ... and only below this many robust sigmas of the page's own contrast scatter (see below)
C_FLAG = 0.55                 # ... below this the bubble is flagged: marks may vanish in the glare
C_MAX_GAIN = 1.0 / C_FLAG

_cache: Dict[tuple, tuple] = {}


# ------------------------------------------------------------------------------ geometry caches
def _annulus(R_mm: float, ppm: float):
    key = ("ann", round(R_mm, 3), round(ppm, 3))
    if key not in _cache:
        r_out = (R_mm + ANN_OUT_MM) * ppm
        h = int(np.ceil(r_out)) + 1
        y, x = np.mgrid[-h:h + 1, -h:h + 1]
        rr = np.hypot(x, y) / ppm
        m = (rr >= R_mm + ANN_IN_MM) & (rr <= R_mm + ANN_OUT_MM)
        ang = np.mod(np.arctan2(y, x), 2 * np.pi)
        sec = np.floor(ang / (2 * np.pi / N_SECT)).astype(int) % N_SECT
        y, x, sec = y[m], x[m], sec[m]
        groups = [np.nonzero(sec == s)[0] for s in range(N_SECT)]
        _cache[key] = (y, x, groups)
    return _cache[key]


def _ring(R_mm: float, ppm: float):
    """Offsets of the printed ring line (for the contrast measurement)."""
    key = ("ring", round(R_mm, 3), round(ppm, 3))
    if key not in _cache:
        h = int(np.ceil((R_mm + 0.6) * ppm))
        y, x = np.mgrid[-h:h + 1, -h:h + 1]
        rr = np.hypot(x, y) / ppm
        m = np.abs(rr - R_mm) <= 0.22
        _cache[key] = (y[m], x[m])
    return _cache[key]


def _sector_geometry(R_mm: float, disc_dy: np.ndarray, disc_dx: np.ndarray, ppm: float):
    """Normalised weights W[k, p] of sector sample k over disc pixel p."""
    key = ("W", round(R_mm, 3), round(ppm, 3), len(disc_dy))
    if key not in _cache:
        th = (np.arange(N_SECT) + 0.5) * 2 * np.pi / N_SECT
        rs = R_mm + 0.5 * (ANN_IN_MM + ANN_OUT_MM)
        sx, sy = rs * np.cos(th), rs * np.sin(th)
        qx, qy = disc_dx / ppm, disc_dy / ppm
        d2 = (qx[None, :] - sx[:, None]) ** 2 + (qy[None, :] - sy[:, None]) ** 2
        W = np.exp(-d2 / (2 * SIGMA_MM ** 2)).astype(np.float32)
        _cache[key] = (W, sx, sy)
    return _cache[key]


def _gather(img: np.ndarray, px: np.ndarray, py: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    H, W = img.shape
    if len(px) and px.min() + dx.min() >= 0 and px.max() + dx.max() < W and py.min() + dy.min() >= 0 and py.max() + dy.max() < H \
            and img.flags.c_contiguous:
        # every window lies inside the image: one flat take instead of two clipped fancy indexings
        return np.take(img, (dy[None, :] + py[:, None]) * W + (dx[None, :] + px[:, None])).astype(np.float32)
    ys = np.clip(dy[None, :] + py[:, None], 0, H - 1)
    xs = np.clip(dx[None, :] + px[:, None], 0, W - 1)
    return img[ys, xs].astype(np.float32)


# ------------------------------------------------------------------------------ paper samples
def paper_samples(img: np.ndarray, centers_mm, diameter_mm: float, ppm: float):
    """(n, N_SECT) gray level of blank paper in the annulus around every bubble, after repairing isolated outliers."""
    c = np.asarray(centers_mm, np.float64) * ppm
    px = np.round(c[:, 0]).astype(int)
    py = np.round(c[:, 1]).astype(int)
    ay, ax, groups = _annulus(diameter_mm / 2, ppm)
    pix = _gather(img, px, py, ay, ax)
    v = np.stack([pct_rows(pix[:, g], SECT_PCT) for g in groups], axis=1)
    # an isolated odd sector (text glyph, streak, a neighbour's smudge): its two neighbours agree and it does not
    prv, nxt = np.roll(v, 1, axis=1), np.roll(v, -1, axis=1)
    m = 0.5 * (prv + nxt)
    odd = (np.abs(v - m) > OUTLIER_TOL * np.maximum(m, 1.0)) & (np.abs(prv - nxt) < 0.5 * OUTLIER_TOL * np.maximum(m, 1.0))
    return np.where(odd, m, v).astype(np.float32)


def _plane_basis(R_mm: float):
    """Fixed least-squares plane fit through the sector sample positions: (A (K,3), pinv (3,K), sx, sy)."""
    key = ("plane", round(R_mm, 3))
    if key not in _cache:
        th = (np.arange(N_SECT) + 0.5) * 2 * np.pi / N_SECT
        rs = R_mm + 0.5 * (ANN_IN_MM + ANN_OUT_MM)
        sx, sy = rs * np.cos(th), rs * np.sin(th)
        A = np.stack([np.ones(N_SECT), sx, sy], 1)
        _cache[key] = (A, np.linalg.pinv(A), sx, sy)
    return _cache[key]


def _wplane(v, w, A):
    """Weighted least-squares plane through the samples, batched: coef (n, 3)."""
    AtA = np.einsum("nk,ki,kj->nij", w, A, A)
    Atv = np.einsum("nk,ki,nk->ni", w, A, v)
    AtA = AtA + 1e-6 * np.eye(3)[None]
    return np.linalg.solve(AtA, Atv[..., None])[..., 0]


def paper_model(v: np.ndarray, R_mm: float, dy: np.ndarray, dx: np.ndarray, ppm: float, W: np.ndarray,
                explained: Optional[np.ndarray] = None):
    """Paper level of every disc pixel from the K annulus samples -> (P (n, m), consistent (n,), is_step (n,)).

    * smooth illumination (a gradient, a soft penumbra, raking light): the samples lie on a plane within
      ``PLANE_TOL``; the plane is evaluated at each disc pixel, so even a 30 % slope across the bubble is undone.
      Up to two inlier-violating samples are ignored when they are junk, i.e. not an adjacent pair (a text glyph,
      a streak, a neighbour's smudge; an adjacent pair is what a shadow edge cutting the annulus looks like).
    * otherwise there is a step (hard shadow edge) near the bubble: samples are clustered by value, each disc pixel
      takes the cluster with the most spatial support at that pixel.  If different pixels of the disc pick
      clusters with different paper levels the edge runs through the bubble: ``consistent`` is False and the caller
      must flag."""
    n = len(v)
    A, _, _, _ = _plane_basis(R_mm)
    scale = np.maximum(v.mean(1), 1.0)
    w = np.ones_like(v) if explained is None else (~explained).astype(np.float32)
    w[w.sum(1) < 4] = 1.0
    coef = _wplane(v, w, A)
    for _ in range(2):                                            # drop the worst sample while the fit is bad
        resid = np.abs(v - coef @ A.T)
        bad = (resid * w).max(1) > PLANE_TOL * scale
        if not bad.any():
            break
        worst = np.argmax(resid * w, axis=1)
        w[bad, worst[bad]] = 0.0
        coef = _wplane(v, w, A)
    signed = v - coef @ A.T
    resid = np.abs(signed)
    out = (w == 0) & (True if explained is None else ~explained)
    # an adjacent pair of ignored samples is what a shadow edge cutting the annulus looks like - unless both are
    # DARKER than the fit: then the bubble is on the bright side and the dark pair is a neighbour's overflowing mark
    # or printed furniture (it can only darken); a nearby edge then touches the disc by < 0.7 mm at worst.
    adj = (out & np.roll(out, 1, axis=1)).any(1) & ~(~out | (signed < 0)).all(1)
    plane_ok = ((resid * w).max(1) <= PLANE_TOL * scale) & ~adj
    qx, qy = dx / ppm, dy / ppm
    P = coef[:, 0:1] + coef[:, 1:2] * qx[None, :] + coef[:, 2:3] * qy[None, :]
    consistent = np.ones(n, bool)
    step = ~plane_ok
    if step.any():
        vs = v[step]
        ref = np.maximum(vs, 1.0)
        C = (np.abs(vs[:, None, :] - vs[:, :, None]) <= CLUSTER_TOL * ref[:, :, None]).astype(np.float32)   # [b,k,j]
        S = np.matmul(C, W)                                                                                 # support of cluster k at pixel p
        kbest = np.argmax(S, axis=1)                                                                        # (ns, m)
        Cb = C[np.arange(len(vs))[:, None], kbest]                                                          # (ns, m, K)
        Wm = Cb * W.T[None]
        Pc = (Wm * vs[:, None, :]).sum(-1) / np.maximum(Wm.sum(-1), 1e-6)                                   # (ns, m)
        consistent[step] = (Pc.max(1) - Pc.min(1)) <= EDGE_TOL * np.maximum(Pc.max(1), 1.0)
        P[step] = Pc
    return np.maximum(P, 1.0), consistent, step


def explained_by_neighbours(centers_mm, hot: np.ndarray, v: np.ndarray, ref: np.ndarray):
    """(n, K) bool: annulus sectors that face an inked neighbour bubble and are darker than the bubble's typical paper.
    A neighbour's mark overflowing its ring (or its blur) darkens exactly the sectors that look at it; a shadow edge
    does not care where the neighbours' marks are.  Such a sector is not paper: leave it out of the fit."""
    cc = np.asarray(centers_mm, np.float64)
    n = len(cc)
    out = np.zeros((n, N_SECT), bool)
    hi = np.nonzero(hot)[0]
    if hi.size == 0:
        return out
    d = cc[hi][None, :, :] - cc[:, None, :]                        # (n, nh, 2)
    dist = np.hypot(d[..., 0], d[..., 1])
    near = (dist > 1e-6) & (dist < NEIGHBOUR_MM)
    r, c = np.nonzero(near)                                        # only the (bubble, inked neighbour) pairs that are close are worked out
    if r.size == 0:
        return out
    dr = d[r, c]
    dist = dist[r, c]
    ang = np.mod(np.arctan2(dr[:, 1], dr[:, 0]), 2 * np.pi)
    half = np.arcsin(np.clip(3.6 / np.maximum(dist, 3.7), 0, 1))     # angular half-width of a (overflowing) mark
    centre = (np.arange(N_SECT) + 0.5) * 2 * np.pi / N_SECT
    dang = np.abs((ang[:, None] - centre[None, :] + np.pi) % (2 * np.pi) - np.pi)    # (pairs, K)
    hit = (dang <= (half[:, None] + np.pi / N_SECT)).astype(np.uint8)
    starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])             # r is sorted: reduce the pairs of each bubble
    anyhit = np.zeros((n, N_SECT), bool)
    anyhit[r[starts]] = np.bitwise_or.reduceat(hit, starts, axis=0) != 0
    return anyhit & (v < 0.93 * ref[:, None])


def halo_lift(centers_mm, hot: np.ndarray, level: np.ndarray) -> np.ndarray:
    """(n,) factor that undoes a mark's own blur halo in the paper samples of an inked bubble.

    Ink only ever darkens, and blur (a defocused or moving camera, a tilted sheet) spreads a mark past its printed
    ring: all eight annulus samples of an inked bubble are then a few percent too dark, its darkness comes out
    too low, and the mark looks weaker than the page's others (measured: -4..-8 % of paper level at blur, none sharp).
    The un-inked bubbles around it have the real paper level; the lift is their median relative to the bubble's own
    level, never below 1 and never above HALO_MAX (a shadow edge between them must not invent a lot of darkness)."""
    cc = np.asarray(centers_mm, np.float64)
    lift = np.ones(len(cc))
    blank = np.nonzero(~hot)[0]
    hi = np.nonzero(hot)[0]
    if hi.size == 0 or blank.size == 0:
        return lift
    d = np.hypot(*(cc[hi][:, None, :] - cc[blank][None, :, :]).transpose(2, 0, 1))      # (nh, nb)
    d = np.where(d <= HALO_REACH_MM, d, np.inf)
    k = min(HALO_NEIGHBOURS, blank.size)
    nn = np.argpartition(d, k - 1, axis=1)[:, :k]
    ok = np.isfinite(np.take_along_axis(d, nn, axis=1))
    lv = np.where(ok, level[blank][nn], np.nan)
    with np.errstate(all="ignore"):
        nb = np.nanmedian(lv, axis=1)
    lift[hi] = np.clip(np.where(np.isfinite(nb), nb / np.maximum(level[hi], 1.0), 1.0), 1.0, HALO_MAX)
    return lift


# ------------------------------------------------------------------------------ contrast (glare)
def contrast_field(img: np.ndarray, centers_mm, diameter_mm: float, ppm: float, paper: np.ndarray):
    """Local contrast factor c per bubble (1 = like the page's typical bubble ring, < 1 = glare / haze).

    The printed ring line of every bubble is a black of the same strength everywhere on the sheet.  Its darkness
    relative to the bubble's paper level, normalised by the page's reference, is c; the median over the nearest
    bubbles makes it a smooth, outlier-proof field (a marked bubble's ring is darker, an erased one lighter: both
    are a minority among their neighbours)."""
    cc = np.asarray(centers_mm, np.float64)
    n = len(cc)
    c = cc * ppm
    px = np.round(c[:, 0]).astype(int)
    py = np.round(c[:, 1]).astype(int)
    ry, rx = _ring(diameter_mm / 2, ppm)
    pix = _gather(img, px, py, ry, rx)
    dark = 1.0 - pct_rows(pix, 25) / np.maximum(paper, 1.0)     # the dark quartile of the ring line
    ref = float(np.percentile(dark, C_REF_PCT))
    if ref < 0.03:
        return np.ones(n, np.float32), ref
    rel = dark / ref
    k = min(C_NEIGHBOURS, n)
    if n <= k:
        return np.full(n, float(np.median(rel)), np.float32), ref
    d2 = ((cc[:, None, :] - cc[None, :, :]) ** 2).sum(-1)
    nn = np.argpartition(d2, k - 1, axis=1)[:, :k]
    return np.median(rel[nn], axis=1).astype(np.float32), ref


# ------------------------------------------------------------------------------ tiles (user idea D)
def _tile_paper(img: np.ndarray, px: np.ndarray, py: np.ndarray, ppm: float, tile_mm: float, pct: float = 92.0):
    """Paper level from the tile each bubble centre falls into: a high percentile of all pixels of the tile."""
    H, W = img.shape
    t = max(int(round(tile_mm * ppm)), 4)
    ny, nx = (H + t - 1) // t, (W + t - 1) // t
    small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    ts = max(t // 4, 1)
    out = np.zeros((ny, nx), np.float32)
    for j in range(ny):
        for i in range(nx):
            blk = small[j * ts:(j + 1) * ts, i * ts:(i + 1) * ts]
            out[j, i] = np.percentile(blk, pct) if blk.size else 255.0
    return out[np.clip(py // t, 0, ny - 1), np.clip(px // t, 0, nx - 1)]


# ------------------------------------------------------------------------------ entry point
METHODS = ("B", "D10", "D20", "F", "E")        # E is the shipped method ("final" is an alias)


def bubble_darkness(smooth: np.ndarray, centers_mm, diameter_mm: float, ppm: float, method: str = "final",
                    bg: Optional[np.ndarray] = None, bgscale: float = 1.0):
    """Per-pixel darkness of the inner disc of every bubble, 0 = paper, 1 = black -> (D (n, npix), reliable (n,)).

    method: B   per-bubble scalar paper level = median of the 8 annulus samples (layout-guided sampling only)
            D10 / D20  tiles of 10 / 20 mm: tile paper level = high percentile of the tile (user idea, for comparison)
            F   per-pixel plane / cluster paper model + flag where an edge crosses the disc
            E   F + local contrast correction and glare flag (the method to use; "final" is an alias)"""
    from .recognize import _disc
    c = np.asarray(centers_mm, np.float64) * ppm
    px = np.round(c[:, 0]).astype(int)
    py = np.round(c[:, 1]).astype(int)
    n = len(px)
    dy, dx, _ = _disc(INNER_DISC_FRAC * diameter_mm / 2 * ppm)
    pix = _gather(smooth, px, py, dy, dx)
    reliable = np.ones(n, bool)
    if method in ("D10", "D20"):
        P = np.maximum(_tile_paper(smooth, px, py, ppm, 10.0 if method == "D10" else 20.0), 1.0)[:, None]
        return 1.0 - pix / P, reliable
    v = paper_samples(smooth, centers_mm, diameter_mm, ppm)
    if method == "B":
        P = np.maximum(np.median(v, axis=1), 1.0) * PAPER_GAIN
        return 1.0 - pix / P[:, None], reliable
    W, _, _ = _sector_geometry(diameter_mm / 2, dy, dx, ppm)
    ref = np.median(v, axis=1)
    hot = (1.0 - pix.mean(1) / np.maximum(ref, 1.0)) > HOT_DARKNESS
    v = v * halo_lift(centers_mm, hot, ref)[:, None]
    ref = np.median(v, axis=1)
    expl = explained_by_neighbours(centers_mm, hot, v, ref)
    expl |= hot[:, None] & (v < 0.93 * ref[:, None])      # a mark may overflow its own ring into the annulus
    expl[expl.sum(1) > 3] = False                          # more than 3 dark sectors is a shadow edge, not an overflow
    P, consistent, _ = paper_model(v, diameter_mm / 2, dy, dx, ppm, W, expl)
    reliable = consistent & (v.max(1) >= MIN_PAPER)
    P = P * PAPER_GAIN
    D = 1.0 - pix / P
    if method[0] == "E" or method == "final":
        c_flag = int(method[1:]) / 100.0 if method[1:].isdigit() else C_FLAG
        c_loc, _ = contrast_field(smooth, centers_mm, diameter_mm, ppm, np.median(v, axis=1))
        # The ring is a thin line: at low resolution / blur its darkness is a weak signal (0.1 of paper vs 0.26 sharp),
        # so c scatters by 10-20 % with no glare at all.  The correction is one-sided (it only raises darkness), so
        # applying it to that scatter inflates a random fifth of the marks and with them the page's ink level, and
        # every ordinary mark then looks weak by comparison.  Correct only what stands out from the page's own scatter.
        sigma = 1.4826 * float(np.median(np.abs(c_loc - np.median(c_loc))))
        c_apply = min(C_APPLY, float(np.median(c_loc)) - C_NOISE_SIGMAS * sigma)
        gain = np.where(c_loc < c_apply, 1.0 / np.clip(c_loc, C_FLAG, 1.0), 1.0)
        D = D * gain[:, None]
        reliable &= c_loc >= c_flag
    D = np.where(reliable[:, None], D, 0.0).astype(np.float32)
    return D, reliable
