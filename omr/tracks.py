"""Timing tracks: measure where the ticks along the four edges really are and bend the page flat again.

A sheet folded once (two flat halves at an angle), curled or waved is not what the four corner markers' homography
assumes: points between the markers are displaced by several mm.  Sheets printed with header version >= 2 carry rows
of small black squares along the four edges (omr.layout.track_tick_rects).  After the coarse warp from the corner
markers each tick sits near its nominal place; this module

  1. finds every tick (blob response on a thin band along the edge, candidates per tick),
  2. picks the chain of candidates that is smooth along the edge and complete (a small dynamic programme over the
     ticks; it tolerates missing ticks, a hand over part of an edge, header cells / frame lines next to the track),
  3. interpolates the displacement of the four edges into the page (transfinite / Coons patch: exact on all four
     edges, linear across the page between opposite edges - what two flat halves at a crease look like), and
  4. resamples the canvas so that everything (header, bubbles, serial strip, name box) is at its nominal place.

All in canvas pixels of the corner-marker warp (``ppm`` px per mm), displacement d = measured - nominal in mm.
Nothing here decides anything about marks; recognition falls back to the old (quadratic) model when the tracks
cannot be trusted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from . import layout as lay

# ---- tunables
RING_MM = 2.6                        # blob detector: dark box (tick) inside a brighter ring this much larger (total, both sides)
DETECT_SCALE = 0.5                   # ticks are found on a half-resolution copy of the band (sub-pixel peak refinement)
MIN_RESP = 0.22                      # contrast (1 - tick/paper) a blob needs to be a candidate
NMS_MM = 3.0                         # candidates closer than this are one blob
WIN_ALONG_MM = 7.5                   # where a tick may be found relative to its nominal place along the track ...
WIN_ACROSS_MM = {"L": 8.0, "R": 8.0, "T": 5.5, "B": 5.5}   # ... and across it (top: the header rows start 2.4 mm below the ticks)
EXT_ACROSS_MM = {"L": 14.0, "R": 14.0, "T": 7.0, "B": 9.0}  # ticks beyond those windows are still followed from the chain's ends (a hidden corner marker: the
EXT_WIN_MM = 2.2                     # ... extrapolated page runs far off): each next tick is searched this far around the line the last two ticks predict
EXT_DAMP = 0.9                       # ... with the slope damped by this per tick
EXT_MISSES = 2                       # ... until this many ticks in a row are not found
QUAD = 0.08                          # ... plus this per mm squared: one real step (a crease) is paid once, a detour onto a stray blob and back twice
LAMBDA = 0.25                        # chain cost per mm the displacement changes between neighbouring ticks
MISS = 0.3                           # chain cost per tick left out (missing, hidden, not a candidate)
PRIOR = 0.002                        # chain cost per mm of displacement (a whisper: periodic ticks cannot tell a one-tick slip apart)
ANCHOR_MM, ANCHOR_SLOPE = 2.0, 0.12     # a chain end next to a visible corner marker may differ from the marker's displacement by this + slope * distance (mm)
ANCHOR_COST = 1.0                    # chain cost per mm beyond that
ROUGH_MAX_MM = 0.5                   # a track rougher than this (median second difference) is not a row of ticks
MAX_GAP = 12                         # ticks the chain may skip at once
PRESENT_FRAC = 0.3                  # share of ticks on the bottom or top edge that says "this sheet has tracks"
MIN_FRAC = 0.5                       # share of a track's ticks that must be found for the track to count
FLAT_MM = 0.5                        # a field this small everywhere is not worth resampling the canvas for
MAX_DISP_MM = 14.0                   # a larger displacement is not a bent sheet
OUTLIER_MM = 1.2                     # a tick this far from the median of its neighbours' interpolation is dropped


@dataclass
class Track:
    side: str
    nominal: np.ndarray            # (n, 2) mm
    d: np.ndarray                  # (n, 2) displacement mm, NaN where not found
    resp: np.ndarray               # (n,) blob contrast of the chosen candidate (0 = none)

    @property
    def found(self) -> int:
        return int(np.isfinite(self.d[:, 0]).sum())

    @property
    def frac(self) -> float:
        return self.found / max(len(self.nominal), 1)


# =============================================================================== detection
def _response(band: np.ndarray, ppm: float, size_mm):
    """Blob contrast of a dark box of ``size_mm`` (w, h) inside a brighter ring: (ring - box) / ring, float32."""
    iw, ih = max(int(round(size_mm[0] * ppm)), 3), max(int(round(size_mm[1] * ppm)), 3)
    ow, oh = int(round((size_mm[0] + RING_MM) * ppm)), int(round((size_mm[1] + RING_MM) * ppm))
    inner = cv2.boxFilter(band, -1, (iw, ih), normalize=True, borderType=cv2.BORDER_REPLICATE)
    outer = cv2.boxFilter(band, -1, (ow, oh), normalize=True, borderType=cv2.BORDER_REPLICATE)
    a_in, a_out = iw * ih, ow * oh
    ring = (outer * a_out - inner * a_in) / max(a_out - a_in, 1)
    return np.clip((ring - inner) / np.maximum(ring, 24.0), -1.0, 1.5)


def _peaks(resp: np.ndarray, ppm: float):
    """(row, col, value) of the local maxima of ``resp`` above MIN_RESP, with a parabolic sub-pixel position."""
    k = int(round(NMS_MM * ppm)) | 1
    mx = cv2.dilate(resp, np.ones((k, k), np.uint8))
    ys, xs = np.nonzero((resp >= mx) & (resp >= MIN_RESP))
    out = []
    H, W = resp.shape
    for y, x in zip(ys, xs):
        fy, fx = float(y), float(x)
        if 0 < y < H - 1:
            a, b, c = resp[y - 1, x], resp[y, x], resp[y + 1, x]
            fy += 0.5 * (a - c) / (a - 2 * b + c - 1e-9)
        if 0 < x < W - 1:
            a, b, c = resp[y, x - 1], resp[y, x], resp[y, x + 1]
            fx += 0.5 * (a - c) / (a - 2 * b + c - 1e-9)
        out.append((fy, fx, float(resp[y, x])))
    return out


def _anchor_cost(d: np.ndarray, anchor, dist) -> np.ndarray:
    """Chain cost of starting / ending at displacements ``d`` (m, 2) next to a corner marker of displacement ``anchor``."""
    if anchor is None:
        return np.zeros(len(d))
    return ANCHOR_COST * np.maximum(0.0, np.hypot(d[:, 0] - anchor[0], d[:, 1] - anchor[1]) - ANCHOR_MM - ANCHOR_SLOPE * dist)


def _chain(allc: np.ndarray, T: np.ndarray, n: int, a0=None, a1=None, dist0=None, dist1=None):
    """Best chain: one candidate (d_x, d_y, resp) or nothing per tick.  allc: (m, 3) array of (dx, dy, resp), T: the tick
    each belongs to (non-decreasing).

    a0 / a1: displacement of the corner marker next to the first / last tick (None = hidden), dist0 / dist1: its distance
    (mm) from tick 0 / n-1.  Ticks are periodic, so only these anchors (and the ends) tell the true chain from one slipped by a tick."""
    tot = len(allc)
    if tot == 0:
        return {}
    start = np.searchsorted(T, np.arange(n + 1))
    X, Y, R = allc[:, 0], allc[:, 1], allc[:, 2]
    Z = X + 1j * Y
    unary = R - PRIOR * np.abs(Z)
    first = unary - MISS * T - _anchor_cost(allc[:, :2], a0, (dist0 or 0) + T * lay.TRACK_PITCH)   # chain starts here
    S = first.copy()          # best chain score ending at the candidate
    Sg = S + MISS * T         # ... plus the gap cost already counted for its tick (a skip of g ticks costs MISS * (k - j - 1))
    prev = np.full(tot, -1)
    for k in range(n):
        a, b = start[k], start[k + 1]
        p0 = start[max(0, k - MAX_GAP - 1)]
        if a == b or p0 == a:
            continue
        dd = np.abs(Z[a:b, None] - Z[None, p0:a])
        tr = Sg[None, p0:a] - LAMBDA * dd - QUAD * dd * dd
        arg = tr.argmax(1)
        best = tr[np.arange(b - a), arg] + unary[a:b] - MISS * (k - 1)
        take = best > first[a:b]
        S[a:b] = np.where(take, best, first[a:b])
        Sg[a:b] = S[a:b] + MISS * k
        prev[a:b] = np.where(take, p0 + arg, -1)
    ends = S - MISS * (n - 1 - T)
    if a1 is not None:
        ends = ends - _anchor_cost(np.stack([X, Y], 1), a1, (dist1 or 0) + (n - 1 - T) * lay.TRACK_PITCH)
    cur = int(ends.argmax())
    out: Dict[int, tuple] = {}
    while cur >= 0:
        out[int(T[cur])] = (X[cur], Y[cur], R[cur])
        cur = int(prev[cur])
    return out


_ENDS = {"L": ("TL", "BL"), "R": ("TR", "BR"), "T": ("TL", "TR"), "B": ("BL", "BR")}   # corner markers at the first / last tick


def _detect_side(canvas: np.ndarray, ppm: float, side: str, centers: np.ndarray, anchors: Optional[dict] = None) -> Track:
    """Chain of the ticks of one side in the corner-marker warp ``canvas`` (gray float32)."""
    H, W = canvas.shape
    horiz = side in ("T", "B")                         # track runs left-right
    WA = WIN_ACROSS_MM[side]
    if horiz:
        size = (lay.TRACK_LEN - 0.4, lay.TRACK_TB_H - 0.3)
        across0 = centers[0, 1]
        WB = max(WA, EXT_ACROSS_MM[side])
        lo, hi = across0 - WB - 2.5, across0 + WB + 2.5
        r0, r1 = int(max(lo * ppm, 0)), int(min(hi * ppm, H))
        c0, c1 = int(max((centers[0, 0] - WIN_ALONG_MM - 3) * ppm, 0)), int(min((centers[-1, 0] + WIN_ALONG_MM + 3) * ppm, W))
        off = np.array([c0, r0], np.float64)
    else:
        size = (lay.TRACK_LR_W - 0.4, lay.TRACK_LEN)
        across0 = centers[0, 0]
        WB = max(WA, EXT_ACROSS_MM[side])
        lo, hi = across0 - WB - 2.5, across0 + WB + 2.5
        c0, c1 = int(max(lo * ppm, 0)), int(min(hi * ppm, W))
        r0, r1 = int(max((centers[0, 1] - WIN_ALONG_MM - 3) * ppm, 0)), int(min((centers[-1, 1] + WIN_ALONG_MM + 3) * ppm, H))
        off = np.array([c0, r0], np.float64)
    n = len(centers)
    d = np.full((n, 2), np.nan)
    rsp = np.zeros(n)
    if r1 - r0 < 8 or c1 - c0 < 8:
        return Track(side, centers, d, rsp)
    band = cv2.resize(canvas[r0:r1, c0:c1], None, fx=DETECT_SCALE, fy=DETECT_SCALE, interpolation=cv2.INTER_AREA).astype(np.float32)
    peaks = _peaks(_response(band, ppm * DETECT_SCALE, size), ppm * DETECT_SCALE)
    if not peaks:
        return Track(side, centers, d, rsp)
    pts = (np.array([[p[1], p[0]] for p in peaks]) + 0.5) / DETECT_SCALE - 0.5 + off          # canvas px
    pts_mm = pts / ppm
    val = np.array([p[2] for p in peaks])
    ia, ic = (0, 1) if horiz else (1, 0)
    rel = pts_mm[None, :, :] - centers[:, None, :]                  # (tick, peak, 2)
    kk, pp = np.nonzero((np.abs(rel[..., ia]) <= WIN_ALONG_MM) & (np.abs(rel[..., ic]) <= WIN_ACROSS_MM[side]))
    allc = np.column_stack([rel[kk, pp, 0], rel[kk, pp, 1], val[pp]])
    anchors = anchors or {}
    ca, cb = _ENDS[side]
    mc = {c: np.array([r[0] + r[2] / 2, r[1] + r[3] / 2]) for c, r in lay.marker_rects(0).items()}
    a0, a1 = anchors.get(ca), anchors.get(cb)
    for k, c in _chain(allc, kk, n, a0, a1, float(np.linalg.norm(centers[0] - mc[ca])), float(np.linalg.norm(centers[-1] - mc[cb]))).items():
        d[k] = (c[0], c[1])
        rsp[k] = c[2]
    _extend(d, rsp, centers, pts_mm, val)
    _drop_outliers(d, rsp)
    if _roughness(d) > ROUGH_MAX_MM:        # blobs that do not line up (header cells, text): not a track
        d[:] = np.nan
        rsp[:] = 0.0
    return Track(side, centers, d, rsp)


def _roughness(d: np.ndarray) -> float:
    """Median deviation (mm) of each found tick from the line through its two neighbours (one each side) - 0 for a straight
    or smoothly bending track, large for blobs that are not a row of ticks.  A crease shows as a few kinks: the median ignores them."""
    idx = np.nonzero(np.isfinite(d[:, 0]))[0]
    if len(idx) < 5:
        return 0.0
    a, m, b = idx[:-2], idx[1:-1], idx[2:]
    t = ((m - a) / (b - a))[:, None]
    dev = np.linalg.norm(d[m] - ((1 - t) * d[a] + t * d[b]), axis=1) / (1.0 + 0.15 * (b - a - 2))
    return float(np.median(dev))


def _extend(d: np.ndarray, rsp: np.ndarray, centers: np.ndarray, pts_mm: np.ndarray, val: np.ndarray) -> None:
    """Follow the chain beyond its first / last found tick, one tick at a time along the line of the last two."""
    idx = np.nonzero(np.isfinite(d[:, 0]))[0]
    if len(idx) < 3 or not len(pts_mm):
        return
    n = len(d)
    for step, ends in ((-1, idx[:2]), (1, idx[::-1][:2])):
        k0, k1 = int(ends[0]), int(ends[1])                     # last found tick in this direction, the one before it
        cur, slope = d[k0].copy(), (d[k0] - d[k1]) / float(abs(k0 - k1))      # change of d per tick in the walking direction
        misses, k = 0, k0 + step
        while 0 <= k < n and misses < EXT_MISSES:
            cur = cur + slope
            slope = slope * EXT_DAMP
            rel = pts_mm - centers[k] - cur
            near = np.nonzero(np.hypot(rel[:, 0], rel[:, 1]) <= EXT_WIN_MM)[0]
            if len(near):
                j = near[int(np.argmax(val[near]))]
                if not np.isfinite(d[k, 0]) or val[j] > rsp[k]:
                    d[k] = pts_mm[j] - centers[k]
                    rsp[k] = val[j]
                cur = d[k].copy()
                misses = 0
            else:
                misses += 1
            k += step


def _drop_outliers(d: np.ndarray, rsp: np.ndarray) -> None:
    """Remove ticks that disagree with the line through their two nearest found neighbours (one on each side if possible)."""
    for _ in range(6):
        idx = np.nonzero(np.isfinite(d[:, 0]))[0]
        if len(idx) < 4:
            return
        pred = np.empty((len(idx), 2))
        lim = np.empty(len(idx))
        a, m, b = idx[:-2], idx[1:-1], idx[2:]
        t = ((m - a) / (b - a))[:, None]
        pred[1:-1] = (1 - t) * d[a] + t * d[b]
        lim[1:-1] = OUTLIER_MM + 0.15 * (b - a)
        for e, (i0, i1, i2) in ((0, (idx[0], idx[1], idx[2])), (len(idx) - 1, (idx[-1], idx[-2], idx[-3]))):
            pred[e] = d[i1] + (d[i2] - d[i1]) * ((i0 - i1) / (i2 - i1))
            lim[e] = 2.0 * OUTLIER_MM
        bad = np.linalg.norm(d[idx] - pred, axis=1) > lim
        if not bad.any():
            return
        worst = idx[int(np.argmax(np.where(bad, np.linalg.norm(d[idx] - pred, axis=1) / lim, 0)))]
        d[worst] = np.nan                   # one at a time: a bad tick makes its neighbours look bad too
        rsp[worst] = 0.0


def detect(canvas: np.ndarray, ppm: float, anchors: Optional[dict] = None) -> Dict[str, Track]:
    """Find the ticks of all four sides in the corner-marker warp.  anchors: corner name -> displacement (mm) of its visible marker."""
    return {side: _detect_side(canvas, ppm, side, c, anchors) for side, c in lay.track_tick_centers().items()}


# =============================================================================== the field
_XL = lay.TRACK_EDGE_LR + lay.TRACK_LR_W / 2
_XR = lay.PAGE_W - _XL
_YT = lay.TRACK_EDGE_TB + lay.TRACK_TB_H / 2
_YB = lay.PAGE_H - _YT


class Field:
    """Displacement d(x, y) (mm) of the printed page in the canvas, as a Coons patch of the four edges' tick displacements."""

    def __init__(self, tracks: Dict[str, Track]):
        self.tracks = tracks
        self.sides_ok = {s: t.frac >= MIN_FRAC for s, t in tracks.items()}
        end = {}
        for s, t in tracks.items():          # displacement at the first / last found tick of each side
            idx = np.nonzero(np.isfinite(t.d[:, 0]))[0]
            end[s] = (t.d[idx[0]], t.d[idx[-1]]) if len(idx) and self.sides_ok[s] else (None, None)

        def corner(a, b):
            v = [x for x in (a, b) if x is not None]
            return np.mean(v, axis=0) if v else np.zeros(2)
        # L / R run top -> bottom, T / B left -> right
        self.c = {"TL": corner(end["L"][0], end["T"][0]), "TR": corner(end["R"][0], end["T"][1]),
                  "BL": corner(end["L"][1], end["B"][0]), "BR": corner(end["R"][1], end["B"][1])}
        self._fn = {}
        for s, (lo, hi, c_lo, c_hi) in {"L": (_YT, _YB, "TL", "BL"), "R": (_YT, _YB, "TR", "BR"),
                                        "T": (_XL, _XR, "TL", "TR"), "B": (_XL, _XR, "BL", "BR")}.items():
            t = tracks[s]
            idx = np.nonzero(np.isfinite(t.d[:, 0]))[0] if self.sides_ok[s] else np.array([], int)
            ax = 1 if s in ("L", "R") else 0
            u = np.concatenate([[lo], t.nominal[idx, ax], [hi]])
            v = np.vstack([self.c[c_lo][None], t.d[idx], self.c[c_hi][None]])
            self._fn[s] = (u, v)

    def _edge(self, s: str, u: np.ndarray) -> np.ndarray:
        xs, v = self._fn[s]
        return np.stack([np.interp(u, xs, v[:, 0]), np.interp(u, xs, v[:, 1])], -1)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Displacement (..., 2) at nominal page positions (mm)."""
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        s = np.clip((x - _XL) / (_XR - _XL), 0, 1)[..., None]
        t = np.clip((y - _YT) / (_YB - _YT), 0, 1)[..., None]
        xc, yc = np.clip(x, _XL, _XR), np.clip(y, _YT, _YB)
        c = self.c
        d = ((1 - t) * self._edge("T", xc) + t * self._edge("B", xc) + (1 - s) * self._edge("L", yc) + s * self._edge("R", yc)
             - ((1 - s) * (1 - t) * c["TL"] + s * (1 - t) * c["TR"] + (1 - s) * t * c["BL"] + s * t * c["BR"]))
        return d

    # ---- quality
    @property
    def usable(self) -> bool:
        ok = self.sides_ok
        return (ok["L"] and ok["R"]) or (ok["T"] and ok["B"])

    def max_disp(self) -> float:
        xs, ys = np.meshgrid(np.linspace(_XL, _XR, 15), np.linspace(_YT, _YB, 20))
        return float(np.linalg.norm(self(xs, ys), axis=-1).max())

    def info(self) -> dict:
        t = self.tracks
        return {"tracks_found": float(np.mean([t[s].frac for s in t])),
                "tracks_sides": int(sum(self.sides_ok.values())),
                "tracks_max_mm": self.max_disp()}


def measure(canvas: np.ndarray, ppm: float, anchors: Optional[dict] = None) -> Optional[Field]:
    """Field of a sheet's tracks in the corner-marker warp, or None when they cannot be trusted.

    Cheap exit first: a sheet without tracks (older format, not a sheet) shows none on the bottom nor the top edge."""
    cen = lay.track_tick_centers()
    tr = {s: _detect_side(canvas, ppm, s, cen[s], anchors) for s in ("B", "T")}
    if max(t.frac for t in tr.values()) < PRESENT_FRAC:
        return None
    tr.update({s: _detect_side(canvas, ppm, s, cen[s], anchors) for s in ("L", "R")})
    f = Field(tr)
    if not f.usable or f.max_disp() > MAX_DISP_MM:
        return None
    return f


def rectify(canvas: np.ndarray, field: Field, ppm: float, border: float = 255) -> np.ndarray:
    """Canvas resampled so that every nominal page position (mm * ppm) holds what the field says is printed there."""
    H, W = canvas.shape
    ds = max(int(round(1.5 * ppm)), 2)                  # field sampled every ~1.5 mm, bilinear in between (cv2.resize pixel-centre convention)
    nW, nH = -(-W // ds), -(-H // ds)
    px = ((np.arange(nW) + 0.5) * ds - 0.5) / ppm       # mm of the sample points
    py = ((np.arange(nH) + 0.5) * ds - 0.5) / ppm
    xs, ys = np.meshgrid(px, py)
    d = field(xs, ys)
    mx = ((xs + d[..., 0]) * ppm).astype(np.float32)
    my = ((ys + d[..., 1]) * ppm).astype(np.float32)
    mapx = cv2.resize(mx, (nW * ds, nH * ds), interpolation=cv2.INTER_LINEAR)[:H, :W]
    mapy = cv2.resize(my, (nW * ds, nH * ds), interpolation=cv2.INTER_LINEAR)[:H, :W]
    return cv2.remap(canvas, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


FLAT = object()      # "measured, and the sheet is flat": recognize keeps the plain warp
