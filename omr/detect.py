"""Corner-marker detection that survives print defects.

A cheap printer renders "black" as mid grey with horizontal white streaks / bands.  A streak that crosses
a marker's black border breaks the contour the ArUco detector needs.  Structural remedy: a morphological
*opening* (erode then dilate) removes bright features thinner than its kernel, i.e. the streaks, while the
marker's own white modules (>= one module wide) survive.  The kernel must be thicker than the streaks and no
thicker than a module, so it has to scale with the marker's size in the image:

    module_px  = MARKER_MM / 6 * px_per_mm
    kernel_px  ~ KERNEL_OF_MODULE * module_px

The pixel scale is unknown before a marker is found, so passes are tried over a small geometric ladder of
module-size hypotheses covering what a sheet in view can plausibly be (derived from the frame size; no
magic pixel number), cheapest-first, stopping as soon as a consistent set of markers exists.  If the plain
pass already sees a marker, its size selects the kernel directly.  The opening is square, so it works for
any streak direction (sheet rotated, phone held landscape).

Second-chance decoding (AprilTag-style): ArUco decides each bit with Otsu over the warped candidate, which fails when
a printed "black" cell comes out as a lighter grey than the border (toner dropout over a whole module) or the marker
is low-contrast.  Every candidate ArUco rejected in a pass that was run anyway is re-decoded like AprilTag does
(Olson 2011, Wang & Olson 2016): a black model is fit to the border cells and a white model to a ring in the quiet
zone (each a plane over the marker, so lighting gradients cancel) and each bit is thresholded halfway between them.
Only an exact dictionary match is accepted (no error correction: allowing even the dictionary's 1 correctable bit
produced dozens of wrong ids on the real video), the border must be all black and the ring all white.  It costs
~0.1 ms per candidate and only runs when the plain pass did not already find a full set.

Detections of different passes are merged by id.  An id seen at two different places (in one pass or between
passes) is dropped as inconsistent.  Relaxed bit-error tolerance is NOT used beyond a slightly larger border
tolerance: every sheet still has to pass the geometry check, the meta-strip CRC and the page match.
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import layout as lay
from . import markers as mk

MODULE_MM = lay.MARKER_MM / 6.0          # one marker cell
MAX_SHEET_PPM_OF_FRAME = 1.1 / 210.0     # sheet width fills (at most) the frame's short side, + margin
MIN_PPM = 2.0                            # below this nothing on the sheet is readable anyway
KERNEL_OF_MODULE = 0.85                  # opening kernel / module size: thicker than streaks, thinner than a cell (works for ~0.7..1.0)
LADDER_RATIO = 1.3                       # module-size ratio between ladder steps (a kernel works over a ~1.4x range of sizes)
DECODE_MIN_CONTRAST = 0.25               # white-black model difference / white: below it the "marker" is texture, not print
RING_MODULES = 0.6                       # white model ring: this far outside the border (quiet zone is 3 mm = 1.1 modules)
MIN_FOR_STOP = 3                         # markers of one page group that end the search early
SAME_PLACE = 0.35                        # same id counts as the same detection if centres are within this * marker side

_tls = threading.local()


def _detector() -> "cv2.aruco.ArucoDetector":
    d = getattr(_tls, "det", None)
    if d is None:
        p = cv2.aruco.DetectorParameters()
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.cornerRefinementWinSize = 4
        p.perspectiveRemovePixelPerCell = 8  # more samples per bit: markedly better on blurred / small markers
        p.maxErroneousBitsInBorderRate = 0.5  # a streak through the border breaks a few border bits (default 0.35)
        d = _tls.det = cv2.aruco.ArucoDetector(mk.aruco_dictionary(), p)
    return d


def ladder(shape) -> List[float]:
    """Module-size hypotheses (px), largest first, for a frame of this shape."""
    short = min(shape[:2])
    hi = MODULE_MM * MAX_SHEET_PPM_OF_FRAME * short
    lo = MODULE_MM * MIN_PPM
    out = [hi]
    while out[-1] / LADDER_RATIO >= lo * 1.1:
        out.append(out[-1] / LADDER_RATIO)
    return out


def _kernel(module_px: float, frac: float) -> int:
    return max(3, int(round(frac * module_px)) | 1)   # odd, at least 3


def _side(c: np.ndarray) -> float:
    return float(np.mean([np.linalg.norm(c[(i + 1) % 4] - c[i]) for i in range(4)]))


class _Merge:
    """Detections by id; an id seen in two places is dropped (`conflict`)."""

    def __init__(self):
        self.found: Dict[int, np.ndarray] = {}
        self.conflict: set = set()
        self.dup = False

    def add(self, corners, ids) -> None:
        if ids is None:
            return
        seen_here = set()
        for i, c in zip(ids.ravel(), corners):
            i = int(i)
            if i >= 4 * mk.MAX_PAGES:
                continue
            c = c.reshape(4, 2).astype(np.float64)
            if i in self.conflict:
                continue
            if i in self.found:
                old = self.found[i]
                if np.linalg.norm(old.mean(0) - c.mean(0)) > SAME_PLACE * max(_side(old), _side(c)):
                    self.dup |= i in seen_here
                    self.conflict.add(i)
                    del self.found[i]
                seen_here.add(i)
                continue
            self.found[i] = c
            seen_here.add(i)

    def solid(self) -> bool:
        groups: Dict[int, int] = {}
        for i in self.found:
            groups[i // 4] = groups.get(i // 4, 0) + 1
        return max(groups.values(), default=0) >= MIN_FOR_STOP

    def module_px(self) -> Optional[float]:
        if not self.found:
            return None
        return float(np.median([_side(c) for c in self.found.values()])) / 6.0


# ---------------------------------------------------------------- AprilTag-style re-decoding of rejected candidates
_G = (np.arange(3) - 1) * 0.25                         # 3x3 samples per cell, within +-0.25 module of its centre
_CU, _CV = np.meshgrid(np.arange(6) + 0.5, np.arange(6) + 0.5)
_CELL_UV = np.stack([(_CU[..., None, None] + _G[None, None, None, :]).repeat(3, 2),
                     (_CV[..., None, None] + _G[None, None, :, None]).repeat(3, 3)], -1).reshape(-1, 2)
_T = np.arange(6) + 0.5
_R0, _R1 = -RING_MODULES, 6 + RING_MODULES
_WHITE_UV = np.concatenate([np.stack([_T, np.full(6, _R0)], 1), np.stack([_T, np.full(6, _R1)], 1),
                            np.stack([np.full(6, _R0), _T], 1), np.stack([np.full(6, _R1), _T], 1)])
_BORDER = np.ones((6, 6), bool)
_BORDER[1:5, 1:5] = False
_A_BORDER = np.c_[np.ones(20), _CU[_BORDER], _CV[_BORDER]]
_A_WHITE = np.c_[np.ones(len(_WHITE_UV)), _WHITE_UV]
_A_GRID = np.c_[np.ones(36), _CU.ravel(), _CV.ravel()]
_UNIT = np.float32([[0, 0], [6, 0], [6, 6], [0, 6]])
_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)


def _sample(img: np.ndarray, H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    p = cv2.perspectiveTransform(uv.reshape(-1, 1, 2), H).reshape(-1, 2).astype(np.float32)
    return cv2.remap(img, p[:, :1], p[:, 1:], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1).astype(np.float64)


def model_decode(img: np.ndarray, quad) -> Optional[Tuple[int, np.ndarray]]:
    """Decode a candidate quad (any cyclic order) -> (id, corners TL,TR,BR,BL) or None (see module docstring)."""
    q = np.asarray(quad, np.float64).reshape(4, 2)
    a, b = q[1] - q[0], q[2] - q[1]
    if a[0] * b[1] - a[1] * b[0] < 0:                 # make it clockwise (image y points down)
        q = q[::-1].copy()
    H = cv2.getPerspectiveTransform(_UNIT, q.astype(np.float32))
    cells = _sample(img, H, _CELL_UV).reshape(6, 6, 9).mean(2)
    ring = _sample(img, H, _WHITE_UV)
    bc = np.linalg.lstsq(_A_BORDER, cells[_BORDER], rcond=None)[0]
    wc = np.linalg.lstsq(_A_WHITE, ring, rcond=None)[0]
    blk, wht = (_A_GRID @ bc).reshape(6, 6), (_A_GRID @ wc).reshape(6, 6)
    con = wht - blk
    if con.min() <= 0 or con.mean() < DECODE_MIN_CONTRAST * wht.mean():
        return None
    rel = (cells - blk) / con                            # 0 = black model, 1 = white model
    ring_rel = (ring - _A_WHITE @ bc) / (_A_WHITE @ (wc - bc))
    if (rel[_BORDER] > 0.5).any() or (ring_rel < 0.5).any():
        return None
    ok, mid, rot = mk.aruco_dictionary().identify((rel[1:5, 1:5] > 0.5).astype(np.uint8), 0.0)
    if not ok:
        return None
    return int(mid), np.roll(q, int(rot), axis=0)


def _redecode(m: "_Merge", gray: np.ndarray, imgs, rejected, min_side: float) -> None:
    """Second-chance decode of a pass's rejected candidates (on each of `imgs`); corners refined on `gray`."""
    for r in rejected:
        r = r.reshape(4, 2)
        if _side(r) < min_side:
            continue
        for img in imgs:
            out = model_decode(img, r)
            if out is None:
                continue
            mid, q = out
            if mid in m.found or mid >= 4 * mk.MAX_PAGES:
                break
            win = max(2, int(round(0.3 * _side(q) / 6)))          # 0.3 module, like ArUco's relative window
            q = cv2.cornerSubPix(gray, q.astype(np.float32).reshape(-1, 1, 2), (win, win), (-1, -1), _SUBPIX)
            m.add([q.reshape(1, 4, 2)], np.array([[mid]]))
            break


def find_markers(gray: np.ndarray) -> Tuple[Dict[int, np.ndarray], bool, Dict[str, float]]:
    """-> (id -> 4x2 corners (TL, TR, BR, BL), duplicate/inconsistent id seen, info).

    info: passes (number of detector runs), kernel (px of the pass that completed the set, 0 = plain)."""
    det = _detector()
    m = _Merge()
    corners, ids, rej = det.detectMarkers(gray)
    m.add(corners, ids)
    passes, used = 1, 0
    min_side = lay.MARKER_MM * MIN_PPM                   # smaller quads cannot be a readable marker of the sheet
    if not m.solid():
        _redecode(m, gray, (gray,), rej, min_side)
    if not m.solid():
        steps = ladder(gray.shape)
        hint = m.module_px()
        if hint is not None:
            steps = sorted(steps, key=lambda s: abs(np.log(s / hint)))
            steps = [hint] + steps[:2]   # the marker seen so far gives the scale; neighbours as a fallback
        else:
            steps = steps[1:] + steps[:1] if len(steps) > 2 else steps     # most likely scales first
        done_k = set()
        for s in steps:
            k = _kernel(s, KERNEL_OF_MODULE)
            if k in done_k:
                continue
            done_k.add(k)
            opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
            c2, i2, rej = det.detectMarkers(opened)
            m.add(c2, i2)
            passes += 1
            if not m.solid():
                _redecode(m, gray, (gray, opened), rej, min_side)
            if m.solid():
                used = k
                break
    return m.found, m.dup, {"detect_passes": float(passes), "detect_kernel": float(used)}
