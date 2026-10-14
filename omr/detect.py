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

COMPLETE_ROI = 1.3                       # missing-corner search window half-size / predicted marker side
COMPLETE_MAX_OFF = 0.9                   # a found marker must lie within this * side of where the other three predict it
COMPLETE_WIDE = (2.6, 2.2)               # second, wider (window, max offset) when the first finds nothing: on a curled sheet the
                                         # three markers' homography predicts the fourth up to ~1.6 sides off (measured, synth curl)
COMPLETE_KERNELS = (0.85, 0.55, 0.35)    # opening kernels (of a module) tried there: a marker turned ~45 deg has only ~0.7 module of straight white
COMPLETE_SIDE = (0.6, 1.6)               # blob quad side / predicted side

FLAT_BG_FRAC = 1 / 10                    # lighting-flattened last-resort pass: background = max filter of this fraction of the short side ...
FLAT_MIN_CANDIDATES = 4                  # ... and the plain pass rejected at least this many marker-like quads
FLAT_SCALE = 8                           # ... measured on a copy this much smaller
FLAT_DENOISE_SIGMA = 1.2                 # ... blurred, then a light denoise (dim frames carry sensor noise)

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
    d = np.roll(c, -1, axis=0) - c
    return float(np.sqrt((d * d).sum(axis=1)).mean())


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


def _model_fit(img: np.ndarray, quad):
    """Black/white plane models of a candidate quad -> (quad clockwise, rel cells, rel ring) or None when it has no contrast."""
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
    return q, rel, ring_rel


def model_decode(img: np.ndarray, quad) -> Optional[Tuple[int, np.ndarray]]:
    """Decode a candidate quad (any cyclic order) -> (id, corners TL,TR,BR,BL) or None (see module docstring)."""
    fit = _model_fit(img, quad)
    if fit is None:
        return None
    q, rel, _ = fit
    ok, mid, rot = mk.aruco_dictionary().identify((rel[1:5, 1:5] > 0.5).astype(np.uint8), 0.0)
    if not ok:
        return None
    return int(mid), np.roll(q, int(rot), axis=0)


def markerlike(gray: np.ndarray, rejected, min_side: float) -> int:
    """How many of ArUco's rejected candidates look like a marker (black border, white quiet zone, enough contrast) although
    their bits are not in the dictionary.  Texture, text and bubbles do not: none of them -> nothing here to search harder for."""
    return sum(1 for r in rejected if _side(r.reshape(4, 2)) >= min_side and _model_fit(gray, r) is not None)


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


def flatten_lighting(gray: np.ndarray) -> np.ndarray:
    """gray / local paper level, stretched to 0..255: a dim, shadowed or gradient-lit frame gets the contrast of a lit one.

    The background is a max filter wider than any marker followed by a blur (a marker's black border never reaches it),
    computed on a 1/FLAT_SCALE copy (the full-size max filter costs ~250 ms at 1080p).  Only a last resort: a max filter
    spreads a hard shadow edge, which the normal passes handle better."""
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (max(w // FLAT_SCALE, 1), max(h // FLAT_SCALE, 1)), interpolation=cv2.INTER_AREA)
    k = max(5, int(min(small.shape[:2]) * FLAT_BG_FRAC) | 1)
    bg = cv2.GaussianBlur(cv2.dilate(small, np.ones((k, k), np.uint8)), (0, 0), k / 3.0)
    bg = np.maximum(cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR), 8)       # background stays uint8 (the float32 resize/astype cost ~3x);
    flat = cv2.divide(gray, bg, scale=235.0, dtype=cv2.CV_32F)                       # the quotient stays float: uint8 rounding lost a marker in a frame
    flat = cv2.GaussianBlur(flat, (0, 0), FLAT_DENOISE_SIGMA)                        # that the float version read (h_all seed 1005)
    return np.clip(flat, 0, 255, out=flat).astype(np.uint8)


def _blob_quads(img: np.ndarray, pred_side: float, pred_c: np.ndarray, max_off: float = COMPLETE_MAX_OFF):
    """Dark blobs of a local window that are convex quadrilaterals of about the predicted size and place.

    ArUco's candidate search needs a clean closed contour; a blurred, streaked marker may not give one although a plain Otsu
    blob of the same marker is a perfectly good quad (model_decode then reads it).  Only used where a marker is expected."""
    _, b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cs:
        if cv2.contourArea(c) < (COMPLETE_SIDE[0] * pred_side) ** 2 * 0.5:
            continue
        ap = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(ap) != 4 or not cv2.isContourConvex(ap.reshape(-1, 1, 2)):
            continue
        q = ap.astype(np.float64)
        if COMPLETE_SIDE[0] * pred_side <= _side(q) <= COMPLETE_SIDE[1] * pred_side and np.linalg.norm(q.mean(0) - pred_c) < max_off * pred_side:
            yield q


def _complete(m: "_Merge", gray: np.ndarray) -> int:
    """A page with three (or two) markers found: look for the missing ones where the found ones predict them (their layout
    is known; two markers predict through an affine map and are only searched in the wide window).

    The early stop at three markers (plain/opened passes over the whole frame) leaves a visible fourth marker undetected when
    its own contour was broken (streaks, blur, rotation) -> `corner_hidden`.  The search is a small window, so it is cheap
    and uses passes the whole-frame search cannot afford (several kernels, blob quads).  Returns detector runs used."""
    runs = 0
    for _ in range(2):                                 # 2 markers -> 3 -> 4
        groups: Dict[int, List[int]] = {}
        for i in m.found:
            groups.setdefault(i // 4, []).append(i)
        if any(len(v) >= 4 for v in groups.values()):
            break
        cand = [g for g, v in groups.items() if len(v) == 3] or [g for g, v in groups.items() if len(v) == 2]
        if not cand or (len(cand) > 1 and len(groups[cand[0]]) == 2):
            break                                      # two pages with two markers each: nothing to go by
        page = cand[0]
        ids = sorted(groups[page])
        mm = {mk.marker_id(page, c): np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float64)
              for c, (x, y, w, h) in lay.marker_rects(page).items()}
        src, dst = np.concatenate([mm[i] for i in ids]), np.concatenate([m.found[i] for i in ids])
        if len(ids) == 3:
            H, _ = cv2.findHomography(src, dst, 0)
        else:                                          # two squares cannot pin a perspective: an affine map (well posed)
            A = np.linalg.lstsq(np.c_[src, np.ones(len(src))], dst, rcond=None)[0].T
            H = np.vstack([A, [0, 0, 1]])
        if H is None or not np.isfinite(H).all():
            break
        got = False
        for miss in (i for i in range(4 * page, 4 * page + 4) if i not in m.found):
            pq = cv2.perspectiveTransform(mm[miss].reshape(-1, 1, 2), H).reshape(4, 2)
            side, ctr = _side(pq), pq.mean(0)
            if not (np.isfinite(pq).all() and side > 0):
                continue
            stages = ((COMPLETE_ROI, COMPLETE_MAX_OFF), COMPLETE_WIDE) if len(ids) == 3 else (COMPLETE_WIDE,)
            for roi_f, max_off in stages:
                q, r = _search_window(gray, miss, side, ctr, roi_f, max_off)
                runs += r
                if q is not None:
                    m.add([q.reshape(1, 4, 2)], np.array([[miss]]))
                    got = True
                    break
            if got:
                break                                  # re-predict the rest from one more marker
        if not got:
            break
    return runs


def _off_frame(m: "_Merge", shape) -> bool:
    """True when the markers found so far (one page, 1-2 of them) predict every missing marker of that page well outside the
    frame: the sheet is only partly in view, so no further pass (opened ladder, lighting-flattened, windows) can complete it.
    The prediction is an affine map of the found markers, so a generous margin (the wide window + 15% of the distance) is kept."""
    groups: Dict[int, List[int]] = {}
    for i in m.found:
        groups.setdefault(i // 4, []).append(i)
    if len(groups) != 1:
        return False
    page, ids = next(iter(groups.items()))
    if not 1 <= len(ids) <= 2:
        return False
    mm = {mk.marker_id(page, c): np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float64)
          for c, (x, y, w, h) in lay.marker_rects(page).items()}
    src, dst = np.concatenate([mm[i] for i in ids]), np.concatenate([m.found[i] for i in ids])
    A = np.linalg.lstsq(np.c_[src, np.ones(len(src))], dst, rcond=None)[0].T
    if not np.isfinite(A).all():
        return False
    fh, fw = shape[:2]
    here = dst.mean(0)
    for miss in (i for i in range(4 * page, 4 * page + 4) if i not in m.found):
        pq = mm[miss] @ A[:, :2].T + A[:, 2]
        ctr, side = pq.mean(0), _side(pq)
        margin = COMPLETE_WIDE[0] * side + 0.15 * float(np.linalg.norm(ctr - here))
        if -margin <= ctr[0] <= fw + margin and -margin <= ctr[1] <= fh + margin:
            return False
    return True


def _search_window(gray, miss, side, ctr, roi_f, max_off):
    """Look for marker ``miss`` in a window of half-size roi_f * side around ctr -> (corners or None, detector runs)."""
    h, w = gray.shape[:2]
    x0, y0 = np.maximum((ctr - roi_f * side).astype(int), 0)
    x1, y1 = np.minimum((ctr + roi_f * side).astype(int), [w, h])
    if x1 - x0 < side * 0.5 or y1 - y0 < side * 0.5:
        return None, 0                                 # the predicted marker lies outside the frame
    roi, off = gray[y0:y1, x0:x1], np.array([x0, y0], np.float64)
    local = _Merge()
    det, runs = _detector(), 0
    for frac in (0.0,) + COMPLETE_KERNELS:
        img = roi if frac == 0.0 else cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((_kernel(side / 6.0, frac),) * 2, np.uint8))
        c2, i2, rej = det.detectMarkers(img)
        runs += 1
        local.add(c2, i2)
        if miss not in local.found:
            _redecode(local, roi, (roi, img), rej, side * COMPLETE_SIDE[0])
        if miss not in local.found:
            for q in _blob_quads(img, side, ctr - off, max_off):
                out = model_decode(img, q)
                if out is not None and out[0] == miss:
                    local.add([out[1].reshape(1, 4, 2)], np.array([[miss]]))
                    break
        if miss in local.found:
            q = local.found[miss] + off
            if np.linalg.norm(q.mean(0) - ctr) < max_off * side:
                return q, runs
            local.found.pop(miss)
    return None, runs


def find_markers(gray: np.ndarray, stack_seen: bool = False) -> Tuple[Dict[int, np.ndarray], bool, Dict[str, float]]:
    """-> (id -> 4x2 corners (TL, TR, BR, BL), duplicate/inconsistent id seen, info).

    info: passes (number of detector runs), kernel (px of the pass that completed the set, 0 = plain).

    Cost control (a stream spends most frames without a readable sheet, and those used to cost MORE than a read): the speculative
    full-frame passes (opened ladder, lighting-flattened) only run on evidence.  Evidence = a marker was decoded, or ArUco rejected a
    candidate that looks like a marker (`markerlike`).  When the decoded markers predict the rest of their page outside the frame, nothing more is searched;
    with a marker decoded only its own scale is tried (one opened pass, not the whole ladder).

    `stack_seen`: a prior window (detect_prior) already found markers of the sheet in view, so what is missing is hidden (a hand, the sheet being
    lifted away), not damaged: the speculative passes are skipped, only the plain pass and the cheap `_complete` run."""
    det = _detector()
    m = _Merge()
    corners, ids, rej = det.detectMarkers(gray)
    m.add(corners, ids)
    passes, used = 1, 0
    min_side = lay.MARKER_MM * MIN_PPM                   # smaller quads cannot be a readable marker of the sheet
    flat, probed = None, False
    info = lambda: {"detect_passes": float(passes), "detect_kernel": float(used)}
    if not m.solid():
        _redecode(m, gray, (gray,), rej, min_side)
    if not m.solid() and stack_seen:
        passes += _complete(m, gray)
        return m.found, m.dup, info()
    if not m.solid():
        if _off_frame(m, gray.shape):
            return m.found, m.dup, info()
        if not (m.found or markerlike(gray, rej, min_side)):
            # nothing marker-like on the raw frame: one probe on the lighting-flattened copy (a dim / gradient-lit marker has too little
            # contrast to look like one before flattening); an empty scene stops here after two cheap passes
            flat = flatten_lighting(gray)
            c3, i3, rej = det.detectMarkers(flat)
            m.add(c3, i3)
            passes += 1
            if not m.solid():
                _redecode(m, gray, (gray, flat), rej, min_side)
            if not m.solid() and not (m.found or markerlike(flat, rej, min_side)):
                return m.found, m.dup, info()
            probed = True
        steps = ladder(gray.shape)
        hint = None if probed else m.module_px()      # (after the flat probe the scale is not trusted yet)
        if hint is not None:
            steps = sorted(steps, key=lambda s: abs(np.log(s / hint)))
            steps = [hint] + steps[:2]                   # the marker seen so far gives the scale; neighbours as a fallback
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
        if not m.solid() and m.found:                   # last resort on a lighting-flattened copy (dim / shadowed / gradient-lit frame)
            if not probed:
                flat = flatten_lighting(gray)
                c3, i3, rej = det.detectMarkers(flat)
                m.add(c3, i3)
                passes += 1
                if not m.solid():
                    _redecode(m, gray, (gray, flat), rej, min_side)
            if m.found and not m.solid():
                k = _kernel(m.module_px(), KERNEL_OF_MODULE)
                c4, i4, rej = det.detectMarkers(cv2.morphologyEx(flat, cv2.MORPH_OPEN, np.ones((k, k), np.uint8)))
                m.add(c4, i4)
                passes += 1
            if m.solid():
                used = -1
    passes += _complete(m, gray)
    return m.found, m.dup, info()


# ---------------------------------------------------------------- temporal prior (video)
PRIOR_PAD = 1.2          # a prior marker is searched in a window this many marker sides wider than the marker, on every side
PRIOR_MAX_SHIFT = 0.9    # ... and accepted when its centre moved less than this many marker sides since the previous frame


def prior_from(res_corners) -> Optional[Dict[int, np.ndarray]]:
    """The marker corners of a frame result (RecognitionResult.marker_corners: id -> 4x2 list) as a prior for the next frame, or None
    when fewer than MIN_FOR_STOP markers of one page group are known (a half-hidden sheet is not a position to rely on)."""
    if not res_corners:
        return None
    groups: Dict[int, int] = {}
    for i in res_corners:
        groups[i // 4] = groups.get(i // 4, 0) + 1
    g, n = max(groups.items(), key=lambda kv: kv[1])
    if n < MIN_FOR_STOP:
        return None
    return {i: np.asarray(c, np.float64).reshape(4, 2) for i, c in res_corners.items() if i // 4 == g}


def detect_prior(gray: np.ndarray, prior: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """The prior's markers searched only in a small window around where each was in the previous frame (a camera held over a stack of
    sheets sees the next sheet's markers where the last one's were).  A window is ~3-4 marker sides wide, so a detector run costs
    ~1 ms instead of ~15 ms for the frame.  Only the SAME id found near its old place counts."""
    h, w = gray.shape[:2]
    det, out = _detector(), {}
    for mid, q in prior.items():
        side = _side(q)
        ctr = q.mean(0)
        r = (0.5 + PRIOR_PAD) * side
        x0, y0 = np.maximum((ctr - r).astype(int), 0)
        x1, y1 = np.minimum((ctr + r).astype(int) + 1, [w, h])
        if x1 - x0 < side or y1 - y0 < side:
            continue
        corners, ids, _ = det.detectMarkers(gray[y0:y1, x0:x1])
        if ids is None:
            continue
        for c, i in zip(corners, ids.ravel()):
            if int(i) != mid:
                continue
            c = c.reshape(4, 2).astype(np.float64) + (x0, y0)
            if np.linalg.norm(c.mean(0) - ctr) < PRIOR_MAX_SHIFT * side:
                out[mid] = c
                break
    return out


# ---------------------------------------------------------------- resolution-aware detection
DETECT_LONG_SIDE = 1920                  # frames larger than this are searched on an INTER_AREA copy, corners refined at full size


def find_markers_scaled(gray: np.ndarray, prior: Optional[Dict[int, np.ndarray]] = None) -> Tuple[Dict[int, np.ndarray], bool, Dict[str, float]]:
    """find_markers, but on a copy whose long side is DETECT_LONG_SIDE when the frame is larger (24 MP photos);
    the corners are mapped back and sub-pixel refined on the full-size image. Frames up to that size run unchanged.

    `prior` (omr.detect.prior_from of the previous frame): the markers are first looked for in small windows where they were; when they are
    all (>= 3) there, that is the answer; when some are, the plain pass decides whether another sheet is in view and the speculative passes
    (opened ladder, flattening) are skipped - the rest of the sheet is hidden, not damaged."""
    h, w = gray.shape[:2]
    s = DETECT_LONG_SIDE / max(h, w)
    small = gray if s >= 1.0 else cv2.resize(gray, (max(int(round(w * s)), 1), max(int(round(h * s)), 1)), interpolation=cv2.INTER_AREA)
    k = 1.0 if s >= 1.0 else s
    near, stack_seen = {}, False
    if prior:
        near = detect_prior(small, {i: c * k for i, c in prior.items()})
        stack_seen = bool(near)
        if len(near) >= MIN_FOR_STOP:
            m = _Merge()
            m.add([c.reshape(1, 4, 2) for c in near.values()], np.array(list(near)).reshape(-1, 1))
            passes = _complete(m, small)          # a 4th marker the windows did not give (same search as after a full pass)
            found, dup, info = m.found, m.dup, {"detect_passes": float(passes), "detect_kernel": 0.0, "detect_prior": 1.0}
        else:
            found, dup, info = find_markers(small, stack_seen)
    else:
        found, dup, info = find_markers(small)
    if s >= 1.0:
        return found, dup, info
    sx, sy = w / small.shape[1], h / small.shape[0]
    out = {}
    for i, c in found.items():
        p = ((c + 0.5) * (sx, sy) - 0.5).astype(np.float32)      # pixel-centre convention
        win = max(2, int(round(0.3 * _side(p) / 6)))             # 0.3 module, as the detector's own refinement
        p = cv2.cornerSubPix(gray, p.reshape(-1, 1, 2), (win, win), (-1, -1), _SUBPIX).reshape(4, 2)
        out[i] = p.astype(np.float64)
    info["detect_scale"] = float(s)
    return out, dup, info
