"""Self-calibrating mark-darkness ("ink") floor.  Pure functions over darkness arrays; nothing here does I/O.

Problem
-------
``omr.recognize`` keeps an *ink floor*: a mark must be at least ``ABS_INK_FRAC`` (0.45) of the printed black to
take part in the page's ink level, and the page ink level (never below the floor) scales the per-pixel
fill ramp.  A fixed 0.45 is a compromise between very light pencil (which it rejects as "weak" and flags) and
erasure ghosts / paper noise (which a lower floor would start to read).  The right value depends on the sheet: how light the
pencil is, how much was erased, how pale the printer's black is.

Method
------
The sheet itself says what a mark is.  Every answer question and every ID / subject column has exactly one mark
or none: the *darkest* bubble of a row ("top-1") is a mark or a blank row's strongest noise, the *second
darkest* ("top-2") is a non-mark almost always (ghost, stray, paper noise; ID / subject rows are the cleanest source).
The ink measure of a bubble is ``bubble_ink`` (85th percentile of its disc darkness, same as ``_ink_level``).

1. ``g`` = a high quantile (90 %) of the top-2 population: how dark non-marks get on this sheet and this light.
2. Candidate floors are scanned over the physical bounds [LO_FRAC, HI_FRAC] x printed black; each candidate is
   scored by the rows that have exactly one bubble above it (what a sheet should look like) minus
   MULTI_PENALTY x rows with two or more above it (a floor that is too low turns ghosts into multi marks).
   Floors below ``g + MARGIN_FRAC x black`` are not allowed (erased ghosts must not be read).
3. The chosen floor is the lowest point of the best-scoring plateau (just above the ghosts): every sheet keeps as
   many light marks as the evidence allows.  (The middle of the plateau was measured to flag pale pencil needlessly.)
4. *Separation* = (25th percentile of the marks' top-1 - g) / black.  Too few marks or separation below
   MIN_SEPARATION: the sheet gives no reliable evidence, the caller keeps today's rule (DEFAULT_FRAC x black) exactly
   (source "default").  Tightening the strength requirement there was measured and does not pay: it flags more
   pencil marks on mixed sheets and removes no silent error, so ``weak`` is reported (calibration) but not acted on.

Bounds (relative to the printed black measured on the same frame by the header guard cells / marker borders)
  LO_FRAC = 0.45  today's fixed rule.  Lowering below it was measured (0.30, 0.38) and rejected: on a very light pencil
                  sheet it flags less but reads erasure ghosts (erased gray 190-232 overlaps very light graphite
                  140-185 in the synthetic bench) and silent errors rise.  Only ``LO_FRAC`` needs changing to allow it.
  HI_FRAC = 0.60  above this a light-pencil mark (graphite 110-150 of 255, 0.45-0.6 of a toner black) is rejected.
The floor is therefore never lower than today's and rises where the sheet's own ghosts demand it.

Fusion
------
The floor is chosen per frame (every frame's fill is computed relative to it) and ``PageRaw.ink`` already carries
the median over the track; add ``ink_source`` / ``ink_separation`` the same way (source: any 'default' frame ->
'default'; separation: median).  Nothing else is needed because fills, not darkness arrays, are what gets fused.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import layout as lay

MARKER_CELLS = 6                 # ArUco 4x4 pattern incl. its black border: 6 x 6 cells
MARKER_BLACK_PERCENTILE = 75     # of the border pixels' darkness: the darkest quartile (robust to white toner streaks)
BUBBLE_PERCENTILE = 85           # per-bubble ink measure: as omr.recognize._ink_level


def bubble_ink(D: np.ndarray) -> np.ndarray:
    """Per-bubble ink measure from an (n_bubbles, n_pixels) darkness array: the 85th percentile of the disc."""
    return np.percentile(D, BUBBLE_PERCENTILE, axis=1).astype(np.float32)


def marker_border_samples(warped: np.ndarray, bg: np.ndarray, bgscale: float, ppm: float, page: int,
                          marker_ids: Sequence[int]) -> Dict[int, np.ndarray]:
    """Darkness (0 = local paper, 1 = black) of the pixels in the central half of every black border cell of each
    detected marker, per marker id.  `bg`/`bgscale`: the paper-level map supplied by the caller (today
    e.g. one built from omr.illum's paper samples); this module never estimates paper level or lighting itself,
    pipeline order is illum -> ink."""
    from . import markers as mk
    out: Dict[int, np.ndarray] = {}
    H, W = warped.shape
    rects = lay.marker_rects(page)
    for corner, (x, y, w, h) in rects.items():
        mid = mk.marker_id(page, corner)
        if mid not in marker_ids:
            continue
        cw, ch = w / MARKER_CELLS, h / MARKER_CELLS
        vals = []
        for i in range(MARKER_CELLS):
            for j in range(MARKER_CELLS):
                if 0 < i < MARKER_CELLS - 1 and 0 < j < MARKER_CELLS - 1:
                    continue
                x0, x1 = (x + (j + 0.25) * cw) * ppm, (x + (j + 0.75) * cw) * ppm
                y0, y1 = (y + (i + 0.25) * ch) * ppm, (y + (i + 0.75) * ch) * ppm
                xs, xe = int(round(x0)), max(int(round(x1)), int(round(x0)) + 1)
                ys, ye = int(round(y0)), max(int(round(y1)), int(round(y0)) + 1)
                if xs < 0 or ys < 0 or xe > W or ye > H:
                    continue
                cell = warped[ys:ye, xs:xe].astype(np.float32)
                b = bg[int(np.clip(round(0.5 * (ys + ye) * bgscale), 0, bg.shape[0] - 1)),
                       int(np.clip(round(0.5 * (xs + xe) * bgscale), 0, bg.shape[1] - 1))]
                vals.append((1.0 - cell / max(float(b), 1.0)).ravel())
        if vals:
            out[mid] = np.concatenate(vals)
    return out


# ---------------------------------------------------------------------------------------------- auto floor
LO_FRAC, HI_FRAC = 0.45, 0.60
DEFAULT_FRAC = 0.45              # today's fixed rule (omr.recognize.ABS_INK_FRAC)
MARGIN_FRAC = 0.06               # floor stays this far (x black) above the ghosts' 90th percentile
GHOST_PCT = 95
MULTI_PENALTY = 3.0
MIN_MARK_ROWS = 5
MIN_SEPARATION = 0.08            # x black
N_CANDIDATES = 31


@dataclass
class AutoInk:
    floor: float                 # darkness floor to use in place of ABS_INK_FRAC * black
    source: str                  # "auto" | "default"
    separation: float            # (marks' 25th percentile - ghosts' 90th percentile) / black; 0 when unknown
    weak: bool                   # True: little evidence (reported in calibration only)


def choose_floor(rows: Sequence[np.ndarray], black: float) -> AutoInk:
    """Per-sheet ink floor from the rows of one page: each row is the ink measures (``bubble_ink``) of the bubbles
    of ONE answer question or ONE ID / subject digit column."""
    default = AutoInk(DEFAULT_FRAC * black, "default", 0.0, True)
    rows = [np.sort(np.asarray(r, np.float32))[::-1] for r in rows if len(r) >= 2]
    if len(rows) < MIN_MARK_ROWS:
        return default
    w = min(len(r) for r in rows)
    M = np.stack([r[:w] for r in rows])                 # (rows, w) descending
    t1, t2 = M[:, 0], M[:, 1]
    lo, hi = LO_FRAC * black, HI_FRAC * black
    g = float(np.percentile(t2, GHOST_PCT))
    floor_min = max(lo, g + MARGIN_FRAC * black)
    if floor_min >= hi:
        return default
    cand = np.linspace(floor_min, hi, N_CANDIDATES)
    n1 = np.array([((M >= t).sum(1) == 1).sum() for t in cand], np.float32)
    n2 = np.array([((M >= t).sum(1) >= 2).sum() for t in cand], np.float32)
    score = n1 - MULTI_PENALTY * n2
    best = score.max()
    plateau = np.nonzero(score >= best - 0.01 * len(rows))[0]
    # the run of the plateau around the best candidate; its lowest point is the floor
    k0 = int(np.argmax(score))
    a = b = k0
    while a - 1 in plateau:
        a -= 1
    while b + 1 in plateau:
        b += 1
    floor = float(cand[a])           # just above the ghosts: the lowest floor that still has the best score
    marks = t1[t1 >= floor]
    if len(marks) < MIN_MARK_ROWS:
        return default
    sep = float((np.percentile(marks, 25) - g) / max(black, 1e-6))
    if sep < MIN_SEPARATION:
        return AutoInk(DEFAULT_FRAC * black, "default", sep, True)
    return AutoInk(floor, "auto", sep, False)


# ---------------------------------------------------------------------------------------------- local ink level
LOCAL_RADII = (4, 8, 16)         # answer rows around a row whose marks set its ink level (grown until LOCAL_MIN_MARKS found)
LOCAL_MIN_MARKS = 3
LOCAL_DEADBAND = 0.15            # the level only goes DOWN, and only when it is this fraction under the page level (a region of lighter marks);
                                 # a darker or similar region keeps the page level (single-instrument sheets stay exactly as they were)
LOCAL_PCT = 75                   # windowed rows: this percentile of the neighbourhood's marks (the median is dragged down by a cluster of erasure ghosts)


def local_levels(row_marks: Sequence[float], floor: float, fallback: float,
                 sections: Sequence[Tuple[Sequence[int], bool]]) -> np.ndarray:
    """Ink level per row from the marks of its own neighbourhood instead of the whole page.

    One page can carry several instruments (pencil ID block, dark pen answers, a dying pen swapped for a fresh one half-way).
    A single page-wide level makes the lighter instrument look "weak" (strength = darkness / ink level).  `row_marks`: per
    row, the ink measure of its darkest bubble (0 when it has no hot bubble).  A row with a mark at or above `floor` (which
    the sheet's own ghosts keep above erasure residue, see `choose_floor`) is judged by that mark's own darkness: a mark
    lighter than its neighbours is a mark, not a "weak" one.  Other rows (blank, ghosts): median of the marks in their
    neighbourhood, clipped below at `floor`; too few marks (fewer than `LOCAL_MIN_MARKS`) in reach: `fallback` (the page level).  `sections`: (row indices in reading order, windowed) -
    windowed (answers): the neighbourhood is +-r rows, r from LOCAL_RADII, the first with enough marks; not windowed
    (the ID + subject block): the whole section is one neighbourhood."""
    rm = np.asarray(row_marks, np.float32)
    out = np.full(len(rm), fallback, np.float32)
    for idx, windowed in sections:
        idx = np.asarray(idx, int)
        if not len(idx):
            continue
        v = rm[idx]
        ok = v >= floor
        for j in range(len(idx)):
            for r in (LOCAL_RADII if windowed else (len(idx),)):
                lo, hi = max(0, j - r), min(len(idx), j + r + 1)
                w = v[lo:hi][ok[lo:hi]]
                if len(w) >= LOCAL_MIN_MARKS:
                    lvl = max(floor, float(np.percentile(w, LOCAL_PCT if windowed else 50)))
                    out[idx[j]] = lvl if lvl < (1.0 - LOCAL_DEADBAND) * fallback else fallback
                    break
    return out
