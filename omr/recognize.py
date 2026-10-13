"""One camera image -> structured recognition result. Pure function, no I/O, no state.

    result = recognize(image_bgr, resolve_layout)

Pipeline: ArUco corners -> homography (all 16 corner points) -> canonical warp ->
header decode (the sheet describes its own layout: question count, choices, ID / subject digits, pages, page;
CRC-16) -> layout -> bubble scoring relative to the local paper level, decided per question against page-level
calibration.  No database is needed.  Legacy sheets (single-row meta strip with a 16-bit form id) are still read
when a `resolve_layout(form_id)` callable is given.

Design rule: anything doubtful is flagged or left unread, never guessed.  A wrong student
ID is worse than none, so ID digits use stricter rules than answers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import illum
from . import ink as inkmod
from . import layout as lay
from . import markers as mk
from . import tracks as trk
from .detect import find_markers, find_markers_scaled

DEFAULT_PX_PER_MM = 6.0

# ---- statuses
OK, NO_SHEET, PARTIAL, INCONSISTENT = "ok", "no_sheet", "partial", "inconsistent"
UNKNOWN_FORM, BAD_META, BAD_GEOMETRY = "unknown_form", "bad_meta", "bad_geometry"

# ---- tunables (all relative / physical units, none depends on a particular ink colour)
MIN_MARKERS = 3              # corner markers of one page needed (3 of 4 give 12 points for the homography)
FOREIGN_MIN_SIDE_FRAC = 0.6   # a marker of another page group this much smaller than a corner marker (measured on this sheet's plane) is junk
FOREIGN_MARGIN_MM = 5.0      # markers of another page group must lie this far outside the sheet
MAX_EDGE_RATIO = 3.0         # opposite edges of the projected page may differ at most by this factor (extreme perspective)
MAX_REPROJ_MM = 2.0          # marker corner reprojection error above this -> bad_geometry (paper bow moves the corners)
MAX_REPROJ_TRACKS_MM = 8.0   # ... unless the sheet has timing tracks that straighten it (omr.tracks): then this much
META_MIN_CONTRAST = 0.25     # 1 - black/white of the meta guard cells
HOT_MEAN_DARKNESS = 0.15     # a bubble whose disc is on average this much darker than paper is "not empty"
INK_PERCENTILE = 50
BLACK_PERCENTILE = 25        # printed-black reference: darker quartile of the black guard cells (robust to white streaks)
ABS_INK_FRAC = 0.45          # absolute ink floor = this fraction of the printed-black darkness (meta strip) on the same frame
MIN_CALIB_MARKS = 5          # confident marks needed before the page's own marks may raise the ink level
STRENGTH_UNCALIBRATED = 0.7  # sparse page: only the absolute scale is trusted, so lean towards "ambiguous"
DEFAULT_TYP_FILL = 0.9
T_HI_MIN, T_HI_MAX, T_HI_OF_TYP = 0.45, 0.65, 0.65
ALIGN_PX_PER_MM = 4.0         # ring correlation resolution
ALIGN_SAMPLES_PER_GROUP = 40
ALIGN_MIN_CORR = 0.35
ALIGN_SEARCH_MM = 1.4
ALIGN_REFINE_STAGES = 2        # extra ring searches around the model's prediction when a marker is hidden
ALIGN_MIN_COVERAGE = 0.8       # ... and the share of sampled rings that must confirm the model then
ALIGN_MAX_SHIFT_EXTRAP_MM = 4.0
RING_SCORE_SIGMA_MM, RING_SCORE_MIN, RING_SCORE_FRAC = 0.5, 0.25, 0.5   # per-row ring check on a straightened page (_rows_misfit)
TRACKS_MIN_RING_CORR, TRACKS_MIN_INLIERS = 0.4, 0.55   # a page straightened by its timing tracks: rings must confirm it (median correlation / inlier share)
ALIGN_MAX_SHIFT_MM = 2.0     # a larger correction means something is wrong: fall back to nominal positions
MIN_STRENGTH = 0.6          # mean ink darkness relative to the page's ink level
T_LO_FRAC = 0.2             # t_lo = T_LO_FRAC * t_hi
DIGIT_MARGIN = 0.08         # strict digits (ID, subject) need this much more fill than t_hi when the mark is dark ...
REL_MARGIN = 0.15           # relative rule: a choice whose disc darkness is this much (absolute, 0..1) above the median of the
                            # question's other choices counts as marked even when it is too faint for the absolute fill thresholds
REL_SURE = 0.20             # ... read silently only from this excess on (answers); between REL_MARGIN and this the row is flagged
REL_SURE_DIGIT = 0.20       # ... strict digit columns (ID, subject): a digit is read from a relative-only mark from this excess on
DIGIT_MARGIN_FROM, DIGIT_MARGIN_FULL = 0.7, 1.0   # ... scaled in by strength between these (measured: half-filled bubbles of strength < 0.8
                                                  # never reached t_hi, 0.8-1.0 reached t_hi + 0.02, above 1.0 t_hi + 0.07)


SOFT_FLAGS = ("serial_unreadable", "corner_hidden", "guessed_answers", "marker_dropped")   # informational: never a reason for human review
SERIAL_ABSENT_RATIO = 0.65            # darkest/brightest serial cell above this = nothing printed


def hard_flags(flags) -> list:
    return [f for f in flags or [] if f not in SOFT_FLAGS]


# =============================================================================== results
@dataclass
class QuestionResult:
    number: int
    state: str                 # "single" | "blank" | "multi"
    marked: List[int]          # choice indices (0-based) judged marked
    labels: List[str]
    confidence: float          # 0..1
    ambiguous: bool            # a close call: a human should look
    scores: List[float]        # darkness per choice (0 = paper, 1 = black)
    guess: Optional[int] = None  # best-guess choice of an UNCERTAIN row (always together with ambiguous); None = confident / no choice stands out
    guessed: bool = False      # the row's value is a best guess (== guess is not None); a human should look

    @property
    def answer(self) -> Optional[int]:
        """The row's value: the single mark, or - for an uncertain blank row - the best guess.
        A clearly blank row and a multi-mark row (its answer is the whole `marked` set) stay None."""
        if self.state == "single":
            return self.marked[0]
        return self.guess


@dataclass
class DigitResult:
    index: int
    state: str
    value: Optional[int]       # the digit; for an uncertain column the best guess (guessed=True); None = clearly blank / nothing stands out
    confidence: float
    ambiguous: bool
    scores: List[float]
    guessed: bool = False      # value is a best guess, not a confident read (always ambiguous)


@dataclass
class RecognitionResult:
    status: str
    form_key: Optional[str] = None         # identity of the printed form (omr.markers.Header.form_key)
    form_id: Optional[int] = None          # legacy sheets only
    page: Optional[int] = None             # 0-based
    total_pages: Optional[int] = None
    serial: Optional[int] = None           # printed per-copy number; None = none printed / unreadable (flag "serial_unreadable")
    answers: List[QuestionResult] = field(default_factory=list)
    student_id: Optional[str] = None
    id_digits: List[DigitResult] = field(default_factory=list)
    subject: Optional[str] = None          # marked subject code; "" = the form has none; None = not readable
    subject_digits: List[DigitResult] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    marker_ids: List[int] = field(default_factory=list)
    marker_corners: Dict[int, List[List[float]]] = field(default_factory=dict)  # id -> 4x(x,y) in the source image
    missing_corners: List[str] = field(default_factory=list)
    homography: Optional[List[List[float]]] = None   # source image px -> canvas px (canvas px = mm * px_per_mm)
    px_per_mm: float = DEFAULT_PX_PER_MM
    quality: Dict[str, float] = field(default_factory=dict)
    calibration: Dict[str, object] = field(default_factory=dict)
    # alignment-corrected bubble centres in page mm: "answers" [question][choice] -> (x, y), "id" [digit][value] -> (x, y)
    bubble_centers_mm: Dict[str, list] = field(default_factory=dict)
    timing_ms: Dict[str, float] = field(default_factory=dict)
    warped: Optional[np.ndarray] = None    # canonical gray canvas (flattened by the timing tracks when the sheet has them)
    name_crop: Optional[np.ndarray] = None  # gray crop of the name box (from the canonical canvas)
    raw: Optional["PageRaw"] = None        # per-bubble measurements (for multi-frame fusion)
    fused_frames: int = 0                  # > 0: decisions fused over this many frames of one track (omr.fusion)
    dewarp: Optional[object] = None        # omr.tracks.Field: page mm -> displacement in the homography canvas (None = flat sheet)

    @property
    def id_guessed(self) -> bool:
        return "id_guessed" in self.flags

    @property
    def subject_guessed(self) -> bool:
        return "subject_guessed" in self.flags

    @property
    def needs_review(self) -> bool:
        return (self.status == OK and (self.student_id is None or self.subject is None or bool(hard_flags(self.flags))
                or any(a.ambiguous or a.guessed or a.state != "single" for a in self.answers)))

    def image_points(self, pts_mm) -> np.ndarray:
        """Map layout mm coordinates (page space) to source-image pixels (for overlays)."""
        if self.homography is None:
            raise ValueError("no homography")
        p = np.asarray(pts_mm, np.float64).reshape(-1, 2)
        if self.dewarp is not None:          # canvas position of a point of the flattened page
            p = p + self.dewarp(p[:, 0], p[:, 1])
        p = p * self.px_per_mm
        Hi = np.linalg.inv(np.asarray(self.homography))
        return cv2.perspectiveTransform(p.reshape(-1, 1, 2), Hi).reshape(-1, 2)

    def to_dict(self) -> dict:
        """JSON-serialisable view (no image arrays)."""
        def conv(v):
            if isinstance(v, np.generic):
                return v.item()
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, dict):
                return {str(k): conv(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [conv(x) for x in v]
            if hasattr(v, "__dataclass_fields__"):
                return {k: conv(getattr(v, k)) for k in v.__dataclass_fields__}
            return v
        d = {k: conv(getattr(self, k)) for k in self.__dataclass_fields__ if k not in ("warped", "name_crop", "raw", "dewarp")}
        d["needs_review"] = self.needs_review
        d["id_guessed"], d["subject_guessed"] = self.id_guessed, self.subject_guessed
        d["guessed_answers"] = sum(1 for a in self.answers if a.guessed)
        return d


# =============================================================================== helpers
_disc_cache: Dict[Tuple[int, float], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _disc(radius_px: float):
    """(dy, dx, mask) index offsets for a disc of the given radius."""
    key = (0, round(radius_px, 2))
    if key not in _disc_cache:
        r = int(np.ceil(radius_px))
        y, x = np.mgrid[-r:r + 1, -r:r + 1]
        m = (x * x + y * y) <= radius_px * radius_px
        _disc_cache[key] = (y[m], x[m], m)
    return _disc_cache[key]


def _marker_corners_mm(page: int) -> Dict[int, np.ndarray]:
    out = {}
    for corner, (x, y, w, h) in lay.marker_rects(page).items():
        # ArUco returns corners in the marker's own TL, TR, BR, BL order
        out[mk.marker_id(page, corner)] = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float64)
    return out


def _quality_sharpness(gray: np.ndarray, quad: np.ndarray) -> float:
    x0, y0 = np.maximum(quad.min(0).astype(int), 0)
    x1, y1 = quad.max(0).astype(int)
    roi = gray[y0:y1, x0:x1]
    if roi.size < 100:
        return 0.0
    if max(roi.shape) > 700:  # keep cost bounded & comparable
        roi = cv2.resize(roi, None, fx=700 / max(roi.shape), fy=700 / max(roi.shape), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(roi, cv2.CV_32F).var())


def _plausible_sheet(Hinv: np.ndarray, ppm: float) -> bool:
    """The page rectangle must project to a convex quadrilateral with sane perspective."""
    q = cv2.perspectiveTransform(np.array([[[0, 0], [lay.PAGE_W * ppm, 0], [lay.PAGE_W * ppm, lay.PAGE_H * ppm],
                                            [0, lay.PAGE_H * ppm]]], np.float64), Hinv)[0]
    cr = [float((q[(i + 1) % 4] - q[i])[0] * (q[(i + 2) % 4] - q[(i + 1) % 4])[1] - (q[(i + 1) % 4] - q[i])[1] * (q[(i + 2) % 4] - q[(i + 1) % 4])[0]) for i in range(4)]
    if not (all(c > 0 for c in cr) or all(c < 0 for c in cr)):
        return False
    e = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    return min(e) > 0 and max(e[0], e[2]) / min(e[0], e[2]) < MAX_EDGE_RATIO and max(e[1], e[3]) / min(e[1], e[3]) < MAX_EDGE_RATIO


# =============================================================================== meta strip
def _read_meta(warped: np.ndarray, ppm: float):
    """Decode the 40-cell strip with guard-cell black/white references interpolated along the strip.

    Returns (Meta | None, info dict)."""
    rects = lay.meta_cell_rects()
    means = np.empty(len(rects))
    for i, (x, y, w, h) in enumerate(rects):
        x0, x1 = (x + 0.2 * w) * ppm, (x + 0.8 * w) * ppm
        y0, y1 = (y + 0.2 * h) * ppm, (y + 0.8 * h) * ppm
        means[i] = warped[int(round(y0)):max(int(round(y1)), int(round(y0)) + 1),
                          int(round(x0)):max(int(round(x1)), int(round(x0)) + 1)].mean()
    n = len(means)
    sg = [i for i, b in enumerate(mk.START_GUARD)]
    eg = [n - len(mk.STOP_GUARD) + i for i in range(len(mk.STOP_GUARD))]
    black_l = np.mean([means[i] for i in sg if mk.START_GUARD[i]])
    white_l = np.mean([means[i] for i in sg if not mk.START_GUARD[i]])
    black_r = np.mean([means[j] for i, j in enumerate(eg) if mk.STOP_GUARD[i]])
    white_r = np.mean([means[j] for i, j in enumerate(eg) if not mk.STOP_GUARD[i]])
    t = (np.arange(n) - 1.5) / (n - 4.0)
    black = black_l + (black_r - black_l) * t
    white = white_l + (white_r - white_l) * t
    contrast = 1.0 - min(black_l / max(white_l, 1), black_r / max(white_r, 1))
    info = {"meta_contrast": float(contrast)}
    if contrast < META_MIN_CONTRAST or min(white_l, white_r) <= min(black_l, black_r):
        return None, info
    info["meta_black"] = _printed_black(warped, ppm, rects, white_l, white_r)
    bits = means < 0.5 * (black + white)
    margin = np.abs(means - 0.5 * (black + white)) / np.maximum(white - black, 1)
    info["meta_min_margin"] = float(margin.min())
    return mk.decode_meta(bits), info


def _printed_black(warped, ppm, rects, white_l: float, white_r: float) -> float:
    """Darkness (0 = paper, 1 = black) of the printed black on this frame, measured on the meta strip's black guard cells.

    A cheap printer renders black as grey and drops thin lines of toner (white streaks): the cell *mean*
    then mixes ink and paper.  The darker quartile of the cell's pixels is the ink itself, independent of how
    many streaks fall into the cell; for a solid print it equals the mean."""
    vals = []
    paper = 0.5 * (white_l + white_r)
    for i, black in enumerate(mk.START_GUARD):
        if black:
            vals.append(_cell_quartile(warped, ppm, rects[i]))
    n = len(rects)
    for j, black in enumerate(mk.STOP_GUARD):
        if black:
            vals.append(_cell_quartile(warped, ppm, rects[n - len(mk.STOP_GUARD) + j]))
    return float(np.clip(1.0 - np.mean(vals) / max(paper, 1.0), 0.0, 1.0))


def _cell_quartile(warped, ppm, rect) -> float:
    x, y, w, h = rect
    c = warped[int(round((y + 0.2 * h) * ppm)):max(int(round((y + 0.8 * h) * ppm)), int(round((y + 0.2 * h) * ppm)) + 1),
               int(round((x + 0.2 * w) * ppm)):max(int(round((x + 0.8 * w) * ppm)), int(round((x + 0.2 * w) * ppm)) + 1)]
    return float(np.percentile(c, BLACK_PERCENTILE))


def _read_serial(warped: np.ndarray, ppm: float):
    """Serial strip -> (serial or None, state) with state in "ok" | "absent" | "unreadable".

    "absent": nothing printed (serial 0 prints nothing) *or nothing readable there* (a hand / shadow over the
    footer): no guard-cell pattern with printed contrast on either side.  Only a strip that visibly is there
    (clean start or stop guard) and still fails to decode is "unreadable" and flagged.
    Black/white references come from the strip's own guard cells, interpolated like the meta strip."""
    rects = lay.serial_cell_rects()
    means = np.empty(len(rects))
    for i, (x, y, w, h) in enumerate(rects):
        x0, x1 = (x + 0.2 * w) * ppm, (x + 0.8 * w) * ppm
        y0, y1 = (y + 0.2 * h) * ppm, (y + 0.8 * h) * ppm
        means[i] = warped[int(round(y0)):max(int(round(y1)), int(round(y0)) + 1),
                          int(round(x0)):max(int(round(x1)), int(round(x0)) + 1)].mean()
    if means.min() >= SERIAL_ABSENT_RATIO * means.max():
        return None, "absent"
    n = len(means)
    sg, eg = mk.START_GUARD, mk.STOP_GUARD
    st = list(range(len(sg)))
    en = [n - len(eg) + i for i in range(len(eg))]
    black_l = np.mean([means[i] for i in st if sg[i]])
    white_l = np.mean([means[i] for i in st if not sg[i]])
    black_r = np.mean([means[j] for i, j in enumerate(en) if eg[i]])
    white_r = np.mean([means[j] for i, j in enumerate(en) if not eg[i]])
    ok_l = 1.0 - black_l / max(white_l, 1) >= META_MIN_CONTRAST and white_l > black_l
    ok_r = 1.0 - black_r / max(white_r, 1) >= META_MIN_CONTRAST and white_r > black_r
    if not (ok_l or ok_r):
        return None, "absent"
    if not ok_l:
        black_l, white_l = black_r, white_r
    if not ok_r:
        black_r, white_r = black_l, white_l
    t = (np.arange(n) - 1.5) / (n - 4.0)
    black = black_l + (black_r - black_l) * t
    white = white_l + (white_r - white_l) * t
    v = mk.decode_serial(means < 0.5 * (black + white))
    return (v, "ok") if v is not None else (None, "unreadable")


# =============================================================================== self-describing header
HEADER_SEARCH_MM = 2.0       # local registration window of a header row (paper bow / curl near the top edge); < half a row pitch
HEADER_BAND_MM = 1.0         # white paper band above / below a row used as that row's local paper level
HEADER_WIDE_SEARCH_MM = (4.0, 2.6)   # (dx, dy) retry window when the sheet is warped from only 3 corner markers or the first pass failed: the
                                     # extrapolated corner (paper curl / bow) moves the far end of the header 2-3 mm; dy stays < half a row pitch
HEADER_WIDE_DISAGREE_MM = (3.5, 2.8) # ... and the two ends may then differ this much (dx, dy)
TRACK_SEG, TRACK_HOP = 8, 4          # tracked retry: windows of 8 cells, every 4 cells
TRACK_STEP_MM = 1.2                  # ... each searched this far around the previous window's offset
TRACK_FIRST_MM = 3.0                 # ... the first one this far around the nominal position (< half the row pitch)
TRACK_MIN_CONTRAST = 40.0            # flat-field gray levels (paper = 235): bright bands minus dark band needed to trust a window
HEADER_FLIP_CANDIDATES = 3   # least-certain data cells that may be flipped (singly or in pairs) to satisfy the CRC ...
HEADER_FLIP_MARGIN = 0.3     # ... only when they are this close to the threshold (fraction of the black-white contrast)


def _box_mean(ii, x0, y0, x1, y1) -> float:
    """Mean of the canvas over [x0, x1) x [y0, y1) (px, clipped) from an integral image."""
    H, W = ii.shape[0] - 1, ii.shape[1] - 1
    x0, x1 = min(max(round(x0), 0), W - 1), min(max(round(x1), 1), W)      # plain ints: np.clip per call dominated the header read
    y0, y1 = min(max(round(y0), 0), H - 1), min(max(round(y1), 1), H)
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    return float((ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]) / ((x1 - x0) * (y1 - y0)))


def _box_means(ii, x0, y0, x1, y1) -> np.ndarray:
    """`_box_mean` over arrays of boxes at once (same clipping / rounding, so the same numbers)."""
    H, W = ii.shape[0] - 1, ii.shape[1] - 1
    x0 = np.minimum(np.maximum(np.rint(x0), 0), W - 1).astype(np.intp)
    x1 = np.minimum(np.maximum(np.rint(x1), 1), W).astype(np.intp)
    y0 = np.minimum(np.maximum(np.rint(y0), 0), H - 1).astype(np.intp)
    y1 = np.minimum(np.maximum(np.rint(y1), 1), H).astype(np.intp)
    x1, y1 = np.maximum(x1, x0 + 1), np.maximum(y1, y0 + 1)
    return (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]) / ((x1 - x0) * (y1 - y0))


def _band_scores(ii, ppm, xs0, xs1, y, h, dys) -> np.ndarray:
    """bright band - dark band - bright band score of a header row segment at every vertical offset of `dys`."""
    inside = _box_means(ii, xs0 * ppm, (y + dys + 0.15 * h) * ppm, xs1 * ppm, (y + dys + 0.85 * h) * ppm)
    above = _box_means(ii, xs0 * ppm, (y + dys - 0.2 - HEADER_BAND_MM) * ppm, xs1 * ppm, (y + dys - 0.2) * ppm)
    below = _box_means(ii, xs0 * ppm, (y + dys + h + 0.2) * ppm, xs1 * ppm, (y + dys + h + 0.2 + HEADER_BAND_MM) * ppm)
    return np.minimum(above, below) - inside


def _register_row(ii, ppm, row, guard, side, search=(HEADER_SEARCH_MM, HEADER_SEARCH_MM), dy_fixed=None):
    """(dx, dy) in mm that put the nominal header row on the printed one, at one end of the row.

    Edge-based and independent of the data bits: dy maximises the darkness inside the row's band against the white
    bands above and below it (a box correlation of the vertical profile); dx aligns the known guard pattern plus the
    white paper beyond the row's end (a 1-D correlation of the horizontal profile)."""
    x, y, w, h = row[0] if side == "L" else row[-1]
    n_end = 8                                     # cells at this end used for the vertical profile
    xs0 = row[0][0] if side == "L" else row[-n_end][0]
    xs1 = row[n_end - 1][0] + w if side == "L" else row[-1][0] + w
    step = 1.0 / ppm
    best = (-1e9, 0.0)
    if dy_fixed is None:
        dys = np.arange(-search[1], search[1] + 1e-9, step)
        if len(dys):
            sc = _band_scores(ii, ppm, xs0, xs1, y, h, dys)
            k = int(np.argmax(sc))
            if sc[k] > best[0]:
                best = (float(sc[k]), float(dys[k]))
    dy = best[1] if dy_fixed is None else dy_fixed
    # horizontal: guard cells + one white cell beyond the end, as a +1 (black) / -1 (white) template
    pat = ([False] + list(guard)) if side == "L" else (list(guard) + [False])
    x_first = (row[0][0] - w) if side == "L" else row[-len(guard)][0]
    best = (-1e9, 0.0)
    dxs = np.arange(-search[0], search[0] + 1e-9, step)
    if len(dxs):
        base = (x_first + dxs)[:, None]
        i_ = np.arange(len(pat))[None, :]
        vals = _box_means(ii, (base + (i_ + 0.25) * w) * ppm, (y + dy + 0.2 * h) * ppm,
                          (base + (i_ + 0.75) * w) * ppm, (y + dy + 0.8 * h) * ppm)           # (ndx, ncell)
        sign = np.where(pat, -1.0, 1.0)          # bright where white, dark where black
        sc = (sign[None, :] * (vals - vals.mean(axis=1, keepdims=True))).sum(axis=1)
        k = int(np.argmax(sc))
        if sc[k] > best[0]:
            best = (float(sc[k]), float(dxs[k]))
    return best[1], dy


HEADER_REACH_MM = 40.0       # nothing of the header read touches the canvas further than this below the flat-field band
HEADER_FLAT_MARGIN_MM = 4.0  # paper sampled this far above / below the header rows for the flat field
HEADER_FLAT_PCT = 85         # printed black covers < 50 % of any column of that window, so this percentile is paper
HEADER_FLAT_SIGMA_MM = 1.2   # smoothing of the per-column paper level along the row


def _flatten_header(warped: np.ndarray, ppm: float) -> np.ndarray:
    """float32 copy of the canvas whose header band is divided by the paper level of its own column.

    Registration and thresholds then see a lighting-independent picture: a hard shadow edge, glare or a gradient *along*
    the row (the usual case: the header spans 144 mm) becomes a flat white background, where the raw picture has two
    different "papers" and a ``bright band - dark band`` score picks the edge instead of the cells.  Per column, not
    per pixel: the black cells never cover half of a column of the window, so a high percentile is the paper itself."""
    rows = lay.header_cell_rects()
    y0 = max(int(round((rows[0][0][1] - HEADER_FLAT_MARGIN_MM) * ppm)), 0)
    y1 = min(int(round((rows[-1][0][1] + rows[-1][0][3] + HEADER_FLAT_MARGIN_MM) * ppm)), warped.shape[0])
    # rows far below the header are never read (registration reaches < 25 mm from a row): neither converted nor integrated
    out = warped[:min(y1 + int(round(HEADER_REACH_MM * ppm)), warped.shape[0])].astype(np.float32)
    band = out[y0:y1]
    paper = np.percentile(band, HEADER_FLAT_PCT, axis=0).astype(np.float32)
    paper = cv2.GaussianBlur(paper.reshape(1, -1), (0, 0), HEADER_FLAT_SIGMA_MM * ppm).reshape(-1)
    out[y0:y1] = np.clip(band / np.maximum(paper, 8.0) * 235.0, 0, 255)
    return out


def _track_dy(ii, ppm, row, first_search: float):
    """Vertical offset (mm) of one header row at every cell, followed along the row.

    A sheet warped from three corners (or a bowed / rotated one) can have its header run 5 mm uphill over its 144 mm: far more
    than a two-end registration may search without jumping onto the other row (pitch 5.5 mm).  The vertical profile of a
    window of SEG cells (bright band - dark band - bright band, as in _register_row, independent of the data bits) is
    searched +-TRACK_STEP_MM around the previous window's offset, starting with the widest search at the left end.  A window
    with no dark cells (all-white data bits) has no contrast: the previous offset is kept."""
    n = len(row)
    x, y, w, h = row[0]
    step = 1.0 / ppm
    centres, dys = [], []
    prev = 0.0
    for k, i0 in enumerate(sorted(set(list(range(0, n - TRACK_SEG + 1, TRACK_HOP)) + [n - TRACK_SEG]))):
        xs0, xs1 = row[i0][0], row[i0 + TRACK_SEG - 1][0] + w
        half = first_search if k == 0 else TRACK_STEP_MM
        best = (-1e9, prev)
        dys_ = np.arange(prev - half, prev + half + 1e-9, step)
        if len(dys_):
            sc = _band_scores(ii, ppm, xs0, xs1, y, h, dys_)
            k = int(np.argmax(sc))
            if sc[k] > best[0]:
                best = (float(sc[k]), float(dys_[k]))
        if best[0] >= TRACK_MIN_CONTRAST:
            prev = best[1]
        centres.append(i0 + (TRACK_SEG - 1) / 2.0)
        dys.append(prev)
    return np.interp(np.arange(n), centres, dys)


def _read_header(warped: np.ndarray, ppm: float):
    """Header decode with a wider registration retry (see _read_header_pass); the CRC-16 judges every attempt."""
    flat = _flatten_header(warped, ppm)
    ii = cv2.integral(flat, sdepth=cv2.CV_64F)
    h, info = _read_header_pass(flat, ii, ppm, (HEADER_SEARCH_MM, HEADER_SEARCH_MM), (1.2, 1.5))
    if h is None and info.get("meta_contrast", 1.0) >= META_MIN_CONTRAST:
        h2, info2 = _read_header_pass(flat, ii, ppm, HEADER_WIDE_SEARCH_MM, HEADER_WIDE_DISAGREE_MM)
        if h2 is not None:
            info2["header_wide"] = 1.0
            return h2, info2
        h3, info3 = _read_header_pass(flat, ii, ppm, HEADER_WIDE_SEARCH_MM, (HEADER_WIDE_DISAGREE_MM[0], 8.0), track=True)
        if h3 is not None:
            info3["header_tracked"] = 1.0
            return h3, info3
    return h, info


def _read_header_pass(flat: np.ndarray, ii: np.ndarray, ppm: float, search, disagree, track=False):
    """Decode the two-row self-describing header.  Returns (Header | None, info).

    Each row is registered locally at both ends (_register_row) and the cell positions are interpolated along it,
    so a bow of the paper near the top edge does not move the sampling windows off the cells.  Each cell's darkness
    is measured against the paper right above and below it (shadow / glare / lighting gradient independent) and
    thresholded halfway between the black and white guard cells of its row, interpolated along the row.  If the CRC
    fails, the few least-certain cells may be flipped (cells that sit almost exactly on the threshold, e.g. under a
    toner streak); CRC-16 keeps the chance of accepting a wrong header with these few tries around 1e-4 per failed
    frame, and a wrong layout is caught by the page check and the ring alignment downstream."""
    rows = lay.header_cell_rects()
    n = mk.HEADER_ROW_CELLS
    g0, g1 = len(mk.HEADER_START_GUARD), len(mk.HEADER_STOP_GUARD)
    info: Dict[str, float] = {}
    D, B, Wt = [], [], []
    blacks: List[float] = []
    black_guard = {i for i, b in enumerate(mk.HEADER_START_GUARD) if b} | \
        {n - g1 + i for i, b in enumerate(mk.HEADER_STOP_GUARD) if b}
    for r, row in enumerate(rows):
        dys = _track_dy(ii, ppm, row, TRACK_FIRST_MM) if track else None
        dxl, dyl = _register_row(ii, ppm, row, mk.HEADER_START_GUARD, "L", search, None if dys is None else float(dys[0]))
        dxr, dyr = _register_row(ii, ppm, row, mk.HEADER_STOP_GUARD, "R", search, None if dys is None else float(dys[-1]))
        if abs(dxr - dxl) > disagree[0] or abs(dyr - dyl) > disagree[1]:     # the two ends disagree beyond any plausible bow
            info["header_reg_mm"] = float(max(abs(dxr - dxl), abs(dyr - dyl)))
            return None, info
        cx, cy, cw_, ch_ = (np.array([c[k] for c in row]) for k in range(4))
        t = np.arange(n) / (n - 1)
        dx, dy = dxl + (dxr - dxl) * t, (dyl + (dyr - dyl) * t if dys is None else np.asarray(dys, float))
        cell = _box_means(ii, (cx + dx + 0.25 * cw_) * ppm, (cy + dy + 0.2 * ch_) * ppm, (cx + dx + 0.75 * cw_) * ppm,
                          (cy + dy + 0.8 * ch_) * ppm)
        up = _box_means(ii, (cx + dx + 0.1 * cw_) * ppm, (cy + dy - 0.25 - HEADER_BAND_MM) * ppm, (cx + dx + 0.9 * cw_) * ppm,
                        (cy + dy - 0.25) * ppm)
        dn = _box_means(ii, (cx + dx + 0.1 * cw_) * ppm, (cy + dy + ch_ + 0.25) * ppm, (cx + dx + 0.9 * cw_) * ppm,
                        (cy + dy + ch_ + 0.25 + HEADER_BAND_MM) * ppm)
        paper = np.maximum(0.5 * (up + dn), 1.0)
        d = 1.0 - cell / paper
        for i in sorted(black_guard):   # printed black: darker quartile of the cell's pixels (robust to white toner streaks)
            x, y, w, h = row[i]
            y0_, x0_ = int(round((y + dy[i] + 0.2 * h) * ppm)), int(round((x + dx[i] + 0.2 * w) * ppm))
            c = flat[max(y0_, 0):max(int(round((y + dy[i] + 0.8 * h) * ppm)), y0_ + 1),
                     max(x0_, 0):max(int(round((x + dx[i] + 0.8 * w) * ppm)), x0_ + 1)]
            if c.size:
                blacks.append(1.0 - float(np.percentile(c, BLACK_PERCENTILE)) / paper[i])
        sb = [i for i, b in enumerate(mk.HEADER_START_GUARD) if b]
        sw = [i for i, b in enumerate(mk.HEADER_START_GUARD) if not b]
        eb = [n - g1 + i for i, b in enumerate(mk.HEADER_STOP_GUARD) if b]
        ew = [n - g1 + i for i, b in enumerate(mk.HEADER_STOP_GUARD) if not b]
        bl, br = float(np.mean(d[sb])), float(np.mean(d[eb]))
        wl, wr = float(np.mean(d[sw])), float(np.mean(d[ew]))
        t = np.arange(n) / (n - 1)
        D.append(d)
        B.append(bl + (br - bl) * t)
        Wt.append(wl + (wr - wl) * t)
    D, B, Wt = np.array(D), np.array(B), np.array(Wt)
    contrast = float((B - Wt).min())
    info["meta_contrast"] = contrast
    if contrast < META_MIN_CONTRAST:
        return None, info
    info["meta_black"] = float(np.clip(np.mean(blacks), 0.0, 1.0)) if blacks else float(B.mean())   # printed black
    thr = 0.5 * (B + Wt)
    bits = D > thr
    margin = np.abs(D - thr) / np.maximum(B - Wt, 1e-6)
    for row in bits:
        if tuple(row[:g0]) != mk.HEADER_START_GUARD or tuple(row[-g1:]) != mk.HEADER_STOP_GUARD:
            info["meta_min_margin"] = float(margin.min())
            return None, info
    data = np.concatenate([row[g0:n - g1] for row in bits])
    dmargin = np.concatenate([m[g0:n - g1] for m in margin])
    info["meta_min_margin"] = float(dmargin.min())
    h = mk.decode_header_data(data)
    if h is None:
        weak = [int(i) for i in np.argsort(dmargin)[:HEADER_FLIP_CANDIDATES] if dmargin[i] < HEADER_FLIP_MARGIN]
        tries = [(i,) for i in weak] + [(i, j) for k, i in enumerate(weak) for j in weak[k + 1:]]
        for flip in tries:
            d2 = data.copy()
            d2[list(flip)] ^= True
            h = mk.decode_header_data(d2)
            if h is not None:
                info["header_flips"] = float(len(flip))
                break
    return h, info


_layout_cache: Dict[str, lay.Layout] = {}


def layout_for_header(h: mk.Header) -> lay.Layout:
    L = _layout_cache.get(h.form_key)
    if L is None:
        L = _layout_cache[h.form_key] = lay.layout_from_header(h)
    return L


# =============================================================================== scoring
def _ring_offsets(gray_small, ppm_s, centers_mm, diameter_mm, search_mm=1.4):
    """Where is the printed ring really?  Zero-mean normalised correlation of a ring template against
    a small window around each expected bubble centre.  Returns (dx_mm, dy_mm, corr) arrays."""
    R = diameter_mm / 2
    srch = int(round(search_mm * ppm_s))
    half = int(np.ceil((R + 0.7) * ppm_s))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    rr = np.sqrt(xx * xx + yy * yy) / ppm_s
    tmpl = np.exp(-0.5 * ((rr - R) / 0.3) ** 2).astype(np.float32)
    gray_small = gray_small.astype(np.float32)
    H, W = gray_small.shape
    n = len(centers_mm)
    out = np.full((n, 3), np.nan)
    pad = half + srch
    for i, (cx, cy) in enumerate(centers_mm):
        x, y = int(round(cx * ppm_s)), int(round(cy * ppm_s))
        if x - pad < 0 or y - pad < 0 or x + pad >= W or y + pad >= H:
            continue
        roi = gray_small[y - pad:y + pad + 1, x - pad:x + pad + 1]
        r = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
        j = np.unravel_index(int(np.argmax(r)), r.shape)
        dy = j[0] - srch
        dx = j[1] - srch
        if 0 < j[0] < r.shape[0] - 1:   # parabolic sub-pixel refinement
            a, b, c = r[j[0] - 1, j[1]], r[j[0], j[1]], r[j[0] + 1, j[1]]
            dy += 0.5 * (a - c) / (a - 2 * b + c + 1e-9)
        if 0 < j[1] < r.shape[1] - 1:
            a, b, c = r[j[0], j[1] - 1], r[j[0], j[1]], r[j[0], j[1] + 1]
            dx += 0.5 * (a - c) / (a - 2 * b + c + 1e-9)
        out[i] = (dx / ppm_s, dy / ppm_s, r[j])
    return out[:, 0], out[:, 1], out[:, 2]


def _poly_terms(xy):
    x = (xy[:, 0] - lay.PAGE_W / 2) / 100.0
    y = (xy[:, 1] - lay.PAGE_H / 2) / 100.0
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)


def _robust_fit(A, offs, good):
    keep = good.copy()
    coef = np.linalg.lstsq(A[keep], offs[keep], rcond=None)[0]
    for _ in range(4):  # iteratively re-weighted: drop outliers (marked bubbles, glyph overlap)
        coef = np.linalg.lstsq(A[keep], offs[keep], rcond=None)[0]
        res = np.linalg.norm(A @ coef - offs, axis=1)
        sig = max(1.4826 * np.median(res[keep]), 0.05)
        new_keep = good & (res < 3 * sig + 0.05)
        if new_keep.sum() < 12 or (new_keep == keep).all():
            break
        keep = new_keep
    return coef, keep


def _fit_alignment(warped, groups, ppm, extrapolated=False, strict=False, diag=None):
    """Smooth (quadratic) correction of the nominal bubble positions, fitted to measured ring positions.

    groups: [(centers_mm, diameter_mm)].  Returns (model or None, info).  The page is warped from the visible
    corner markers only, so paper bow / lens distortion in between shows up here as a smooth offset field.

    The ring search window stays small (ALIGN_SEARCH_MM, well below half the bubble pitch) so a ring can never
    be matched to its neighbour: a lattice slip by one pitch would read a neighbouring bubble, undetectably.
    With a hidden marker the page is *extrapolated* beyond the visible markers and the nominal positions can be
    off by more than the window, so the search is repeated around the positions predicted by the model so far
    (ALIGN_REFINE_STAGES times) and the model must then be backed by rings over (almost) the whole page.
    ``diag`` (a dict) receives the half-resolution ink image used here (for the per-row ring check below).
    ``strict``: the page was straightened with its timing tracks (omr.tracks); the rings must then confirm it - a track
    field that is wrong somewhere in the interior (a crease close to an edge) leaves rings far from where it put them."""
    ppm_s = ALIGN_PX_PER_MM
    small = cv2.resize(255 - warped, None, fx=ppm_s / ppm, fy=ppm_s / ppm, interpolation=cv2.INTER_AREA)
    sels = [(np.asarray(c, np.float64)[::max(1, len(c) // ALIGN_SAMPLES_PER_GROUP)], dia) for c, dia in groups]
    pts = np.concatenate([c for c, _ in sels])
    A = _poly_terms(pts)
    coef = None
    info: dict = {}
    for stage in range(1 + (ALIGN_REFINE_STAGES if extrapolated else 0)):
        pred = np.zeros_like(pts) if coef is None else A @ coef
        offs, cor, i0 = [], [], 0
        for sel, dia in sels:
            p = pred[i0:i0 + len(sel)]
            dx, dy, cr = _ring_offsets(small, ppm_s, sel + p, dia, ALIGN_SEARCH_MM)
            offs.append(np.stack([dx, dy], 1) + p)
            cor.append(cr)
            i0 += len(sel)
        offs, cor = np.concatenate(offs), np.concatenate(cor)
        good = np.isfinite(cor) & (cor > ALIGN_MIN_CORR)
        info = {"align_ring_corr": float(np.nanmedian(cor)) if np.isfinite(cor).any() else 0.0,
                "align_points": int(good.sum()), "align_stages": stage + 1}
        if good.sum() < 12:
            return None, info
        coef, keep = _robust_fit(A, offs, good)
    if diag is not None:
        diag["small"], diag["ppm_s"] = small, ppm_s
    res = np.linalg.norm(A[keep] @ coef - offs[keep], axis=1)
    info["align_resid_mm"] = float(np.median(res))
    info["align_used"] = int(keep.sum())
    info["align_inlier_frac"] = float(keep.mean())
    info["align_max_shift_mm"] = float(np.linalg.norm(A @ coef, axis=1).max())
    # a page straightened by its tracks may still be off by a few mm in the interior (a crease through the middle is where an
    # edge-only field cannot be exact): the larger shift is accepted when the rings back the model over (almost) the whole page
    wide = extrapolated or (strict and info["align_inlier_frac"] >= ALIGN_MIN_COVERAGE)
    if info["align_max_shift_mm"] > (ALIGN_MAX_SHIFT_EXTRAP_MM if wide else ALIGN_MAX_SHIFT_MM):
        return None, info
    if extrapolated and info["align_inlier_frac"] < ALIGN_MIN_COVERAGE:
        return None, info      # rings not confirmed over the extrapolated part of the page: do not trust it
    if strict and (info["align_ring_corr"] < TRACKS_MIN_RING_CORR or info["align_inlier_frac"] < TRACKS_MIN_INLIERS):
        return None, info
    return coef, info


def _ring_scores(small, ppm_s, centers_mm, diameter_mm) -> np.ndarray:
    """Zero-mean normalised correlation of a (slightly tolerant) ring template with the image exactly at each centre, no search."""
    R = diameter_mm / 2
    half = int(np.ceil((R + 1.0) * ppm_s))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    rr = np.sqrt(xx * xx + yy * yy) / ppm_s
    t = np.exp(-0.5 * ((rr - R) / RING_SCORE_SIGMA_MM) ** 2).astype(np.float32)
    t -= t.mean()
    t /= np.linalg.norm(t)
    c = np.round(np.asarray(centers_mm, np.float64).reshape(-1, 2) * ppm_s).astype(int)
    H, W = small.shape
    pad = np.pad(small.astype(np.float32), half, mode="edge")
    iy = c[:, 1][:, None, None] + half + np.arange(-half, half + 1)[None, :, None]
    ix = c[:, 0][:, None, None] + half + np.arange(-half, half + 1)[None, None, :]
    iy, ix = np.clip(iy, 0, H + 2 * half - 1), np.clip(ix, 0, W + 2 * half - 1)
    P = pad[iy, ix]
    P = P - P.mean(axis=(1, 2), keepdims=True)
    return (P * t).sum(axis=(1, 2)) / np.maximum(np.linalg.norm(P.reshape(len(P), -1), axis=1), 1e-3)


def _rows_misfit(diag, groups) -> list:
    """Per group (centres (n_rows, n_per_row, 2) mm, bubble diameter, rings that must show): which rows have too few printed rings
    exactly where the model puts them.

    A page straightened by its timing tracks is only exact along the edges; a crease or tight curl close to an edge
    leaves the interior wrong there, and a few mm off a bubble reads its neighbour (or nothing), silently.  In a row
    whose bubbles are all in place at least ``need`` of them show their bare ring (the others may carry marks); a row
    where fewer do is not where it is believed to be: unreliable (ambiguous, to be looked at)."""
    small, ppm_s = diag["small"], diag["ppm_s"]
    scores = [_ring_scores(small, ppm_s, c, dia).reshape(c.shape[:2]) for c, dia, _ in groups]
    ref = float(np.percentile(np.concatenate([s.ravel() for s in scores]), 75))
    thr = max(RING_SCORE_MIN, RING_SCORE_FRAC * ref)
    diag["ring_ref"] = ref
    return [(s >= thr).sum(axis=1) < need for s, (_, _, need) in zip(scores, groups)]


def _apply_alignment(centers, coef):
    c = np.asarray(centers, np.float64)
    if coef is None:
        return c
    return c + _poly_terms(c) @ coef


@dataclass
class _Calib:
    ink: float     # typical darkness of a mark's pixels on this page
    typ: float     # typical fill (area fraction) of the darkest bubble of marked questions
    t_hi: float    # fill above which a bubble counts as marked
    t_lo: float    # fill below which it counts as empty
    min_strength: float = MIN_STRENGTH


def _ink_level(groups, floor: float):
    """(page ink level or None, number of confident marks).

    Only bubbles whose own ink is at least `floor` (a fraction of printed black, measured on the same frame)
    take part, so erasure residue can never define "ink".  The page may therefore only tighten the scale
    above the absolute floor, never loosen it below."""
    vals = []
    for D in groups:
        hot = D.mean(axis=1) > HOT_MEAN_DARKNESS
        if hot.any():
            p85 = np.percentile(D[hot], 85, axis=1)
            vals.extend(p85[p85 >= floor])
    if len(vals) < MIN_CALIB_MARKS:
        return None, len(vals)
    return float(np.percentile(vals, INK_PERCENTILE)), len(vals)


UNRELIABLE_FILL, UNRELIABLE_STRENGTH = 0.5, 0.3   # an unknowable bubble (see omr.illum): ambiguous, never a read digit


def unreliable_mask(f: np.ndarray, strength: np.ndarray) -> np.ndarray:
    """Bubbles carrying the UNRELIABLE sentinel of `_fill` (measured fill/strength are never exactly this pair)."""
    return (np.asarray(f) == UNRELIABLE_FILL) & (np.asarray(strength) == UNRELIABLE_STRENGTH)


def _weak_pick(f: np.ndarray, strength: np.ndarray, t_lo: float, need: float = 0.0) -> list:
    """Best-guess choice of a row with no mark reaching t_in: the fullest MEASURED bubble (at least `need`), never a tie
    broken by index (equal fullest bubbles -> no pick, the row stays blank and flagged).  An unknowable (sentinel) bubble is
    picked only by elimination: it is the single one and every measured bubble of the row is clearly out (< t_lo)."""
    bad = unreliable_mask(f, strength)
    ok = ~bad
    if bad.sum() == 1 and ok.any() and (f[ok] < t_lo).all():
        return [int(np.argmax(bad))]
    if not ok.any():
        return []
    top = float(f[ok].max())
    best = np.nonzero(ok & (f >= top - 1e-6))[0]
    return [int(best[0])] if len(best) == 1 and top >= need else []


def _fill(D: np.ndarray, ink: float, unreliable: Optional[np.ndarray] = None):
    """(fill, strength) per bubble.

    fill     : area fraction of the disc that is inked.  A pixel counts as ink around half the page's ink
               darkness (ramp 0.35..0.65), which keeps the area estimate independent of edge blur, makes a
               lighter pencil mark and a dark pen mark of equal extent give a similar value, and rejects
               faint residue.
    strength : mean darkness of the inked pixels relative to the page's ink level.  A uniformly grey bubble
               (smudge, erased ghost) has full "fill" but low strength."""
    ink = np.asarray(ink, np.float32)
    if ink.ndim:                    # one level per bubble (omr.ink.local_levels)
        ink = ink[:, None]
    fill = np.clip((D - 0.35 * ink) / (0.3 * ink), 0, 1)
    m = D >= 0.35 * ink
    strength = (D * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1) / ink[..., 0] if ink.ndim else \
        (D * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1) / ink
    fill = fill.mean(axis=1)
    if unreliable is not None and unreliable.any():
        # A bubble whose darkness cannot be known (shadow edge across it, glare) is flagged, not guessed: with
        # t_hi >= T_HI_MIN, t_lo <= T_LO_FRAC * T_HI_MAX and these values `_decide` can only call its row
        # ambiguous, whatever the other bubbles are (strict digits: never read).  The flag lives in the fills,
        # so multi-frame fusion carries it without knowing about it.
        fill, strength = fill.copy(), strength.copy()
        fill[unreliable], strength[unreliable] = UNRELIABLE_FILL, UNRELIABLE_STRENGTH
    return fill, strength


REL_COVER_PCT = 35          # a relative mark must be uniform: the 35th percentile of its disc ...
REL_COVER_FRAC = 0.6        # ... at least this fraction of the 85th.  Measured: faint full marks 0.83+ (10th pct), half-filled bubbles <= 0.3,
                            # even on the small ID bubbles where blur smears a dark half over the whole disc


def _mean_dark(D: np.ndarray, unreliable: Optional[np.ndarray] = None) -> np.ndarray:
    """Darkness a bubble shows over (nearly) its whole disc, paper-relative 0..1, for the relative rule: the 85th
    percentile of the disc (`omr.ink.bubble_ink`, the ink floor's measure) when the disc is evenly dark (p35 >=
    REL_COVER_FRAC x p85), else 0.  So a faint but complete mark counts with its darkness, while a half-filled bubble, a stray
    or a dot (dark on a part only) never fires the rule - they stay with the absolute rules (flagged).  Independent of
    the ink floor.  An unknowable bubble is NaN: the rule then leaves its row alone."""
    p85 = inkmod.bubble_ink(D)
    m = np.where(np.percentile(D, REL_COVER_PCT, axis=1) >= REL_COVER_FRAC * p85, p85, 0.0).astype(np.float32)
    if unreliable is not None and unreliable.any():
        m[unreliable] = np.nan
    return m


def _relative_marks(md: Optional[np.ndarray], strict: bool):
    """(indices, excess) of the choices whose mean darkness exceeds the median of the question's OTHER choices by
    REL_MARGIN.  Median of the others: one other mark (multi-mark) or a ghost cannot hide a faint mark, and a shadow or
    glare that lifts the whole row cancels out.  A row with an unknowable bubble gives nothing."""
    if md is None or md.size < 3 or np.isnan(md).any():
        return [], np.zeros(0)
    ex = np.array([md[i] - np.median(np.delete(md, i)) for i in range(md.size)])
    idx = np.nonzero(ex >= REL_MARGIN)[0]
    return [int(i) for i in idx], ex[idx]


_OTHERS: Dict[int, np.ndarray] = {}


def _rel_rows(M: Optional[np.ndarray], strict: bool = False):
    """`_relative_marks` for every row of a mean-darkness matrix (rows x choices) in one vectorised pass (numerically the
    same numbers: the median of the OTHER choices, float32)."""
    if M is None:
        return None
    M = np.asarray(M)
    if M.ndim != 2:
        return [([], np.zeros(0))] * len(M)
    n, k = M.shape
    if k < 3:
        return [([], np.zeros(0))] * n
    oth = _OTHERS.get(k)
    if oth is None:
        oth = _OTHERS[k] = np.array([[j for j in range(k) if j != i] for i in range(k)])
    ok = ~np.isnan(M).any(axis=1)
    out = [([], np.zeros(0))] * n
    if not ok.any():
        return out
    sel = np.nonzero(ok)[0]
    Ms = M[sel]
    ex = Ms - np.median(Ms[:, oth], axis=2)
    hit = ex >= REL_MARGIN
    for r in np.nonzero(hit.any(axis=1))[0]:
        idx = np.nonzero(hit[r])[0]
        out[sel[r]] = ([int(i) for i in idx], ex[r][idx])
    return out


def _calibrate(F: np.ndarray, ink: float, default_typ: Optional[float] = None, min_strength: float = MIN_STRENGTH) -> _Calib:
    """Page-level thresholds from the fills of all bubbles (F: questions x choices)."""
    top = F.max(axis=1)
    hot = top[top >= 0.55]  # clearly filled bubbles only; half-filled ones must not drag the threshold down
    typ = float(np.median(hot)) if len(hot) >= 5 else (default_typ if default_typ is not None else DEFAULT_TYP_FILL)
    t_hi = float(np.clip(T_HI_OF_TYP * typ, T_HI_MIN, T_HI_MAX))
    return _Calib(ink, typ, t_hi, T_LO_FRAC * t_hi, min_strength)


def _decide(f: np.ndarray, strength: np.ndarray, c: _Calib, strict: bool = False, md: Optional[np.ndarray] = None,
            trust_rel: bool = True, pre=None):
    """Absolute rules (`_decide_abs`), plus the relative rule (REL_MARGIN) for rows the absolute rules cannot call
    with confidence: a faint pencil mark under glare is only ~20 % darker than the paper but clearly darker than its
    neighbours.  The relative rule only ADDS reads to rows that would otherwise be blank / flagged guesses; a row with
    a confident absolute mark is never changed (erasure ghosts and strays beside a real mark stay out)."""
    # pre = ((_decide_abs result), (_relative_marks result)) already computed by the caller for this very row
    state, marked, conf, amb, confident = _decide_abs(f, strength, c, strict) if pre is None else pre[0]
    if md is None or not trust_rel:
        return state, marked, conf, amb
    if confident:
        # faint-pencil page: every real mark is evenly dark (md > 0).  A bubble the absolute rules read as marked but that is
        # dark on a part of its disc only (md == 0: a half-filled bubble whose fill the low page ink overstates) is flagged.
        if marked and not amb and any(md[i] == 0.0 for i in marked):
            return state, marked, conf, True
        return state, marked, conf, amb
    rel, ex = _relative_marks(md, strict) if pre is None else pre[1]
    if not rel:
        return state, marked, conf, amb
    sure = ex.min() >= (REL_SURE_DIGIT if strict else REL_SURE)
    if not (set(marked) <= set(rel)):        # the absolute guess is a different bubble: leave the row to the flag
        return state, marked, conf, amb
    rconf = float(np.clip(0.5 + (ex.min() - REL_MARGIN) / 0.3, 0.0, 0.8))
    return ("multi" if len(rel) >= 2 else "single"), rel, rconf, not sure


def _decide_abs(f: np.ndarray, strength: np.ndarray, c: _Calib, strict: bool = False):
    """One question (or ID column) from its bubbles' fills.  Returns (state, marked, confidence, ambiguous).

    in   : fill >= t_hi, at least 70% of the fullest bubble in the question, and inked strongly enough
    out  : fill < t_lo, or under 35% (strict: 25%) of the fullest bubble
    grey : anything else -> ambiguous, flagged for a human

    Strict (digit) reading also needs a margin above t_hi.  A half-filled bubble measures up to t_hi + 0.07 (277
    synthetic ones under blur, glare, pale print), because the fill ramp is relative to the *page's* ink level: the
    darker the mark is than that level (strength > 1) the further its blurred edge counts as ink and the more a
    partial mark is overstated.  A wrong student ID is the worst outcome, so the margin grows with strength
    (DIGIT_MARGIN from DIGIT_MARGIN_FROM to DIGIT_MARGIN_FULL).  A faint, honestly partial mark is not overstated, so it
    keeps the plain t_hi - which keeps the cost in honest digit reads small (a flat margin would not read 0.75 % of marks).
    """
    fmax = float(f.max())
    rel_out = 0.25 if strict else 0.35
    t_in = c.t_hi + (DIGIT_MARGIN * np.clip((strength - DIGIT_MARGIN_FROM) / (DIGIT_MARGIN_FULL - DIGIT_MARGIN_FROM), 0.0, 1.0)
                     if strict else 0.0)             # per bubble
    if not (f >= t_in).any():
        if fmax < c.t_lo:
            return "blank", [], float(min(1.0, 0.5 + 0.5 * (c.t_lo - fmax) / c.t_lo)), False, False
        guess = _weak_pick(f, strength, c.t_lo, 0.5 * (c.t_lo + c.t_hi)) if fmax >= 0.5 * (c.t_lo + c.t_hi) else []  # best guess, always flagged
        return ("single" if guess else "blank"), guess, 0.2, True, False
    r = f / fmax
    is_in = (f >= t_in) & (r >= 0.7) & (strength >= c.min_strength)
    is_out = (f < c.t_lo) | (r < rel_out)
    grey = ~is_in & ~is_out
    marked = [int(i) for i in np.nonzero(is_in)[0]]
    if not marked:  # filled but weak (smudge / ghost): best guess, always flagged
        pick = _weak_pick(f, strength, c.t_lo)
        return ("single" if pick else "blank"), pick, 0.2, True, False
    # confidence: how far the weakest "in" bubble is above the threshold, and how small the strongest
    # competing bubble is relative to the fullest one
    conf_in = float(np.clip(0.5 + (f[is_in] - (t_in[is_in] if strict else c.t_hi)).min() / 0.3, 0.0, 1.0))
    rest = f[~is_in]
    conf_out = float(np.clip(1.0 - (rest.max() / fmax) / 0.5, 0.0, 1.0)) if rest.size else 1.0
    conf = min(conf_in, conf_out)
    ambiguous = bool(grey.any())
    state = "multi" if len(marked) >= 2 else "single"
    return state, marked, conf, ambiguous, True


@dataclass
class PageRaw:
    """Per-bubble raw measurements of one frame (what `decide_page` needs; fusion medians these over a track)."""
    numbers: List[int]
    labels: List[List[str]]
    digit_index: List[int]
    FA: np.ndarray            # questions x choices: fill
    SA: np.ndarray            # strength
    FI: np.ndarray            # id digits x 10
    SI: np.ndarray
    ink: float
    min_strength: float
    verified: bool = True     # bubble positions confirmed against the printed rings
    FS: Optional[np.ndarray] = None   # subject-code digits x 10 (None: the form has no subject code)
    SS: Optional[np.ndarray] = None
    ink_source: str = "default"       # how the ink floor was chosen: "auto" (the sheet's own evidence) | "default" (0.45 x black)
    ink_separation: float = 0.0       # marks vs. ghosts, x printed black (omr.ink)
    MA: Optional[np.ndarray] = None   # disc darkness per bubble (answers / ID / subject): input of the relative rule
    MI: Optional[np.ndarray] = None
    MS: Optional[np.ndarray] = None


@dataclass
class PageDecision:
    answers: List[QuestionResult]
    id_digits: List[DigitResult]
    student_id: Optional[str]
    flags: List[str]                  # id_blank / id_unreadable / subject_blank / subject_unreadable
    calibration: Dict[str, object]
    subject_digits: List[DigitResult]
    subject: Optional[str]            # "" when the form has no subject code, None when not readable


GUESS_GAP = 0.15            # best guess: the fullest bubble leads the runner-up by at least this much fill (and reaches t_lo) ...
GUESS_REL_GAP = 0.10        # ... or (faint ink) its disc darkness exceeds the others' median by REL_MARGIN and the runner-up's excess by this much


def _guess(f: np.ndarray, strength: np.ndarray, c: _Calib, md: Optional[np.ndarray] = None) -> Optional[int]:
    """Most plausible choice of an UNCERTAIN row, or None when no choice stands out (then the row stays without a value).
    Used only for rows that are already flagged: a clearly blank row never gets a guess.  Unknowable bubbles (the
    UNRELIABLE sentinel of `_fill`) are never picked - their value is invented, not measured."""
    ok = ~unreliable_mask(f, strength)
    if f.size >= 2 and ok.sum() >= 1:
        cand = np.nonzero(ok)[0]
        order = cand[np.argsort(-f[cand])]
        top = float(f[order[0]])
        second = float(f[order[1]]) if len(order) > 1 else 0.0
        if top >= c.t_lo and top - second >= GUESS_GAP:
            return int(order[0])
    if md is not None and md.size >= 3 and not np.isnan(md).any():
        ex = np.array([md[i] - np.median(np.delete(md, i)) for i in range(md.size)])
        o = np.argsort(-ex)
        if ex[o[0]] >= REL_MARGIN and ex[o[0]] - ex[o[1]] >= GUESS_REL_GAP:
            return int(o[0])
    return None


def _read_digits(F, S_, cal, verified, M_=None, trust_rel=True, pre=None):
    """Digit columns (ID, subject code): strict rules.  A digit is CONFIDENT only when unambiguous; an uncertain column
    (ambiguous, or alignment unverified) still carries its best guess (`_guess`), marked `guessed` and ambiguous.  A
    clearly blank column, or one where nothing stands out, has no value.  Returns (digits, value string or None,
    why-not-read flag or None, any digit guessed): with every digit valued the string is produced even when some digits
    are guessed (the caller flags it)."""
    digits = []
    for di, (f, st) in enumerate(zip(F, S_)):
        md = None if M_ is None else M_[di]
        state, marked, conf, amb = _decide(f, st, cal, strict=True, md=md, trust_rel=trust_rel,
                                           pre=None if pre is None else pre[di])
        val, guessed = None, False
        if state == "single" and not amb and verified:
            val = int(marked[0])
        elif amb or not verified:
            g = _guess(f, st, cal, md)       # (a leader is needed also when the rules above guessed one: two equal greys have none)
            if g is not None and (state != "blank" or amb):
                val, guessed, amb = g, True, True
        digits.append(DigitResult(di, state, val, round(conf, 3), amb, [round(float(x), 4) for x in f], guessed))
    any_guess = any(d.guessed for d in digits)
    if all(d.value is not None for d in digits):
        return digits, "".join(str(d.value) for d in digits), None, any_guess
    return digits, None, ("blank" if all(d.state == "blank" and not d.ambiguous for d in digits) else "unreadable"), any_guess


REL_TRUST_ROWS = 0.4        # a region is a "faint pencil region" when this share of its rows has a relative-only mark ...
REL_TRUST_GHOSTS = 0.02     # ... and (almost) no other-choice beside a confident mark is as dark (erasure ghost / stray evidence), per row
REL_TRUST_WINDOW = 7        # answers: the region of a question = the questions within this many places before / after it
REL_STRONG = 0.30           # a region (not the whole page) is trusted on marks at least this much darker than their row (excess; erasure ghosts measured <= 0.25)
REL_TRUST_MIN_DIGITS = 3    # ID / subject block: at least this many digit columns with a relative-only mark


def _rel_evidence(F, S_, M_, cal, strict, pre=None):
    """Per row: (has a relative-only mark, the same with excess >= REL_STRONG, ghost-like other choices beside a confident mark).
    pre: per row ((_decide_abs result), (_relative_marks result)) when the caller has them."""
    if pre is None:
        pre = _pre_rows(F, S_, M_, cal, strict)
    rel_only = np.zeros(len(F), bool)
    strong = np.zeros(len(F), bool)
    ghosts = np.zeros(len(F), int)
    for i, ((state, marked, conf, amb, confident), (rel, ex)) in enumerate(pre):
        if confident:
            ghosts[i] = len([j for j in rel if j not in marked])
        elif rel:
            rel_only[i] = True
            strong[i] = bool(ex.min() >= REL_STRONG)
    return rel_only, strong, ghosts


def _pre_rows(F, S_, M_, cal, strict):
    """Absolute decision + relative marks of every row, computed once (shared by the trust evidence and the decisions)."""
    rel = _rel_rows(M_, strict)
    if rel is None:
        rel = [([], np.zeros(0))] * len(F)
    return [(_decide_abs(f, st, cal, strict), r) for f, st, r in zip(F, S_, rel)]


def _rel_trust_answers(raw: PageRaw, cal: _Calib, pre=None):
    """Per question: does the page, or ITS REGION, say that relative-only marks are marks rather than erasure ghosts?  A faint
    graphite mark and the ghost of an erased one look alike (both ~15-25 % darker than paper, evenly), so the relative rule
    applies only where (a) a large share of the questions has such a mark and nothing else explains them - the sheet is
    written in faint ink, or lit by glare - and (b) the confidently marked questions show no ghosts beside their marks.
    Elsewhere the rule is OFF (measured: on ordinary sheets with erasures it turned ghosts into flagged guesses / silent
    reads and cost ID read-rate).  Two scopes: the whole page (>= REL_TRUST_ROWS of all questions: faint graphite throughout), or a
    sliding window of neighbouring questions where the relative marks are STRONG (>= REL_STRONG, darker than any ghost): a sheet
    written half in faint pen / half in dark pen gets the rule where it is needed; a window is small and ghost-only rows can
    fill it by chance, hence the stronger evidence required."""
    n = len(raw.numbers)
    if raw.MA is None or not n:
        return np.zeros(n, bool), False
    rel_only, strong, ghosts = _rel_evidence(raw.FA, raw.SA, raw.MA, cal, False, pre)
    if rel_only.sum() >= REL_TRUST_ROWS * n and ghosts.sum() <= REL_TRUST_GHOSTS * n:
        return np.ones(n, bool), True
    out = np.zeros(n, bool)
    for i in range(n):
        lo, hi = max(0, i - REL_TRUST_WINDOW), min(n, i + REL_TRUST_WINDOW + 1)
        m = hi - lo
        out[i] = strong[lo:hi].sum() >= REL_TRUST_ROWS * m and ghosts[lo:hi].sum() <= REL_TRUST_GHOSTS * m
    return out, False


def _rel_trust_digits(F, S_, M_, cal, pre=None) -> bool:
    """The same for the ID + subject block as one region (strict digit rules): written in pencil next to a pen-written
    answer area, the block has its own evidence - most of its columns hold a relative-only mark and none shows a ghost."""
    if M_ is None or not len(F):
        return False
    rel_only, strong, ghosts = _rel_evidence(F, S_, M_, cal, True, pre)
    n = len(F)
    return bool(strong.sum() >= max(REL_TRUST_MIN_DIGITS, REL_TRUST_ROWS * n) and ghosts.sum() <= REL_TRUST_GHOSTS * n)


def decide_page(raw: PageRaw) -> PageDecision:
    """The one place where bubble measurements become decisions.  Used per frame by `recognize` and on
    track-fused measurements by `omr.fusion`."""
    cal = _calibrate(raw.FA, raw.ink, min_strength=raw.min_strength)
    answers = []
    preA = _pre_rows(raw.FA, raw.SA, raw.MA, cal, False)         # absolute decision + relative marks per row, computed once
    trust, page_trust = _rel_trust_answers(raw, cal, preA)
    MA = raw.MA if raw.MA is not None else [None] * len(raw.numbers)
    for n, lab, f, st, md, tr, pre in zip(raw.numbers, raw.labels, raw.FA, raw.SA, MA, trust, preA):
        state, marked, conf, amb = _decide(f, st, cal, md=md, trust_rel=bool(tr), pre=pre)
        amb = bool(amb or not raw.verified)
        guess = None
        if amb:       # an uncertain row keeps its flag AND gets the most plausible value (a clear blank / confident row: none)
            if state == "single":
                guess = int(marked[0])
            elif state == "blank":     # a multi-mark row gets NO guess: all its marks are the answer (flagged 복수마킹)
                guess = _guess(f, st, cal, md)
        answers.append(QuestionResult(n, state, marked, list(lab), round(conf, 3), amb,
                                      [round(float(x), 4) for x in f], guess, guess is not None))
    FD = raw.FI if raw.FS is None else np.concatenate([raw.FI, raw.FS])
    ical = _calibrate(FD, raw.ink, default_typ=cal.typ, min_strength=raw.min_strength)
    SD = raw.SI if raw.SS is None else np.concatenate([raw.SI, raw.SS])
    MD = None if raw.MI is None else (raw.MI if raw.MS is None else np.concatenate([raw.MI, raw.MS]))
    preD = _pre_rows(FD, SD, MD, ical, True)
    nI = len(raw.FI)
    dtrust = page_trust or _rel_trust_digits(FD, SD, MD, ical, preD)
    digits, student_id, why, gs = _read_digits(raw.FI, raw.SI, ical, raw.verified, raw.MI, dtrust, preD[:nI])
    for d, di in zip(digits, raw.digit_index):
        d.index = di
    flags: List[str] = [] if why is None else ["id_" + why]
    if gs and student_id is not None:
        flags.append("id_guessed")
    sdigits, subject = [], ""
    if raw.FS is not None and len(raw.FS):
        sdigits, subject, why, gs = _read_digits(raw.FS, raw.SS, ical, raw.verified, raw.MS, dtrust, preD[nI:])
        if why is not None:
            flags.append("subject_" + why)
        if gs and subject is not None:
            flags.append("subject_guessed")
    if any(a.guessed for a in answers):
        flags.append("guessed_answers")
    calib = {"ink": cal.ink, "typ_fill": cal.typ, "t_lo": cal.t_lo, "t_hi": cal.t_hi,
             "id_typ_fill": ical.typ, "id_t_lo": ical.t_lo, "id_t_hi": ical.t_hi,
             "ink_auto": float(raw.ink_source == "auto"), "ink_source": raw.ink_source, "ink_separation": raw.ink_separation}
    return PageDecision(answers, digits, student_id, flags, calib, sdigits, subject)


# =============================================================================== main
DROP_RETRY_STATUSES = (BAD_GEOMETRY, BAD_META)
DROP_RETRY_SKIP_FLAGS = ("page_mismatch", "layout_mismatch")


def recognize(image: np.ndarray, resolve_layout: Optional[Callable[[int], Optional[lay.Layout]]] = None,
              px_per_mm: float = DEFAULT_PX_PER_MM, keep_images: bool = True) -> RecognitionResult:
    """Read one frame.  A curled corner can carry a marker that is found but does not lie on the sheet's plane: the
    four-marker homography then fails (marker reprojection error, header off its cells) where the other three alone,
    with the ring alignment / tracks bending the rest, read fine.  So a frame that fails geometry or header with all four
    markers is retried once without the marker that fits the others worst (as if it were hidden)."""
    t_start = time.perf_counter()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    det = find_markers_scaled(gray)
    t_det = (time.perf_counter() - t0) * 1000.0
    res = _recognize(gray, det, resolve_layout, px_per_mm, keep_images, t_start, t_det)
    if res.status in DROP_RETRY_STATUSES and not any(f in res.flags for f in DROP_RETRY_SKIP_FLAGS):
        found, dup, dinfo = det
        page = res.page
        ids = [i for i in found if page is not None and i // 4 == page]
        if len(ids) == 4:
            worst = _worst_marker(found, ids, page)
            red = {i: c for i, c in found.items() if i != worst}
            res2 = _recognize(gray, (red, dup, {**dinfo, "dropped_marker": float(worst)}), resolve_layout, px_per_mm, keep_images,
                              t_start, t_det)
            if res2.status == OK:
                res2.flags.append("marker_dropped")
                return res2
            res.timing_ms["total"] = (time.perf_counter() - t_start) * 1000.0
    return res


def _worst_marker(found, ids, page) -> int:
    """The marker whose corners the homography of all of them reprojects worst (mean corner error)."""
    mm = _marker_corners_mm(page)
    src = np.concatenate([found[i] for i in ids])
    dst = np.concatenate([mm[i] for i in ids])
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        return ids[0]
    err = np.linalg.norm(cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2) - dst, axis=1).reshape(len(ids), 4).mean(1)
    return ids[int(np.argmax(err))]


def _recognize(gray: np.ndarray, det, resolve_layout, px_per_mm: float, keep_images: bool, t_start: float,
               t_det: float) -> RecognitionResult:
    tm: Dict[str, float] = {}

    def lap(name, t0):
        tm[name] = (time.perf_counter() - t0) * 1000.0

    def done(res: RecognitionResult) -> RecognitionResult:
        tm["total"] = (time.perf_counter() - t_start) * 1000.0
        res.timing_ms = tm
        return res

    # ---- 1. markers (found by the caller)
    found, dup, dinfo = det
    found = dict(found)
    tm["detect"] = t_det
    res = RecognitionResult(status=NO_SHEET, px_per_mm=px_per_mm)
    if not found:
        return done(res)
    res.marker_ids = sorted(found)
    res.marker_corners = {i: c.tolist() for i, c in found.items()}
    if dup:
        res.status = INCONSISTENT
        res.flags.append("duplicate_marker")
        return done(res)
    by_group: Dict[int, List[int]] = {}
    for i in found:
        by_group.setdefault(i // 4, []).append(i)
    solid = [g for g, v in by_group.items() if len(v) >= MIN_MARKERS]
    if len(solid) > 1:
        res.status = INCONSISTENT
        res.flags.append("mixed_page_markers")
        return done(res)
    if not solid:
        # fewer than MIN_MARKERS of any page group: the sheet is in view but not usable (covered / cut off)
        top = max(len(v) for v in by_group.values())
        lead = [g for g, v in by_group.items() if len(v) == top]
        if len(lead) > 1:
            res.status = INCONSISTENT
            res.flags.append("mixed_page_markers")
            return done(res)
        page = lead[0]
        res.status = PARTIAL
        res.page = page
        res.missing_corners = [c for c in mk.CORNERS if mk.marker_id(page, c) not in found]
        return done(res)
    page = solid[0]
    foreign_ids = [i for i in found if i // 4 != page]
    found = {i: found[i] for i in found if i // 4 == page}
    res.missing_corners = [c for c in mk.CORNERS if mk.marker_id(page, c) not in found]

    # ---- 2. homography from all visible corner points (>= 12)
    t0 = time.perf_counter()
    mm = _marker_corners_mm(page)
    ids_sorted = sorted(found)
    src = np.concatenate([found[i] for i in ids_sorted])
    dst = np.concatenate([mm[i] for i in ids_sorted]) * px_per_mm
    Hm, _ = cv2.findHomography(src, dst, 0)
    if Hm is None or not np.isfinite(Hm).all():
        res.status = BAD_GEOMETRY
        return done(res)
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), Hm).reshape(-1, 2)
    err_mm = np.linalg.norm(proj - dst, axis=1) / px_per_mm
    Hinv = np.linalg.inv(Hm)
    back = cv2.perspectiveTransform(dst.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
    err_px = np.linalg.norm(back - src, axis=1)
    c0 = cv2.perspectiveTransform(np.array([[[lay.PAGE_W / 2 * px_per_mm, lay.PAGE_H / 2 * px_per_mm]]]), Hinv)[0, 0]
    # local scale at the page centre: source px per mm
    d = cv2.perspectiveTransform(np.array([[[lay.PAGE_W / 2 * px_per_mm + px_per_mm, lay.PAGE_H / 2 * px_per_mm]],
                                           [[lay.PAGE_W / 2 * px_per_mm, lay.PAGE_H / 2 * px_per_mm + px_per_mm]]]), Hinv)
    src_ppm = float(np.sqrt(np.linalg.norm(d[0, 0] - c0) * np.linalg.norm(d[1, 0] - c0)))
    # displacement (mm) of every visible corner marker in the warp: anchors of the timing-track chains (omr.tracks)
    corner_d = {}
    for n_, i in enumerate(ids_sorted):
        corner_d[mk.CORNERS[i % 4]] = ((proj - dst)[4 * n_:4 * n_ + 4].mean(0) / px_per_mm)
    res.homography = Hm.tolist()
    res.page = page
    res.quality = {"reproj_max_mm": float(err_mm.max()), "reproj_rms_mm": float(np.sqrt((err_mm ** 2).mean())),
                   "reproj_max_px": float(err_px.max()), "source_px_per_mm": src_ppm,
                   "sharpness": _quality_sharpness(gray, src), "foreign_markers": len(foreign_ids),
                   "markers_used": len(found), **dinfo}
    lap("homography", t0)
    # a sheet with timing tracks may be bent far more than a plain one (the markers sit where the paper is, not on a plane):
    # a large marker reprojection error is then only a suspicion, settled below by whether the tracks can bend it flat
    geom_suspect = MAX_REPROJ_MM < err_mm.max() <= MAX_REPROJ_TRACKS_MM
    if (err_mm.max() > MAX_REPROJ_MM and not geom_suspect) or not _plausible_sheet(Hinv, px_per_mm):
        res.status = BAD_GEOMETRY
        return done(res)
    # markers of another page group are tolerated only when they lie outside this sheet (another sheet / object)
    for i in foreign_ids:
        quad = cv2.perspectiveTransform(np.asarray(res.marker_corners[i], np.float64).reshape(-1, 1, 2), Hm).reshape(4, 2) / px_per_mm
        if np.mean([np.linalg.norm(quad[(k + 1) % 4] - quad[k]) for k in range(4)]) < FOREIGN_MIN_SIDE_FRAC * lay.MARKER_MM:
            continue                     # far smaller than a corner marker, yet inside the sheet: a pattern in the print, not a marker
        ctr = quad.mean(0)
        if any(np.linalg.norm(ctr - cv2.perspectiveTransform(c.reshape(-1, 1, 2), Hm).reshape(4, 2).mean(0) / px_per_mm) < lay.MARKER_MM
               for c in found.values()):
            continue                     # overlaps a marker of this page: two markers cannot share a spot, it is a mis-decode of that marker's surroundings
        if -FOREIGN_MARGIN_MM <= ctr[0] <= lay.PAGE_W + FOREIGN_MARGIN_MM and -FOREIGN_MARGIN_MM <= ctr[1] <= lay.PAGE_H + FOREIGN_MARGIN_MM:
            res.status = INCONSISTENT
            res.flags.append("mixed_page_markers")
            return done(res)
    if res.missing_corners:
        res.flags.append("corner_hidden")

    # ---- 3. warp
    t0 = time.perf_counter()
    cw, ch = int(round(lay.PAGE_W * px_per_mm)), int(round(lay.PAGE_H * px_per_mm))
    warped = cv2.warpPerspective(gray, Hm, (cw, ch), flags=cv2.INTER_LINEAR, borderValue=255)
    lap("warp", t0)
    res.warped = warped if keep_images else None

    # ---- 4. header: the sheet describes its own layout (legacy sheets: 16-bit form id, layout from the resolver)
    t0 = time.perf_counter()
    tfield = None
    if geom_suspect:       # see above: only a sheet whose tracks bend it flat is accepted
        tfield = trk.measure(warped, px_per_mm, corner_d)
        if tfield is None:
            res.status = BAD_GEOMETRY
            return done(res)
        warped = trk.rectify(warped, tfield, px_per_mm)
        res.warped = warped if keep_images else None
    hdr, info = _read_header(warped, px_per_mm)
    if geom_suspect and (hdr is None or hdr.version < 2):
        res.status = BAD_GEOMETRY          # not a bent sheet with tracks: the marker error stands
        return done(res)
    meta_pre, meta_src = None, warped
    if hdr is None:
        meta_pre = _read_meta(warped, px_per_mm)         # a readable legacy strip: an older sheet, nothing to straighten
    if hdr is None and meta_pre[0] is None:
        # header unreadable on the plain warp: a folded / curled sheet with timing tracks reads after they straighten it
        tfield = trk.measure(warped, px_per_mm, corner_d)
        if tfield is not None:
            flat_ = trk.rectify(warped, tfield, px_per_mm)
            hdr, info2 = _read_header(flat_, px_per_mm)
            if hdr is not None and hdr.version >= 2:
                info = info2
                warped = flat_
                res.warped = warped if keep_images else None
            else:
                hdr, tfield = None, None
    res.quality.update(info)
    if hdr is not None:
        layout = layout_for_header(hdr)
        hpage, total = hdr.page, hdr.pages
    else:
        meta, linfo = meta_pre if (meta_pre is not None and warped is meta_src) else _read_meta(warped, px_per_mm)
        if meta is None:
            lap("meta", t0)
            res.status = BAD_META
            return done(res)
        res.quality.update(linfo)
        res.form_id = meta.form_id
        hpage, total = meta.page, meta.total_pages
        layout = resolve_layout(meta.form_id) if resolve_layout is not None else None
        if layout is None:
            lap("meta", t0)
            res.status = UNKNOWN_FORM
            return done(res)
        if layout.form_id != meta.form_id:
            lap("meta", t0)
            res.status = BAD_META
            res.flags.append("layout_mismatch")
            return done(res)
    lap("meta", t0)
    res.form_key, res.total_pages = layout.form_key, total
    if hpage != page:
        res.status = BAD_META
        res.flags.append("page_mismatch")
        return done(res)
    if layout.total_pages != total or page >= len(layout.pages):
        res.status = BAD_META
        res.flags.append("layout_mismatch")
        return done(res)
    pg = layout.pages[page]

    # ---- 4b. timing tracks (header version >= 2): bend the page flat again, so everything below sits at its nominal place
    t0 = time.perf_counter()
    if pg.ticks:
        if tfield is None:          # not yet measured (the header read fine on the plain warp)
            tfield = trk.measure(warped, px_per_mm, corner_d)
            if tfield is not None and tfield.max_disp() >= trk.FLAT_MM:
                warped = trk.rectify(warped, tfield, px_per_mm)
                res.warped = warped if keep_images else None
            elif tfield is not None:                    # flat sheet: the tracks confirm the homography, nothing to resample
                res.quality.update(tfield.info())
                res.quality["tracks"] = 1.0
                tfield = trk.FLAT
        if tfield is not trk.FLAT:
            res.quality["tracks"] = 1.0 if tfield is not None else 0.0
            if tfield is not None:
                res.quality.update(tfield.info())
                res.dewarp = tfield
    lap("tracks", t0)

    # ---- 5. bubbles
    t0 = time.perf_counter()
    smooth = cv2.GaussianBlur(warped, (0, 0), 0.25 * px_per_mm)
    # serial strip: a failure never invalidates the frame (serial stays None, soft flag)
    res.serial, sstate = _read_serial(warped, px_per_mm)
    if sstate == "unreadable":
        res.flags.append("serial_unreadable")
    qs = sorted((q for col in pg.columns for q in col.questions), key=lambda q: q.number)
    nc = layout.template.choices
    cent = [(b.cx, b.cy) for q in qs for b in q.bubbles]
    g = pg.id_grid
    sg = pg.subject_grid
    # the ID and subject grids share one bubble geometry: one group for alignment and scoring
    icols = list(g.columns) + (list(sg.columns) if sg is not None else [])
    n_id = len(g.columns)
    icent = [(b.cx, b.cy) for col in icols for b in col.bubbles]
    adiag: dict = {}
    coef, ainfo = _fit_alignment(smooth, [(cent, pg.bubble_diameter), (icent, g.bubble_diameter)], px_per_mm,
                                 extrapolated=bool(res.missing_corners) and tfield is None, strict=tfield is not None and tfield is not trk.FLAT, diag=adiag)
    res.quality.update(ainfo)
    cent = _apply_alignment(cent, coef)
    icent = _apply_alignment(icent, coef)
    ic = np.round(icent, 2).reshape(len(icols), 10, 2)
    res.bubble_centers_mm = {"answers": np.round(cent, 2).reshape(len(qs), nc, 2).tolist(),
                             "id": ic[:n_id].tolist(), "subject": ic[n_id:].tolist()}
    DA, UA = illum.bubble_darkness(smooth, cent, pg.bubble_diameter, px_per_mm, "E")
    DI, UI = illum.bubble_darkness(smooth, icent, g.bubble_diameter, px_per_mm, "E")
    UA, UI = ~UA, ~UI                                    # illum returns "reliable"; the fills want "unreliable"
    if coef is not None and tfield is not None and tfield is not trk.FLAT and adiag:   # a straightened page: interior check
        bad_q, bad_i = _rows_misfit(adiag, [(cent.reshape(len(qs), nc, 2), pg.bubble_diameter, 2 if nc >= 4 else 1),
                                           (icent.reshape(len(icols), 10, 2), g.bubble_diameter, 4)])
        UA, UI = UA | np.repeat(bad_q, nc), UI | np.repeat(bad_i, 10)
        res.quality["rows_misfit"] = float(bad_q.sum() + bad_i.sum())
    black = float(np.clip(res.quality.get("meta_black", 0.9), META_MIN_CONTRAST, 1.0))   # printed-black darkness on this frame
    # the sheet chooses its own ink floor (never below ABS_INK_FRAC x black); illum before ink: unreliable bubbles are 0 in D
    rows = list(inkmod.bubble_ink(DA).reshape(-1, nc)) + list(inkmod.bubble_ink(DI).reshape(-1, 10))
    ia = inkmod.choose_floor(rows, black)
    abs_ink = ia.floor
    page_ink, n_marks = _ink_level([DA, DI], abs_ink)
    ink = max(abs_ink, page_ink or 0.0)
    min_str = MIN_STRENGTH if page_ink is not None else STRENGTH_UNCALIBRATED
    res.quality.update({"ink_abs": abs_ink, "ink_marks_used": n_marks, "unreliable_bubbles": int(UA.sum() + UI.sum())})
    # local ink levels: pencil ID block next to dark-pen answers, or a fresh pen half-way through, must each be judged by
    # their own neighbourhood's marks, not by the page-wide level (the pen's) - omr.ink.local_levels
    hot_ink = lambda D: np.where(D.mean(axis=1) > HOT_MEAN_DARKNESS, inkmod.bubble_ink(D), 0.0)
    rm_a, rm_i = hot_ink(DA).reshape(len(qs), nc).max(1), hot_ink(DI).reshape(len(icols), 10).max(1)
    lv = inkmod.local_levels(np.concatenate([rm_a, rm_i]), abs_ink, ink,
                             [(range(len(qs)), True), (range(len(qs), len(qs) + len(icols)), False)])
    inkA, inkI = np.repeat(lv[:len(qs)], nc), np.repeat(lv[len(qs):], 10)
    res.quality["ink_local_span"] = float(lv.max() / max(lv.min(), 1e-6))
    FA, SA = _fill(DA, inkA, UA)
    FI, SI = _fill(DI, inkI, UI)
    MA, MI = _mean_dark(DA, UA).reshape(len(qs), nc), _mean_dark(DI, UI).reshape(len(icols), 10)
    FA, SA = FA.reshape(len(qs), nc), SA.reshape(len(qs), nc)
    FI, SI = FI.reshape(len(icols), 10), SI.reshape(len(icols), 10)
    raw = PageRaw([q.number for q in qs], [[b.label for b in q.bubbles] for q in qs], [col.digit_index for col in g.columns],
                  FA, SA, FI[:n_id], SI[:n_id], float(ink), float(min_str), verified=coef is not None,
                  FS=FI[n_id:] if sg is not None else None, SS=SI[n_id:] if sg is not None else None,
                  ink_source=ia.source, ink_separation=float(ia.separation),
                  MA=MA, MI=MI[:n_id], MS=MI[n_id:] if sg is not None else None)
    dec = decide_page(raw)
    res.answers, res.id_digits, res.student_id, idflags, res.calibration = \
        dec.answers, dec.id_digits, dec.student_id, dec.flags, dec.calibration
    res.subject_digits, res.subject = dec.subject_digits, dec.subject
    order = sorted(range(len(res.answers)), key=lambda i: res.answers[i].number)
    res.answers = [res.answers[i] for i in order]
    res.raw = raw
    if coef is None:  # bubble positions could not be verified against the printed rings: trust nothing
        res.flags.append("alignment_unverified")
    res.flags.extend(idflags)
    lap("score", t0)

    # ---- 7. name box crop (kept so a later stage can read/OCR it)
    x, y, w, h = pg.name_box
    res.name_crop = warped[int(y * px_per_mm):int((y + h) * px_per_mm), int(x * px_per_mm):int((x + w) * px_per_mm)].copy() \
        if keep_images else None
    res.status = OK
    return done(res)
