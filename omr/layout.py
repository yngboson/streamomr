"""Template params -> Layout. All coordinates in mm, page-relative, origin top-left.

The Layout is the single source of truth for both the PDF renderer and the
recognizer. Rect = [x, y, w, h]. Text y is the baseline.
"""
from __future__ import annotations

import dataclasses
import json
import math
import re
import typing
from dataclasses import dataclass, field
from typing import List, Optional

from . import markers as mk

# ------------------------------------------------------------------ design constants
PAGE_W, PAGE_H = 210.0, 297.0
MARGIN = 8.0
MARKER_MM = 16.0
QUIET_MM = 3.0
META_CELL_W, META_CELL_H, META_Y = 3.6, 4.5, 9.0        # legacy single-row meta strip (form id only)
HEADER_CELL_W, HEADER_ROW_H = 3.6, 4.0                 # self-describing header: 2 rows
HEADER_ROW_Y = (8.0, 13.5)                             # 1.5 mm white gap between rows (and above / below): local paper level
TITLE_Y = 23.5
SUBJ_GAP = 7.0                                         # ID grid -> subject grid (room for its row labels)
SERIAL_CELL_W, SERIAL_CELL_H, SERIAL_Y = 3.6, 4.5, 275.0   # footer strip, right end kept clear of the BR marker
SERIAL_CLEARANCE = 6.0   # mm between the strip / "No." label and the marker squares (ink spread, blur)
INFO_TOP = 30.0
PAD = 3.0
FRAME_INSET = 0.25  # keeps frame strokes inside the margin
ID_WRITEIN_H = 7.0
ID_ROW_LABEL_W = 5.0
NAME_BOX_H = 25.0
ANS_HEADER_H = 6.0
ANS_GAP = 3.0
QNUM_W = 8.0
MIN_PITCH = 5.5
MAX_PITCH = 7.5
MAX_BUBBLE = 6.0
BAND_ROWS = 5
MAX_CHOICES, MAX_ID_DIGITS = 10, 12


@dataclass
class Template:
    """What a form looks like.  Everything that decides geometry is encoded in the sheet's header
    (omr.markers.Header), so a sheet can be read without knowing the template in advance; the texts
    (title, labels, instruction) are only printed."""
    questions: int = 45                 # questions printed on the form
    choices: int = 5
    id_digits: int = 8
    pages: Optional[int] = None         # None = as few as fit
    title: str = ""
    choice_labels: Optional[List[str]] = None    # one of the header label styles: 1.., A.., a.., 가..
    min_bubble_mm: float = 4.0          # 4.0, 4.25, ... 5.75
    used: Optional[int] = None          # questions actually used (1..questions); None = all.  Unused ones are not printed
    subject_digits: int = 2             # marked subject code next to the student ID (0 = none)
    label_id: str = "수험번호"
    label_name: str = "성명"
    label_subject: str = "과목코드"
    instruction: str = "보기 원 안을 진하게 완전히 칠하세요. 고칠 때는 깨끗이 지우세요."


@dataclass
class Text:
    text: str
    x: float
    y: float
    size_pt: float
    align: str = "l"  # l / c / r
    gray: float = 0.0
    kind: str = ""


@dataclass
class MarkerSpec:
    id: int
    corner: str
    rect: List[float]
    quiet: List[float]


@dataclass
class MetaCell:
    rect: List[float]
    bit: bool


@dataclass
class Bubble:
    choice: int
    label: str
    cx: float
    cy: float


@dataclass
class Question:
    number: int  # 1-based, continuous across pages
    num_text: Text
    bubbles: List[Bubble]


@dataclass
class Band:
    rect: List[float]
    gray: float


@dataclass
class Column:
    rect: List[float]
    headers: List[Text]
    bands: List[Band]
    questions: List[Question]


@dataclass
class IdBubble:
    value: int
    cx: float
    cy: float


@dataclass
class IdColumn:
    digit_index: int  # 0 = leftmost / most significant
    writein: List[float]
    bubbles: List[IdBubble]


@dataclass
class IdGrid:
    bubble_diameter: float
    columns: List[IdColumn]
    row_labels: List[Text]


@dataclass
class Page:
    index: int
    markers: List[MarkerSpec]
    meta_rect: List[float]
    meta_cells: List[MetaCell]   # header cells (all rows), or the legacy meta strip
    info_box: List[float]
    id_grid: IdGrid
    id_label: Text
    name_box: List[float]
    name_label: Text
    answer_box: List[float]
    bubble_diameter: float
    first_question: int
    n_questions: int             # questions read on this page (used ones)
    columns: List[Column]
    texts: List[Text]  # title, footer, ...
    subject_grid: Optional[IdGrid] = None
    subject_label: Optional[Text] = None


@dataclass
class Layout:
    page_w: float
    page_h: float
    margin: float
    form_id: Optional[int]        # legacy sheets only (16-bit id in the old meta strip); None for self-describing ones
    total_pages: int
    questions_per_page: List[int]  # used questions per page
    stroke_mm: float
    template: Template
    pages: List[Page]
    form_key: str = ""            # identity of the printed form (Header.form_key; "F%04X" for legacy sheets)
    version: int = 0              # header format version; 0 = legacy meta strip

    def header(self, page: int) -> mk.Header:
        t = self.template
        return mk.Header(t.questions, t.used or t.questions, t.choices, t.id_digits, t.subject_digits, self.total_pages,
                         page, mk.label_style(t.choice_labels), _bubble_class(t.min_bubble_mm))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kw)

    @classmethod
    def from_dict(cls, d: dict) -> "Layout":
        L = _build(cls, d)
        if L.version == 0:          # legacy layout: no subject code, form identified by its 16-bit id
            L.template.subject_digits = 0
            L.form_key = L.form_key or f"F{L.form_id:04X}"
        return L

    @classmethod
    def from_json(cls, s: str) -> "Layout":
        return cls.from_dict(json.loads(s))


def _build(tp, v):
    origin = typing.get_origin(tp)
    if origin is list:
        (a,) = typing.get_args(tp)
        return [_build(a, x) for x in v]
    if origin is typing.Union:
        arg = [a for a in typing.get_args(tp) if a is not type(None)][0]
        return None if v is None else _build(arg, v)
    if dataclasses.is_dataclass(tp):
        hints = typing.get_type_hints(tp)
        # missing fields take their defaults and unknown ones are ignored, so layouts saved by older versions load
        return tp(**{f.name: _build(hints[f.name], v[f.name]) for f in dataclasses.fields(tp) if f.name in v})
    if tp is float:
        return float(v)
    return v


def _r(v: float) -> float:
    return round(float(v), 3)


def marker_rects(p: int) -> dict:
    """Corner name -> [x, y, w, h] (mm) of the marker's black square for page index p.

    Marker positions are fixed design constants (independent of the template), so a
    recognizer can use them before it knows which form it is looking at.
    """
    far = PAGE_W - MARGIN - MARKER_MM
    low = PAGE_H - MARGIN - MARKER_MM
    pos = {"TL": (MARGIN, MARGIN), "TR": (far, MARGIN), "BR": (far, low), "BL": (MARGIN, low)}
    return {c: [_r(x), _r(y), MARKER_MM, MARKER_MM] for c, (x, y) in pos.items()}


def meta_cell_rects() -> List[List[float]]:
    """[x, y, w, h] (mm) of every cell of the meta bit strip (template independent)."""
    n = mk.META_CELLS
    mx = (PAGE_W - META_CELL_W * n) / 2
    return [_rect(mx + i * META_CELL_W, META_Y, META_CELL_W, META_CELL_H) for i in range(n)]


def header_cell_rects() -> List[List[List[float]]]:
    """[row][cell] -> [x, y, w, h] (mm) of the self-describing header (template independent)."""
    n = mk.HEADER_ROW_CELLS
    x0 = (PAGE_W - HEADER_CELL_W * n) / 2
    return [[_rect(x0 + i * HEADER_CELL_W, y, HEADER_CELL_W, HEADER_ROW_H) for i in range(n)] for y in HEADER_ROW_Y]


def serial_cell_rects() -> List[List[float]]:
    """[x, y, w, h] (mm) of every cell of the serial strip (footer, template independent)."""
    n = mk.SERIAL_CELLS
    x1 = PAGE_W - MARGIN - MARKER_MM - SERIAL_CLEARANCE
    x0 = x1 - n * SERIAL_CELL_W
    return [_rect(x0 + i * SERIAL_CELL_W, SERIAL_Y, SERIAL_CELL_W, SERIAL_CELL_H) for i in range(n)]


def serial_label(serial: int) -> Text:
    """Human-readable serial printed below the strip (right-aligned to it)."""
    x1 = PAGE_W - MARGIN - MARKER_MM - SERIAL_CLEARANCE
    return Text(f"No. {serial:04d}", x1, 285.0, 9, "r", 0.0, "serial")


def _rect(*a) -> List[float]:
    return [_r(x) for x in a]


# ------------------------------------------------------------------ geometry
def _id_geometry(t: Template):
    dia = max(t.min_bubble_mm, 4.2)
    return dia, max(6.2, dia * 1.45), max(MIN_PITCH, dia * 1.3)  # dia, pitch_x, pitch_y


def _info_bottom(t: Template) -> float:
    _, _, py = _id_geometry(t)
    return INFO_TOP + PAD + 5 + ID_WRITEIN_H + 2 + 10 * py + PAD


def _answer_area(t: Template) -> List[float]:
    top = _info_bottom(t) + 3.0
    bottom = PAGE_H - MARGIN - MARKER_MM - QUIET_MM
    left = MARGIN + FRAME_INSET
    return [left, top, PAGE_W - 2 * MARGIN - 2 * FRAME_INSET, bottom - top]


def _choose_grid(t: Template, q: int):
    """Best (ncols, rows, pitch_x, pitch_y, col_w) for q questions on one page, or None."""
    _, _, aw, ah = _answer_area(t)
    avail_h = ah - ANS_HEADER_H - 2.0
    best = None
    for ncols in range(1, q + 1 if q > 0 else 2):
        col_w = (aw - (ncols - 1) * ANS_GAP) / ncols
        px = min(MAX_PITCH, (col_w - QNUM_W) / t.choices)
        rows = math.ceil(q / ncols)
        py = min(MAX_PITCH, avail_h / rows)
        if px < MIN_PITCH or py < MIN_PITCH:
            continue
        score = min(px, py)
        if best is None or score > best[0] + 1e-9:
            best = (score, ncols, rows, px, py, col_w)
    return best[1:] if best else None


def page_capacity(t: Template) -> int:
    """Max questions on one page at minimum pitch."""
    _, _, aw, ah = _answer_area(t)
    avail_h = ah - ANS_HEADER_H - 2.0
    rows = int(avail_h // MIN_PITCH)
    min_col = QNUM_W + t.choices * MIN_PITCH
    ncols = int((aw + ANS_GAP) // (min_col + ANS_GAP))
    return max(0, rows * ncols)


def _resolve_split(t: Template) -> List[int]:
    """Printed questions per page: an even split over the pages (so it follows from questions + pages alone)."""
    Q = t.questions
    cap = page_capacity(t)
    if cap < 1:
        raise ValueError("no room for any question at minimum bubble size")
    if t.pages is not None:
        n = t.pages
        if n < 1 or n > Q:
            raise ValueError("invalid page count")
        if math.ceil(Q / n) > cap:
            raise ValueError(f"{Q} questions in {n} pages exceeds capacity {cap}/page")
    else:
        n = math.ceil(Q / cap)  # automatic split
    base, extra = divmod(Q, n)
    return [base + (1 if i < extra else 0) for i in range(n)]


def _bubble_class(min_bubble_mm: float) -> int:
    k = (min_bubble_mm - mk.BUBBLE_BASE_MM) / mk.BUBBLE_STEP_MM
    if abs(k - round(k)) > 1e-6 or not 0 <= round(k) < 8:
        raise ValueError(f"min_bubble_mm must be {mk.BUBBLE_BASE_MM} + k*{mk.BUBBLE_STEP_MM} (k = 0..7)")
    return int(round(k))


def _text_width_mm(s: str, size_pt: float) -> float:
    em = size_pt * 0.3528
    return sum(em * (1.0 if ord(c) > 0x2E80 else 0.55) for c in s)


def validate(t: Template) -> None:
    if t.questions < 1:
        raise ValueError("questions must be >= 1")
    if not 2 <= t.choices <= MAX_CHOICES:
        raise ValueError(f"choices must be 2..{MAX_CHOICES}")
    if not 1 <= t.id_digits <= MAX_ID_DIGITS:
        raise ValueError(f"id_digits must be 1..{MAX_ID_DIGITS}")
    if t.choice_labels is not None and len(t.choice_labels) != t.choices:
        raise ValueError("choice_labels length must equal choices")
    _bubble_class(t.min_bubble_mm)
    if t.questions >= 1 << 10:
        raise ValueError("questions must be < 1024")
    if t.used is not None and not 1 <= t.used <= t.questions:
        raise ValueError("used must be 1..questions")
    if not 0 <= t.subject_digits <= 3:
        raise ValueError("subject_digits must be 0..3")
    if t.choice_labels is not None and mk.label_style(list(t.choice_labels)) is None:
        raise ValueError("choice_labels must be 1 2 3.., A B C.., a b c.. or 가 나 다.. (the header stores only the style)")


def build_layout(template: Template) -> Layout:
    t = dataclasses.replace(template)
    if isinstance(t.choice_labels, str):
        t.choice_labels = list(t.choice_labels)
    validate(t)
    if t.choice_labels is None:
        t.choice_labels = [str(i + 1) for i in range(t.choices)]
    split = _resolve_split(t)
    total = len(split)
    if total > mk.MAX_PAGES:
        raise ValueError(f"{total} pages needed; max is {mk.MAX_PAGES}")
    used = t.used or t.questions
    if used <= sum(split[:-1]):
        raise ValueError(f"only {used} of {t.questions} questions used: page {total} would be empty (use fewer pages)")
    t.pages = total
    pages, first, used_pp = [], 1, []
    for p, nq in enumerate(split):
        n_used = max(0, min(nq, used - (first - 1)))
        pages.append(_build_page(t, p, total, first, nq, n_used))
        used_pp.append(n_used)
        first += nq
    L = Layout(PAGE_W, PAGE_H, MARGIN, None, total, used_pp, 0.25, t, pages, version=mk.HEADER_VERSION)
    L.form_key = L.header(0).form_key
    for pg in L.pages:   # the header cells are only known once the layout exists
        rows = mk.encode_header(L.header(pg.index))
        pg.meta_cells = [MetaCell(r, b) for rr, bb in zip(header_cell_rects(), rows) for r, b in zip(rr, bb)]
    return L


def layout_from_header(h: mk.Header) -> Layout:
    """The layout a self-describing sheet describes (texts are not part of the header: no title)."""
    return build_layout(Template(questions=h.questions, choices=h.choices, id_digits=h.id_digits, pages=h.pages,
                                 choice_labels=h.choice_labels(), used=h.used, subject_digits=h.subject_digits,
                                 min_bubble_mm=mk.BUBBLE_BASE_MM + mk.BUBBLE_STEP_MM * h.bubble, title=""))


def header_from_form_key(form_key: str, page: int = 0) -> mk.Header:
    """Inverse of ``Header.form_key`` ("Q100U80C5D8S2P2L0B0V1") for self-describing sheets; ValueError otherwise."""
    m = re.fullmatch(r"Q(\d+)U(\d+)C(\d+)D(\d+)S(\d+)P(\d+)L(\d+)B(\d+)V(\d+)", form_key or "")
    if m is None:
        raise ValueError(f"not a self-describing form key: {form_key!r}")
    q, u, c, d, s, p, l, b, v = map(int, m.groups())
    h = mk.Header(q, u, c, d, s, p, page, l, b, v)
    mk.validate_header(h)
    return h


def describe(L: Layout) -> str:
    """One-line human description of a form (Korean)."""
    t = L.template
    used = t.used or t.questions
    q = f"{used}문항" if used == t.questions else f"{t.questions}문항 양식 중 {used}문항"
    return f"{q} · {t.choices}지선다 · {L.total_pages}쪽 · 수험번호 {t.id_digits}자리" + \
        (f" · 과목코드 {t.subject_digits}자리" if t.subject_digits else "")


def _id_columns(n: int, gx0: float, writein_top: float, bub_top: float, ipx: float, ipy: float) -> List[IdColumn]:
    cols = []
    for i in range(n):
        cx = gx0 + (i + 0.5) * ipx
        cols.append(IdColumn(i, _rect(cx - ipx / 2 + 0.4, writein_top, ipx - 0.8, ID_WRITEIN_H),
                             [IdBubble(v, _r(cx), _r(bub_top + (v + 0.5) * ipy)) for v in range(10)]))
    return cols


def _row_labels(gx0: float, bub_top: float, ipy: float, kind: str) -> List[Text]:
    return [Text(str(v), _r(gx0 - 1.5), _r(bub_top + (v + 0.5) * ipy + 1.1), 7, "r", 0.5, kind) for v in range(10)]


def _build_page(t: Template, p: int, total: int, first_q: int, nq: int, n_used: int) -> Page:
    """nq: printed question slots on this page (geometry); n_used: how many of them are used (printed and read)."""
    # ---- corner markers
    mspecs = []
    for c in mk.CORNERS:
        x, y = marker_rects(p)[c][:2]
        mspecs.append(MarkerSpec(mk.marker_id(p, c), c, _rect(x, y, MARKER_MM, MARKER_MM),
                                 _rect(x - QUIET_MM, y - QUIET_MM, MARKER_MM + 2 * QUIET_MM, MARKER_MM + 2 * QUIET_MM)))
    # ---- header (cells are filled in by build_layout once the layout exists)
    hr = header_cell_rects()
    meta_rect = _rect(hr[0][0][0], hr[0][0][1], HEADER_CELL_W * mk.HEADER_ROW_CELLS, hr[-1][0][1] + HEADER_ROW_H - hr[0][0][1])

    texts: List[Text] = []
    # ---- title (shrunk to fit between markers)
    free_w = PAGE_W - 2 * (MARGIN + MARKER_MM + QUIET_MM)
    if t.title:
        size = 13.0
        while size > 7 and _text_width_mm(t.title, size) > free_w:
            size -= 0.5
        texts.append(Text(t.title, PAGE_W / 2, TITLE_Y, size, "c", 0.0, "title"))
    # ---- footer
    fx = MARGIN + MARKER_MM + QUIET_MM
    used = t.used or t.questions
    qtxt = f"Q1-{used}" + (f" of {t.questions}" if used != t.questions else "")
    texts.append(Text(f"page {p + 1} / {total}  |  {qtxt}", fx, 280.0, 8, "l", 0.3, "footer"))
    if t.instruction:
        texts.append(Text(t.instruction, fx, 285.0, 7, "l", 0.4, "instruction"))

    # ---- student info area: ID grid, subject-code grid, name box
    idia, ipx, ipy = _id_geometry(t)
    info_bottom = _info_bottom(t)
    left = MARGIN + FRAME_INSET
    info_box = _rect(left, INFO_TOP, PAGE_W - 2 * left, info_bottom - INFO_TOP)
    label_y = INFO_TOP + PAD + 3.5
    id_label = Text(t.label_id, left + PAD, label_y, 9, "l", 0.0, "id_label")
    writein_top = INFO_TOP + PAD + 5
    bub_top = writein_top + ID_WRITEIN_H + 2
    gx0 = left + PAD + ID_ROW_LABEL_W
    grid = IdGrid(_r(idia), _id_columns(t.id_digits, gx0, writein_top, bub_top, ipx, ipy),
                  _row_labels(gx0, bub_top, ipy, "id_row"))
    nx = gx0 + t.id_digits * ipx + 8
    subject_grid = subject_label = None
    if t.subject_digits:
        sx0 = gx0 + t.id_digits * ipx + SUBJ_GAP
        subject_grid = IdGrid(_r(idia), _id_columns(t.subject_digits, sx0, writein_top, bub_top, ipx, ipy),
                              _row_labels(sx0, bub_top, ipy, "subject_row"))
        subject_label = Text(t.label_subject, _r(sx0 - ID_ROW_LABEL_W + 1.0), label_y, 9, "l", 0.0, "subject_label")
        nx = max(sx0 + t.subject_digits * ipx, sx0 - ID_ROW_LABEL_W + 1.0 + _text_width_mm(t.label_subject, 9)) + 8
    nw = PAGE_W - left - PAD - nx
    if nw < 25:
        raise ValueError("no room for the name box (too many ID / subject digits)")
    name_box = _rect(nx, writein_top, nw, NAME_BOX_H)
    name_label = Text(t.label_name, _r(nx), _r(label_y), 9, "l", 0.0, "name_label")

    # ---- answer area (geometry from all nq slots; only the first n_used are printed and read)
    abox = _answer_area(t)
    ncols, rows, px, py, col_w = _choose_grid(t, nq)
    dia = max(t.min_bubble_mm, min(MAX_BUBBLE, 0.7 * min(px, py)))
    columns, q = [], 0
    y_rows = abox[1] + ANS_HEADER_H + 2.0
    for c in range(ncols):
        cx0 = abox[0] + c * (col_w + ANS_GAP)
        in_col = min(rows, nq - q, n_used - q)
        if in_col <= 0:
            break
        headers = [Text(t.choice_labels[k], _r(cx0 + QNUM_W + (k + 0.5) * px), _r(abox[1] + 4.2), 8, "c", 0.35, "choice_label")
                   for k in range(t.choices)]
        bands, qs = [], []
        for r in range(in_col):
            if (r // BAND_ROWS) % 2 == 1:
                bands.append(Band(_rect(cx0, y_rows + r * py, col_w, py), 0.95))
            cy = y_rows + (r + 0.5) * py
            number = first_q + q
            qs.append(Question(number,
                               Text(str(number), _r(cx0 + QNUM_W - 2.0), _r(cy + 1.2), 9, "r", 0.0, "qnum"),
                               [Bubble(k, t.choice_labels[k], _r(cx0 + QNUM_W + (k + 0.5) * px), _r(cy))
                                for k in range(t.choices)]))
            q += 1
        columns.append(Column(_rect(cx0, abox[1], col_w, ANS_HEADER_H + 2.0 + rows * py), headers, bands, qs))
    return Page(p, mspecs, meta_rect, [], info_box, grid, id_label, _rect(*name_box), name_label,
                _rect(*abox), _r(dia), first_q, n_used, columns, texts, subject_grid, subject_label)
