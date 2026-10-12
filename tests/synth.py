"""Synthetic camera frames with ground truth.

The paper is the real PDF produced by omr.render_pdf, rasterised at 300 dpi. Human-like marks are
drawn onto that raster, then the sheet is placed into a simulated camera frame (perspective, rotation,
clutter, lighting, blur, noise, JPEG).  Nothing here is imported by the recognizer.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pymupdf

from omr import layout as lay
from omr.layout import Layout
from omr.render_pdf import render_pdf

DPI = 300
PPM_HI = DPI / 25.4


# ================================================================= paper rasters
_raster_cache: Dict[Tuple[str, int], np.ndarray] = {}


def paper_raster(pdf_path: str, page: int) -> np.ndarray:
    key = (os.path.abspath(pdf_path), page)
    if key not in _raster_cache:
        doc = pymupdf.open(pdf_path)
        pix = doc[page].get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY)
        _raster_cache[key] = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w).copy()
    return _raster_cache[key]


def build_pdf(layout: Layout, path: str) -> str:
    if not os.path.exists(path):
        render_pdf(layout, path)
    return path


# ================================================================= printer defects
def print_defects(raster: np.ndarray, page: int, rng, black=(0.4, 0.75), streak_mm=(0.1, 0.8), p_band=0.5) -> np.ndarray:
    """What a cheap printer does to the *printed* page (marks are drawn afterwards, they are not printed):
    black comes out as grey (toner density `black`), thin horizontal white streaks (dropped toner, 0.1-0.8 mm),
    dense streak zones, density banding along the paper feed direction, faint paper-tone bands and - with
    probability p_band - a white band across one corner marker's full width, which breaks its black border."""
    H, W = raster.shape
    ink = (255.0 - raster.astype(np.float32)) / 255.0
    ink *= rng.uniform(*black)
    mm = PPM_HI

    def cut(y, h, x0, x1, keep):
        y0 = int(max(0, round(y))); y1 = int(min(H, max(round(y + h), y0 + 1)))
        ink[y0:y1, int(max(0, x0)):int(min(W, x1))] *= keep

    for _ in range(int(rng.integers(8, 30))):                      # isolated streaks, partial width
        x0 = rng.uniform(0, 0.5) * W
        cut(rng.uniform(0, H), rng.uniform(*streak_mm) * mm, x0, x0 + rng.uniform(0.3, 1.0) * W, rng.uniform(0.0, 0.35))
    for _ in range(int(rng.integers(1, 4))):                       # dense zones: periodic thin white lines
        y0, zh = rng.uniform(0, H), rng.uniform(8, 30) * mm
        period = rng.uniform(0.8, 2.5) * mm
        thick = min(rng.uniform(*streak_mm) * mm, 0.7 * period)
        x0 = rng.uniform(0, 0.4) * W
        x1 = x0 + rng.uniform(0.3, 0.6) * W
        y = y0
        while y < min(H, y0 + zh):
            cut(y, thick, x0, x1, rng.uniform(0.0, 0.3))
            y += period * rng.uniform(0.8, 1.2)
    if rng.random() < 0.6:                                          # dense streaks right across a marker
        rects = lay.marker_rects(page)
        x, y, w, h = rects[list(rects)[int(rng.integers(len(rects)))]]
        period = rng.uniform(0.9, 2.0) * mm
        thick = min(rng.uniform(*streak_mm) * mm, 0.7 * period)
        yy_ = (y - 1) * mm
        while yy_ < (y + h + 1) * mm:
            cut(yy_, thick, (x - 2) * mm, (x + w + 2) * mm, rng.uniform(0.0, 0.25))
            yy_ += period * rng.uniform(0.8, 1.2)
    if rng.random() < p_band:                                       # a band through one marker (whole width)
        rects = lay.marker_rects(page)
        x, y, w, h = rects[list(rects)[int(rng.integers(len(rects)))]]
        cut((y + rng.uniform(0.15, 0.85) * h) * mm, rng.uniform(0.5, 1.1) * mm, (x - 2) * mm, (x + w + 2) * mm, 0.0)
    yy = np.arange(H, dtype=np.float32)[:, None]
    ink *= 1.0 + 0.08 * np.sin(2 * np.pi * yy / (rng.uniform(8, 30) * mm) + rng.uniform(0, 6.28))   # density banding
    out = 255.0 * (1.0 - ink)
    tone = np.ones((H, 1), np.float32)
    for _ in range(int(rng.integers(2, 6))):                        # faint grey paper bands
        y0 = int(rng.uniform(0, H)); tone[y0:y0 + int(rng.uniform(5, 20) * mm)] *= rng.uniform(0.93, 0.99)
    return np.clip(out * tone, 0, 255).astype(np.uint8)


# ================================================================= ink
def _hnoise(rng, n=48, harmonics=4, amp=0.12):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    out = np.zeros(n)
    for h in range(1, harmonics + 1):
        out += rng.normal(0, amp / h) * np.cos(h * th + rng.uniform(0, 6.28))
    return th, out


def draw_ink(img: np.ndarray, cx_mm: float, cy_mm: float, R_mm: float, rng, kind: str, **kw) -> None:
    """Draw one mark in place on the 300 dpi raster. kind: fill | stray | dot | erased | half."""
    half = int((R_mm + 2.2) * PPM_HI)
    cx, cy = cx_mm * PPM_HI, cy_mm * PPM_HI
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    if x0 < 0 or y0 < 0 or y0 + 2 * half > img.shape[0] or x0 + 2 * half > img.shape[1]:
        return
    m = np.zeros((2 * half, 2 * half), np.float32)
    ox, oy = cx - x0, cy - y0  # centre inside the patch (px)
    R = R_mm * PPM_HI
    if kind in ("fill", "half", "erased"):
        if kind == "fill":
            f = kw.get("fill", rng.uniform(0.6, 1.0))
        elif kind == "half":
            f = kw.get("fill", rng.uniform(0.35, 0.5))
        else:
            f = rng.uniform(0.7, 1.0)
        if f >= 0.93:
            rho0 = R * rng.uniform(0.92, 1.12)   # tends to overflow the ring a bit
            off = rng.uniform(0, 0.25) * PPM_HI
        else:
            rho0 = R * np.sqrt(f) * rng.uniform(0.95, 1.1)
            off = rng.uniform(0, max(R - rho0, 0) + 0.25 * PPM_HI)
        a = rng.uniform(0, 2 * np.pi)
        px, py = ox + off * np.cos(a), oy + off * np.sin(a)
        th, nz = _hnoise(rng, amp=0.14 if kind != "erased" else 0.2)
        el = rng.uniform(0.85, 1.15)
        pts = np.stack([px + rho0 * (1 + nz) * np.cos(th) * el, py + rho0 * (1 + nz) * np.sin(th) / el], 1)
        cv2.fillPoly(m, [np.round(pts * 16).astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
        # pencil texture: hatch streaks + grain
        yy, xx = np.mgrid[0:2 * half, 0:2 * half].astype(np.float32)
        ph = rng.uniform(0, 6.28)
        ang = rng.uniform(0, np.pi)
        per = rng.uniform(0.35, 0.7) * PPM_HI
        hatch = 0.5 + 0.5 * np.sin(2 * np.pi * (xx * np.cos(ang) + yy * np.sin(ang)) / per + ph)
        grain = cv2.GaussianBlur(rng.random(m.shape, dtype=np.float32), (0, 0), 1.5)
        grain = (grain - grain.mean()) / (grain.std() + 1e-6)
        tex = np.clip(0.88 - 0.12 * hatch + 0.06 * grain, 0.55, 1.0)
        m = cv2.GaussianBlur(m, (0, 0), 0.6) * tex
        if kind == "erased":
            m *= cv2.GaussianBlur(rng.random(m.shape, dtype=np.float32), (0, 0), 6) * 2.2
    elif kind == "chord":  # half-filled bubble: a circular segment covering fraction f of the circle
        f = kw.get("fill", rng.uniform(0.35, 0.5))
        lo_t, hi_t = -1.0, 1.0
        for _ in range(40):
            t = 0.5 * (lo_t + hi_t)
            seg = (np.arccos(t) - t * np.sqrt(1 - t * t)) / np.pi
            lo_t, hi_t = (t, hi_t) if seg > f else (lo_t, t)
        t = 0.5 * (lo_t + hi_t)
        a = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:2 * half, 0:2 * half].astype(np.float32)
        dxp, dyp = (xx - ox) / R, (yy - oy) / R
        inside = (dxp ** 2 + dyp ** 2 <= rng.uniform(0.9, 1.05) ** 2) & (dxp * np.cos(a) + dyp * np.sin(a) > t)
        m = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), 0.8) * rng.uniform(0.85, 1.0)
    elif kind == "stray":
        ang = rng.uniform(0, np.pi)
        L = rng.uniform(1.2, 2.8) * PPM_HI
        c = rng.uniform(-0.8, 0.8, 2) * PPM_HI
        p1 = (ox + c[0] - L / 2 * np.cos(ang), oy + c[1] - L / 2 * np.sin(ang))
        p2 = (ox + c[0] + L / 2 * np.cos(ang), oy + c[1] + L / 2 * np.sin(ang))
        cv2.line(m, tuple(np.round(np.array(p1) * 16).astype(int)), tuple(np.round(np.array(p2) * 16).astype(int)), 1.0,
                 int(rng.uniform(0.35, 0.7) * PPM_HI), cv2.LINE_AA, shift=4)
        m = cv2.GaussianBlur(m, (0, 0), 0.7)
    elif kind == "dot":
        c = rng.uniform(-1.0, 1.0, 2) * PPM_HI
        cv2.circle(m, (int(round(ox + c[0])), int(round(oy + c[1]))), int(rng.uniform(0.5, 0.8) * PPM_HI), 1.0, -1, cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), 0.7)
    # ink gray level
    if "gray" in kw:
        g = kw["gray"]
    elif kind == "fill":
        g = rng.uniform(20, 60) if rng.random() < 0.35 else rng.uniform(55, 120)   # pen / pencil
    elif kind == "chord":
        g = rng.uniform(40, 110)
    elif kind == "erased":
        g = rng.uniform(190, 232)
    elif kind == "dot":
        g = rng.uniform(40, 110)
    else:
        g = rng.uniform(110, 190)
    patch = img[y0:y0 + 2 * half, x0:x0 + 2 * half].astype(np.float32)
    patch *= 1.0 - m * (1.0 - g / 255.0)
    img[y0:y0 + 2 * half, x0:x0 + 2 * half] = np.clip(patch, 0, 255).astype(np.uint8)


# ================================================================= marking a sheet
SPARSE = ("blank", "ghosts", "few_ghosts", "few_light")


def _sparse_plan(scenario, n, rng):
    """Per question: 'blank' | 'ghost' | 'real' | 'real_ghost'."""
    plan = ["blank"] * n
    if scenario in ("ghosts", "few_ghosts"):
        for i in range(0, n, 4):
            plan[i] = "ghost"
    if scenario in ("few_ghosts", "few_light"):
        k = 3 if scenario == "few_ghosts" else 5
        for i in rng.choice(n, size=min(k, n), replace=False):
            plan[i] = "real_ghost" if plan[i] == "ghost" or (scenario == "few_ghosts" and rng.random() < 0.5) else "real"
    return plan


def _draw_sparse(img, q, R, nc, kind, scenario, rng):
    a = int(rng.integers(nc))
    other = (a + 1 + int(rng.integers(nc - 1))) % nc
    t = {"number": q.number}
    if kind in ("real", "real_ghost"):
        kw = {"gray": rng.uniform(125, 150)} if scenario == "few_light" else {}
        draw_ink(img, q.bubbles[a].cx, q.bubbles[a].cy, R, rng, "fill", **kw)
        if kind == "real_ghost":
            draw_ink(img, q.bubbles[other].cx, q.bubbles[other].cy, R, rng, "erased")
        t.update(kind="erased_clear" if kind == "real_ghost" else "clear", expect="single", answer=a)
    elif kind == "ghost":
        draw_ink(img, q.bubbles[a].cx, q.bubbles[a].cy, R, rng, "erased")
        t.update(kind="erased_blank", expect="blank", answer=None)
    else:
        t.update(kind="blank", expect="blank", answer=None)
    return t

HARD_KINDS = ("multi", "stray_blank", "stray_clear", "erased_blank", "erased_clear", "ambiguous")


def mark_sheet(raster: np.ndarray, layout: Layout, page: int, rng, hard_rate: float = 0.2,
               id_hard: bool = True, light_pencil: bool = False, scenario: Optional[str] = None,
               fixed: Optional[dict] = None):
    """Return (marked raster, truth dict). hard_rate = fraction of questions that get a hard case.

    ``fixed`` = {"student_id": "12345678" | None (leave the ID blank), "answers": {question_number: choice | None},
    "subject": "07" | None (leave the subject code blank) - optional, default blank-free random clear code}
    draws exactly those clear marks (used by the end-to-end tests to build multi-page students)."""
    img = raster.copy()
    pg = layout.pages[page]
    R = pg.bubble_diameter / 2
    nc = layout.template.choices
    qtruth = []
    allq = [q for col in pg.columns for q in col.questions]
    plan = _sparse_plan(scenario, len(allq), rng) if scenario else None
    for col in pg.columns:
        for q in col.questions:
            if fixed is not None:
                a = fixed["answers"].get(q.number)
                if a is None:
                    qtruth.append({"number": q.number, "kind": "blank", "expect": "blank", "answer": None})
                else:
                    draw_ink(img, q.bubbles[a].cx, q.bubbles[a].cy, R, rng, "fill")
                    qtruth.append({"number": q.number, "kind": "clear", "expect": "single", "answer": a})
                continue
            if plan is not None:
                t = _draw_sparse(img, q, R, nc, plan[q.number - allq[0].number], scenario, rng)
                qtruth.append(t)
                continue
            u = rng.random()
            a = int(rng.integers(nc))
            other = int(rng.choice([i for i in range(nc) if i != a]))
            kw = {"gray": rng.uniform(110, 150)} if light_pencil else {}
            t = {"number": q.number}
            if u >= hard_rate:
                if rng.random() < 0.06:
                    t.update(kind="blank", expect="blank", answer=None)
                else:
                    draw_ink(img, q.bubbles[a].cx, q.bubbles[a].cy, R, rng, "fill", **kw)
                    t.update(kind="clear", expect="single", answer=a)
            else:
                k = HARD_KINDS[int(rng.integers(len(HARD_KINDS)))]
                if k == "multi":
                    draw_ink(img, q.bubbles[a].cx, q.bubbles[a].cy, R, rng, "fill", fill=rng.uniform(0.8, 1.0))
                    draw_ink(img, q.bubbles[other].cx, q.bubbles[other].cy, R, rng, "fill", fill=rng.uniform(0.8, 1.0))
                    t.update(kind=k, expect="multi", answer=sorted([a, other]))
                elif k in ("stray_blank", "erased_blank"):
                    draw_ink(img, q.bubbles[a].cx, q.bubbles[a].cy, R, rng,
                             rng.choice(["stray", "dot"]) if k == "stray_blank" else "erased")
                    t.update(kind=k, expect="blank", answer=None)
                elif k in ("stray_clear", "erased_clear"):
                    draw_ink(img, q.bubbles[a].cx, q.bubbles[a].cy, R, rng, "fill")
                    draw_ink(img, q.bubbles[other].cx, q.bubbles[other].cy, R, rng,
                             rng.choice(["stray", "dot"]) if k == "stray_clear" else "erased")
                    t.update(kind=k, expect="single", answer=a)
                else:  # ambiguous half-filled
                    draw_ink(img, q.bubbles[a].cx, q.bubbles[a].cy, R, rng, "chord")
                    t.update(kind="ambiguous", expect="flag", answer=a)
            qtruth.append(t)
    # ---- student id
    g = pg.id_grid
    nd = len(g.columns)
    digits = [int(x) for x in rng.integers(0, 10, nd)]
    if fixed is not None:
        id_hard = False
        if fixed["student_id"] is not None:
            digits = [int(ch) for ch in fixed["student_id"]]
    id_must_none = False
    id_hard_kind = None
    hard_col = None
    sparse_id_blank = (scenario is not None and rng.random() < 0.5) or (fixed is not None and fixed["student_id"] is None)   # sparse page: ID often left blank too
    if id_hard and rng.random() < 0.25:
        id_hard_kind = str(rng.choice(["double", "blank", "stray", "erased", "half"]))
        hard_col = int(rng.integers(nd))
    for ci, col in enumerate(g.columns):
        d = digits[ci]
        if sparse_id_blank:
            id_must_none = True
            continue
        if ci == hard_col and id_hard_kind == "blank":
            id_must_none = True
            continue
        if ci == hard_col and id_hard_kind == "half":
            draw_ink(img, col.bubbles[d].cx, col.bubbles[d].cy, g.bubble_diameter / 2, rng, "chord")
            id_must_none = True
            continue
        draw_ink(img, col.bubbles[d].cx, col.bubbles[d].cy, g.bubble_diameter / 2, rng, "fill",
                 **({"gray": rng.uniform(110, 150)} if light_pencil else {}))
        if ci == hard_col:
            o = int(rng.choice([v for v in range(10) if v != d]))
            if id_hard_kind == "double":
                draw_ink(img, col.bubbles[o].cx, col.bubbles[o].cy, g.bubble_diameter / 2, rng, "fill")
                id_must_none = True
            elif id_hard_kind == "stray":
                draw_ink(img, col.bubbles[o].cx, col.bubbles[o].cy, g.bubble_diameter / 2, rng, rng.choice(["stray", "dot"]))
            else:
                draw_ink(img, col.bubbles[o].cx, col.bubbles[o].cy, g.bubble_diameter / 2, rng, "erased")
    # a little handwriting-like scribble in the name box
    x, y, w, h = pg.name_box
    for _ in range(int(rng.integers(3, 7))):
        pts = np.array([[(x + rng.uniform(3, w - 3)) * PPM_HI, (y + rng.uniform(3, h - 3)) * PPM_HI] for _ in range(4)])
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, int(rng.uniform(30, 90)), int(0.5 * PPM_HI), cv2.LINE_AA)
    subj, subj_hard, subj_digits, subj_col = _mark_subject(img, pg, rng, fixed, id_hard and fixed is None, light_pencil)
    truth = {"questions": qtruth, "student_id": None if id_must_none else "".join(map(str, digits)),
             "id_digits": digits, "id_hard": id_hard_kind, "id_must_none": id_must_none, "id_hard_col": hard_col,
             "subject": subj, "subject_digits": subj_digits, "subject_hard": subj_hard, "subject_hard_col": subj_col}
    return img, truth


def _mark_subject(img, pg, rng, fixed, hard: bool, light_pencil: bool):
    """Mark the subject code (drawn AFTER everything else so the rng stream of the rest is unchanged).

    Returns (truth subject, hard kind, digits): the subject is "" when the form has no subject code and None when the
    marks are such that it must stay unread (blank / double / half-filled digit)."""
    g = pg.subject_grid
    if g is None:
        return "", None, [], None
    nd = len(g.columns)
    digits = [int(x) for x in rng.integers(0, 10, nd)]
    must_none, kind, hard_col = False, None, None
    if fixed is not None and "subject" in fixed:
        if fixed["subject"] is None:
            return None, "blank", digits, None            # nothing drawn
        digits = [int(ch) for ch in fixed["subject"]]
    elif hard and rng.random() < 0.3:
        kind = str(rng.choice(["double", "blank", "stray", "erased", "half"]))
        hard_col = int(rng.integers(nd))
    R = g.bubble_diameter / 2
    for ci, col in enumerate(g.columns):
        d = digits[ci]
        if ci == hard_col and kind == "blank":
            must_none = True
            continue
        if ci == hard_col and kind == "half":
            draw_ink(img, col.bubbles[d].cx, col.bubbles[d].cy, R, rng, "chord")
            must_none = True
            continue
        draw_ink(img, col.bubbles[d].cx, col.bubbles[d].cy, R, rng, "fill",
                 **({"gray": rng.uniform(110, 150)} if light_pencil else {}))
        if ci == hard_col:
            o = int(rng.choice([v for v in range(10) if v != d]))
            if kind == "double":
                draw_ink(img, col.bubbles[o].cx, col.bubbles[o].cy, R, rng, "fill")
                must_none = True
            elif kind == "stray":
                draw_ink(img, col.bubbles[o].cx, col.bubbles[o].cy, R, rng, rng.choice(["stray", "dot"]))
            else:
                draw_ink(img, col.bubbles[o].cx, col.bubbles[o].cy, R, rng, "erased")
    return (None if must_none else "".join(map(str, digits))), kind, digits, hard_col


# ================================================================= camera
@dataclass
class Cond:
    name: str
    frame: Tuple[int, int] = (1920, 1080)      # (W, H)
    ppm: Tuple[float, float] = (2.7, 3.5)
    tilt: float = 20.0                         # degrees, max
    rot: str = "any"                           # "any" | "upright" (0/180 +- 15) | "none"
    defocus: Tuple[float, float] = (0.6, 1.2)  # gaussian sigma in output px
    motion: Tuple[float, float] = (0.0, 0.0)   # blur length px (applied with p=0.5)
    gain: Tuple[float, float] = (0.8, 1.0)
    gradient: float = 0.25                     # max brightness falloff across the frame
    shadow: float = 0.3                        # max shadow band depth
    noise: Tuple[float, float] = (2.0, 4.0)    # gray levels (sigma), before JPEG
    hard_rate: float = 0.2
    bend: float = 0.5                          # paper curl / lens: smooth displacement amplitude (mm)
    light_pencil: bool = False
    scenario: Optional[str] = None             # sparse-page scenario (see SPARSE)
    printer: bool = False                      # printer defects: grey black, white streaks, bands (print_defects)
    occlude: int = 0                           # corner markers hidden by the holding hand (0..2)
    # ---- physical paper shape and light (see paper_surface / surface_light)
    curl: Tuple[float, float] = (0.0, 0.0)     # sagitta (mm) of a cylindrical bow over the whole sheet
    edge_curl: float = 0.0                     # probability of an extra tight curl along one edge (held / stacked sheet)
    raking: float = 0.0                        # 0 = light from the camera side; 1 = grazing light (shading follows the curl)
    glare: Tuple[float, float] = (0.0, 0.0)    # peak specular highlight (fraction of white added), applied with p=0.8
    hard_shadow: float = 0.0                   # depth of a sharp-edged shadow (phone / hand) over part of the sheet


CONDS: Dict[str, Cond] = {
    "clean": Cond("clean", ppm=(3.2, 3.5), tilt=3, rot="none", defocus=(0.5, 0.8), gain=(0.95, 1.0),
                  gradient=0.05, shadow=0.0, noise=(1, 2), bend=0.1),
    "tilted": Cond("tilted", tilt=25, rot="any"),
    "blurred": Cond("blurred", tilt=15, defocus=(1.3, 1.9), motion=(4, 8), noise=(2, 4)),
    "dark": Cond("dark", tilt=15, gain=(0.28, 0.45), noise=(3, 6), shadow=0.35),
    "phone": Cond("phone", frame=(1080, 1920), tilt=20),
    "lowres": Cond("lowres", ppm=(2.0, 2.4), tilt=20, defocus=(0.8, 1.3)),
    # sparse pages: almost no real marks, so the page cannot calibrate "ink" from its own marks
    "sp_blank": Cond("sp_blank", scenario="blank"),
    "sp_ghosts": Cond("sp_ghosts", scenario="ghosts"),
    "sp_fewghost": Cond("sp_fewghost", scenario="few_ghosts"),
    "sp_pencil": Cond("sp_pencil", scenario="few_light"),
    "sp_dark": Cond("sp_dark", scenario="few_ghosts", gain=(0.28, 0.45), noise=(3, 6), shadow=0.35),
    # print defects (grey black + streaks + bands) and a thumb over one corner marker: normal handheld conditions
    "printer": Cond("printer", frame=(1080, 1920), tilt=15, rot="upright", printer=True),
    "handheld": Cond("handheld", frame=(1080, 1920), tilt=15, rot="upright", printer=True, occlude=1),
    "hand_only": Cond("hand_only", tilt=20, occlude=1),
    # paper shape / light: a bowed or edge-curled sheet, light from the side, glare, a sharp shadow
    "curl": Cond("curl", frame=(1080, 1920), tilt=15, rot="upright", printer=True, curl=(4, 14), edge_curl=0.5,
                 raking=0.4),
    "curl_hand": Cond("curl_hand", frame=(1080, 1920), tilt=15, rot="upright", printer=True, occlude=1,
                      curl=(4, 14), edge_curl=0.5, raking=0.4),
    "glare": Cond("glare", frame=(1080, 1920), tilt=15, rot="upright", printer=True, curl=(2, 8),
                  glare=(0.25, 0.6)),
    "shadow": Cond("shadow", frame=(1080, 1920), tilt=15, rot="upright", printer=True, curl=(0, 6),
                   hard_shadow=0.55),
    "raking": Cond("raking", frame=(1080, 1920), tilt=15, rot="upright", printer=True, curl=(6, 16), edge_curl=0.3,
                   raking=1.0, gain=(0.5, 0.9)),
}
DEFECT_CONDS = ["printer", "handheld", "hand_only"]
PAPER_CONDS = ["curl", "curl_hand", "glare", "shadow", "raking"]
MAIN_CONDS = ["clean", "tilted", "blurred", "dark", "phone", "lowres"]


def _rot_matrix(ax, ay, az):
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def _lowfreq(rng, h, w, scale):
    n = rng.random((max(2, h // scale), max(2, w // scale)), dtype=np.float32)
    return cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)


def background(rng, W, H) -> np.ndarray:
    """Cluttered desk-like background, float32 BGR 0..255."""
    base = np.array(rng.uniform(40, 200, 3), np.float32)
    bg = np.empty((H, W, 3), np.float32)
    tex = 0.6 * _lowfreq(rng, H, W, 200) + 0.3 * _lowfreq(rng, H, W, 40) + 0.1 * _lowfreq(rng, H, W, 6)
    for c in range(3):
        bg[..., c] = base[c] * (0.65 + 0.7 * tex)
    img = np.clip(bg, 0, 255).astype(np.uint8)
    for _ in range(int(rng.integers(6, 18))):  # clutter: books, cables, tape, text
        col = tuple(int(v) for v in rng.integers(0, 255, 3))
        k = rng.integers(4)
        p = rng.integers(0, [W, H], 2)
        if k == 0:
            q = p + rng.integers(30, 280, 2)
            cv2.rectangle(img, tuple(p), tuple(q), col, int(rng.choice([-1, 2, 3])))
        elif k == 1:
            cv2.line(img, tuple(p), tuple(rng.integers(0, [W, H], 2)), col, int(rng.integers(2, 12)))
        elif k == 2:
            cv2.circle(img, tuple(p), int(rng.integers(10, 120)), col, int(rng.choice([-1, 3])))
        else:
            cv2.putText(img, "".join(rng.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789 "), 14)), tuple(p),
                        cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.6, 2.4)), col, int(rng.integers(1, 4)))
    # a few quads with thick dark borders (decoys for the marker detector)
    for _ in range(int(rng.integers(0, 5))):
        c = rng.integers(0, [W, H], 2)
        s = int(rng.integers(20, 90))
        a = rng.uniform(0, 3.14)
        box = cv2.boxPoints(((float(c[0]), float(c[1])), (s, s), float(np.degrees(a)))).astype(np.int32)
        cv2.fillPoly(img, [box], (15, 15, 15))
        inner = cv2.boxPoints(((float(c[0]), float(c[1])), (s * .6, s * .6), float(np.degrees(a)))).astype(np.int32)
        cv2.fillPoly(img, [inner], (235, 235, 235))
        for _ in range(3):
            cc = c + rng.integers(-s // 5, s // 5 + 1, 2)
            cv2.circle(img, tuple(int(v) for v in cc), max(2, s // 8), (20, 20, 20), -1)
    return img.astype(np.float32)


def _lighting(rng, W, H, cond: Cond) -> np.ndarray:
    """Smooth multiplicative illumination (gradient + soft shadow band), computed at 1/8 resolution."""
    sc = 8
    w, h = W // sc, H // sc
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx, yy = xx * sc, yy * sc
    a = rng.uniform(0, 2 * np.pi)
    d = (xx - W / 2) * np.cos(a) + (yy - H / 2) * np.sin(a)
    d = d / (np.abs(d).max() + 1e-6)
    L = 1.0 - cond.gradient * (0.5 + 0.5 * d) * rng.uniform(0.5, 1.0)
    if cond.shadow > 0 and rng.random() < 0.8:
        b = rng.uniform(0, 2 * np.pi)
        n = (xx - W / 2) * np.cos(b) + (yy - H / 2) * np.sin(b)
        c = rng.uniform(-0.3, 0.3) * max(W, H)
        wid = rng.uniform(0.05, 0.2) * max(W, H)
        soft = rng.uniform(15, 70)
        band = 1.0 / (1.0 + np.exp((np.abs(n - c) - wid) / soft * 3))
        L = L * (1.0 - cond.shadow * rng.uniform(0.5, 1.0) * band)
    L = cv2.resize((L * rng.uniform(*cond.gain)).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    return L


def _camera_degrade(rng, frame: np.ndarray, cond: Cond, jpeg_q=70) -> np.ndarray:
    H, W = frame.shape[:2]
    frame = frame * _lighting(rng, W, H, cond)[..., None]
    s = rng.uniform(*cond.defocus)
    frame = cv2.GaussianBlur(frame, (0, 0), s)
    if cond.motion[1] > 0 and rng.random() < 0.5:
        L = int(round(rng.uniform(*cond.motion)))
        k = np.zeros((L, L), np.float32)
        cv2.line(k, (0, L // 2), (L - 1, L // 2), 1.0)
        M = cv2.getRotationMatrix2D(((L - 1) / 2, (L - 1) / 2), rng.uniform(0, 180), 1.0)
        k = cv2.warpAffine(k, M, (L, L))
        frame = cv2.filter2D(frame, -1, k / max(k.sum(), 1e-6))
    sig = rng.uniform(*cond.noise)
    frame = frame + rng.standard_normal(frame.shape, dtype=np.float32) * (sig * (0.6 + 0.4 * frame / 255.0))
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _homography(rng, cond: Cond, W, H, mode: str):
    """mm (page coords) -> frame px. Returns (3x3 H, scale px/mm)."""
    f = rng.uniform(900, 1500)
    s = rng.uniform(*cond.ppm)
    tilt = np.radians(cond.tilt) * np.sqrt(rng.random()) if cond.tilt > 0 else 0.0
    ta = rng.uniform(0, 2 * np.pi)
    ax, ay = tilt * np.cos(ta), tilt * np.sin(ta)
    if cond.rot == "any":
        az = rng.uniform(0, 2 * np.pi) if rng.random() < 0.6 else rng.choice([0, np.pi]) + rng.uniform(-0.3, 0.3)
    elif cond.rot == "upright":
        az = rng.choice([0, np.pi]) + rng.uniform(-0.25, 0.25)
    else:
        az = 0.0
    R = _rot_matrix(ax, ay, az)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    corners = np.array([[0, 0], [210, 0], [210, 297], [0, 297]], float) - [105, 148.5]

    def build(s_, cx, cy):
        d = f / s_
        t = np.array([cx, cy, d])
        M = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ np.array([[1, 0, -105], [0, 1, -148.5], [0, 0, 1.0]])
        return M

    def bbox(M):
        q = cv2.perspectiveTransform(np.array([[[0, 0], [210, 0], [210, 297], [0, 297]]], np.float64), M)[0]
        return q.min(0), q.max(0)

    M = build(s, 0, 0)
    lo, hi = bbox(M)
    ext = hi - lo
    # fit the sheet inside the frame (shrinking the scale if needed) unless the sheet is meant to be cut off
    margin = 0.97
    while mode != "cut" and (ext[0] > W * margin or ext[1] > H * margin) and s > 1.4:
        s *= 0.97
        M = build(s, 0, 0)
        lo, hi = bbox(M)
        ext = hi - lo
    # translate so the bbox lands at a random legal place
    if mode == "cut":
        side = int(rng.integers(4))
        tgt_lo = np.array([rng.uniform(*sorted((0, W - ext[0]))), rng.uniform(*sorted((0, H - ext[1])))])
        frac = rng.uniform(0.18, 0.4)
        if side == 0:
            tgt_lo[0] = -frac * ext[0]
        elif side == 1:
            tgt_lo[0] = W - (1 - frac) * ext[0]
        elif side == 2:
            tgt_lo[1] = -frac * ext[1]
        else:
            tgt_lo[1] = H - (1 - frac) * ext[1]
    else:
        tgt_lo = np.array([rng.uniform(W * (1 - margin) / 2, max(W * (1 + margin) / 2 - ext[0], W * (1 - margin) / 2 + 1)),
                           rng.uniform(H * (1 - margin) / 2, max(H * (1 + margin) / 2 - ext[1], H * (1 - margin) / 2 + 1))])
    T = np.array([[1, 0, tgt_lo[0] - lo[0]], [0, 1, tgt_lo[1] - lo[1]], [0, 0, 1.0]])
    M = T @ M
    # effective px/mm at the sheet centre
    c = cv2.perspectiveTransform(np.array([[[105, 148.5], [106, 148.5], [105, 149.5]]], np.float64), M)[0]
    eff = float(np.sqrt(np.linalg.norm(c[1] - c[0]) * np.linalg.norm(c[2] - c[0])))
    cam = -R.T @ np.array([0.0, 0.0, f / s]) + np.array([105.0, 148.5, 0.0])   # camera centre in page mm (z = height)
    return M, eff, cam


# ================================================================= paper shape and light
def paper_surface(rng, cond: Cond, step: float = 2.0):
    """3D shape of a (nearly inextensible) bent sheet on a grid of flat page points.

    Returns (gx, gy, X, N): grid of flat page coordinates (mm), their 3D positions (page frame: x, y in mm, z = height
    above the flat page plane) and unit normals.  A cylindrical bow over the whole sheet (sagitta cond.curl) plus,
    with probability cond.edge_curl, a tight curl along one edge (the part a hand holds, or a sheet off a stack)."""
    gx, gy = np.meshgrid(np.arange(0, 210 + step, step), np.arange(0, 297 + step, step))
    X = np.stack([gx, gy, np.zeros_like(gx)], -1).astype(np.float64)
    if cond.curl[1] > 0:
        across_x = rng.random() < 0.5                      # bow across the width (axis vertical) or the length
        th = (0.0 if across_x else np.pi / 2) + rng.uniform(-0.25, 0.25)
        n = np.array([np.cos(th), np.sin(th)])
        span = abs(210 * n[0]) + abs(297 * n[1])
        sag = rng.uniform(*cond.curl) * rng.choice([-1, 1])
        if abs(sag) > 0.05:
            Rc = span ** 2 / (8 * abs(sag))
            u = (gx - 105 - rng.uniform(-20, 20)) * n[0] + (gy - 148.5 - rng.uniform(-20, 20)) * n[1]
            du = Rc * np.sin(u / Rc) - u
            X[..., 0] += du * n[0]
            X[..., 1] += du * n[1]
            X[..., 2] += np.sign(sag) * Rc * (1 - np.cos(u / Rc))
    if cond.edge_curl > 0 and rng.random() < cond.edge_curl:
        side = int(rng.integers(4))                        # 0 left, 1 right, 2 top, 3 bottom edge
        dist = [gx, 210 - gx, gy, 297 - gy][side]          # distance from that edge
        m = [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])][side]   # inward
        v0 = rng.uniform(25, 70)
        Rc2 = rng.uniform(45, 130)
        w = np.clip(v0 - dist, 0, None)                    # how far beyond the start of the curl
        X[..., 0] += (w - Rc2 * np.sin(w / Rc2)) * m[0]
        X[..., 1] += (w - Rc2 * np.sin(w / Rc2)) * m[1]
        X[..., 2] += rng.choice([-1, 1]) * Rc2 * (1 - np.cos(w / Rc2))
    dX_dx = np.gradient(X, step, axis=1)
    dX_dy = np.gradient(X, step, axis=0)
    N = np.cross(dX_dx, dX_dy)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    return gx, gy, X, N


def paper_in_space(small: np.ndarray, ppm: float, cam: np.ndarray, cond: Cond, rng, truth: dict) -> np.ndarray:
    """Bend the paper raster in 3D, light it, and flatten it again *as the camera sees it*: every surface point is moved
    to where its camera ray meets the flat page plane, so the frame's (planar) homography then produces the exact
    perspective image of the bent sheet.  Also: Lambertian shading by the surface normal (light from the camera side
    or grazing, cond.raking), a specular highlight that lifts paper and glossy ink alike (cond.glare; saturates the
    sensor where it is strong) and a sharp-edged cast shadow (cond.hard_shadow).  ``small`` is a gray raster at
    ``ppm`` px/mm; returns (float32 raster - may exceed 255 under glare, clipped by the camera later - , coverage)."""
    step = 2.0
    gx, gy, X, N = paper_surface(rng, cond, step)
    C = np.asarray(cam, np.float64)
    if C[2] < 0:                                            # look from the +z side
        C = C * np.array([1, 1, -1])
    s = C[2] / (C[2] - X[..., 2])
    qx = C[0] + s * (X[..., 0] - C[0])
    qy = C[1] + s * (X[..., 1] - C[1])
    Dx, Dy = (qx - gx).astype(np.float32), (qy - gy).astype(np.float32)
    truth["curl_max_mm"] = float(np.sqrt(Dx ** 2 + Dy ** 2).max())
    # light (all on the grid, in flat page coordinates)
    view = C - X
    view /= np.linalg.norm(view, axis=-1, keepdims=True)
    az = rng.uniform(0, 2 * np.pi)
    el = np.radians(rng.uniform(12, 30))
    graze = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    Lc = view[view.shape[0] // 2, view.shape[1] // 2]
    L = (1 - cond.raking) * Lc + cond.raking * graze
    L /= np.linalg.norm(L)
    amb = 0.3
    lam = np.clip((N * L).sum(-1), 0, None)
    shade = amb + (1 - amb) * lam / max(float(lam.max()), 1e-6)
    if cond.hard_shadow > 0:
        a = rng.uniform(0, 2 * np.pi)
        d = (gx - 105) * np.cos(a) + (gy - 148.5) * np.sin(a)
        c0 = rng.uniform(0.05, 0.45) * (abs(210 * np.cos(a)) + abs(297 * np.sin(a)))
        wob = rng.uniform(2, 10) * np.sin((-(gx - 105) * np.sin(a) + (gy - 148.5) * np.cos(a)) / rng.uniform(25, 80) + rng.uniform(0, 6.3))
        soft = rng.uniform(0.4, 2.0)
        inside = 1 / (1 + np.exp(-(d - c0 + wob) / soft))
        shade = shade * (1 - cond.hard_shadow * rng.uniform(0.7, 1.0) * inside)
    glare = np.zeros_like(gx, np.float32)
    if cond.glare[1] > 0 and rng.random() < 0.8:
        # specular highlight where the mirror direction of the light meets the camera
        Lg = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 1.0])
        Lg /= np.linalg.norm(Lg)
        Rm = 2 * (N * Lg).sum(-1, keepdims=True) * N - Lg
        spec = np.clip((Rm * view).sum(-1), 0, 1) ** rng.uniform(150, 600)
        # the lamp is not a point: widen, then scale so the peak is in the requested range (a big lamp off-sheet
        # may still leave only a weak highlight on the sheet)
        spec = cv2.GaussianBlur(spec.astype(np.float32), (0, 0), rng.uniform(2, 6))
        if spec.max() < 1e-3:                               # highlight off the sheet: put it somewhere on it
            cx, cy = rng.uniform(30, 180), rng.uniform(30, 267)
            spec = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * rng.uniform(12, 35) ** 2)).astype(np.float32)
        glare = (spec / spec.max() * rng.uniform(*cond.glare)).astype(np.float32)
    truth["glare_peak"] = float(glare.max())
    # inverse map: for each flat-plane pixel q find the paper point p with p + D(p) = q (fixed point; D is smooth)
    h, w = small.shape
    up = lambda a: cv2.resize(a.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    Dxp, Dyp = up(Dx) * ppm, up(Dy) * ppm
    qy_, qx_ = np.mgrid[0:h, 0:w].astype(np.float32)
    px_, py_ = qx_.copy(), qy_.copy()
    for _ in range(6):
        px_ = qx_ - cv2.remap(Dxp, px_, py_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        py_ = qy_ - cv2.remap(Dyp, px_, py_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    paper = cv2.remap(small.astype(np.float32), px_, py_, cv2.INTER_LINEAR, borderValue=0)
    sh = cv2.remap(up(shade), px_, py_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    gl = cv2.remap(up(glare), px_, py_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    inkness = 1.0 - paper / 255.0
    out = paper * sh + 255.0 * gl * (1.0 + 0.6 * inkness)     # graphite / toner is glossier than paper
    cover = ((px_ >= 0) & (py_ >= 0) & (px_ <= w - 1) & (py_ <= h - 1)).astype(np.uint8) * 255
    return out, cover      # cover: where the bent sheet still is (an edge curled towards the camera leaves a gap)


def hand_blob(frame: np.ndarray, quads: List[np.ndarray], rng, strong: bool = False) -> None:
    """Skin-coloured blob over each given marker quad, centred a little outside the marker (a thumb holding the
    sheet's edge) and covering it completely, with little spill into the page."""
    H, W = frame.shape[:2]
    for q in quads:
        c = q.mean(0)
        side = float(np.linalg.norm(q[1] - q[0]))
        out = c - np.array([W / 2, H / 2])
        out = out / (np.linalg.norm(out) + 1e-6)
        scale = side / lay.MARKER_MM                         # px per mm at the marker
        c = c + out * 9.0 * scale
        ax = (rng.uniform(19, 22) if not strong else rng.uniform(26, 34)) * scale
        ay = (rng.uniform(19, 22) if not strong else rng.uniform(26, 34)) * scale
        th = np.zeros((H, W), np.float32)
        cv2.ellipse(th, (int(c[0]), int(c[1])), (int(ax), int(ay)), float(rng.uniform(0, 180)), 0, 360, 1.0, -1, cv2.LINE_AA)
        th = cv2.GaussianBlur(th, (0, 0), 3)
        skin = np.array([rng.uniform(110, 150), rng.uniform(140, 175), rng.uniform(190, 230)], np.float32)
        frame[:] = frame * (1 - th[..., None]) + skin * th[..., None]


def draw_serial(img: np.ndarray, serial: int) -> None:
    """Print the serial strip onto a 300 dpi paper raster (same cells the PDF renderer uses)."""
    from omr import markers as mk
    for (x, y, w, h), bit in zip(lay.serial_cell_rects(), mk.encode_serial(serial)):
        if bit:
            img[int(round(y * PPM_HI)):int(round((y + h) * PPM_HI)), int(round(x * PPM_HI)):int(round((x + w) * PPM_HI))] = 0


def make_frame(layout: Layout, pdf_path: str, cond: Cond, rng, page: Optional[int] = None, mode: str = "sheet",
               fixed: Optional[dict] = None, serial: Optional[int] = None):
    """mode: sheet | none | thumb (one marker hidden: a normal condition) | thumb2 (two hidden: negative) | cut.  Returns (frame_bgr, truth).  ``serial``: print this copy number
    (None / 0 = nothing printed, like the plain master PDF)."""
    W, H = cond.frame
    bg = background(rng, W, H)
    truth: dict = {"mode": mode, "cond": cond.name, "form_key": layout.form_key}
    if mode == "none":
        if rng.random() < 0.5:  # a page of plain text, no markers
            bg = bg.astype(np.uint8)
            x = rng.uniform(0.2, 0.6) * W
            cv2.rectangle(bg, (int(x), int(0.1 * H)), (int(x + 0.3 * W), int(0.9 * H)), (235, 235, 235), -1)
            for i in range(int(rng.integers(10, 30))):
                cv2.putText(bg, "lorem ipsum dolor sit amet", (int(x + 10), int(0.13 * H + i * 0.025 * H)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        truth["expect_status"] = ["no_sheet"]
        return _camera_degrade(rng, bg.astype(np.float32), cond), truth
    page = int(rng.integers(layout.total_pages)) if page is None else page
    raster = paper_raster(pdf_path, page)
    if serial or cond.printer:
        raster = raster.copy()
        if serial:
            draw_serial(raster, serial)                 # printed, like everything else on the raster
        if cond.printer:
            raster = print_defects(raster, page, rng)
    marked, tr = mark_sheet(raster, layout, page, rng, cond.hard_rate, light_pencil=cond.light_pencil,
                            scenario=cond.scenario, fixed=fixed)
    truth.update(tr, page=page, total_pages=layout.total_pages, serial=serial or 0)
    M, eff, cam = _homography(rng, cond, W, H, mode)
    truth["px_per_mm"] = eff
    # pre-shrink the raster to ~2x the final scale, then warp
    pre = min(1.0, 2.0 * eff / PPM_HI)
    small = cv2.resize(marked, None, fx=pre, fy=pre, interpolation=cv2.INTER_AREA) if pre < 1 else marked
    if cond.bend > 0:  # smooth displacement field (paper bow, lens) applied to the paper before projection
        h_, w_ = small.shape
        amp = cond.bend * PPM_HI * pre * rng.uniform(0.3, 1.0)
        fx = (_lowfreq(rng, h_, w_, max(h_ // 3, 8)) - 0.5) * 2 * amp
        fy = (_lowfreq(rng, h_, w_, max(h_ // 3, 8)) - 0.5) * 2 * amp
        gy, gx = np.mgrid[0:h_, 0:w_].astype(np.float32)
        small = cv2.remap(small, gx + fx, gy + fy, cv2.INTER_LINEAR, borderValue=255)
    if cond.curl[1] > 0 or cond.glare[1] > 0 or cond.hard_shadow > 0 or cond.raking > 0:
        small, cover = paper_in_space(small, PPM_HI * pre, cam, cond, rng, truth)
    else:
        cover = np.full(small.shape, 255, np.uint8)
    S = np.diag([1 / (PPM_HI * pre), 1 / (PPM_HI * pre), 1.0])
    Hp = M @ S
    sheet = cv2.warpPerspective(small, Hp, (W, H), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32)
    mask = cv2.warpPerspective(cover, Hp, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    tint = np.array(rng.uniform(0.9, 1.0, 3), np.float32)
    sheet3 = sheet[..., None] * tint[None, None, :]
    frame = bg * (1 - mask[..., None]) + sheet3 * mask[..., None]
    # marker corners in the frame (for ground truth / diagnostics)
    mc = {}
    for m in layout.pages[page].markers:
        x, y, w, h = m.rect
        q = cv2.perspectiveTransform(np.array([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]], np.float64), M)[0]
        mc[m.id] = q.tolist()
    truth["marker_corners"] = mc
    hide = {"thumb": 1, "thumb2": 2}.get(mode, cond.occlude if mode == "sheet" else 0)
    if hide:
        which = [int(i) for i in rng.choice(4, size=hide, replace=False)]
        keys = list(mc.values())
        if hide == 2 and rng.random() < 0.5:         # a hand holding the sheet's side covers two neighbours
            first = int(rng.integers(4))
            which = [first, (first + 1) % 4]
        hand_blob(frame, [np.array(keys[k]) for k in which], rng, strong=hide == 2)
        truth["hidden"] = which
        truth["expect_status"] = (["partial", "no_sheet", "bad_geometry", "inconsistent"] if hide == 2 else ["ok"])
    if hide:
        pass
    elif mode == "cut":
        truth["expect_status"] = ["partial", "no_sheet", "inconsistent", "bad_geometry", "bad_meta"]
        # (a cut-off frame may still happen to contain all four markers; then a correct read is also fine)
        inside = all(0 <= p[0] < W and 0 <= p[1] < H for q in mc.values() for p in q)
        truth["all_markers_inside"] = inside
        # three corners in view is enough to read the sheet (like a hand over one marker)
        truth["n_inside"] = sum(all(0 <= p[0] < W and 0 <= p[1] < H for p in q) for q in mc.values())
    else:
        truth["expect_status"] = ["ok"]
    return _camera_degrade(rng, frame, cond), truth


# ================================================================= evaluation
def evaluate(res, truth: dict) -> dict:
    """Compare a RecognitionResult with truth. Returns counters (see bench/test)."""
    out = {"q_clear": 0, "q_clear_ok": 0, "q_clear_flagged": 0, "q_clear_wrong": 0,
           "q_silent": 0, "q_hard": 0, "q_hard_flagged": 0, "q_hard_ok": 0, "q_total": 0,
           "q_silent_clear": 0, "id_wrong": 0, "id_ok": 0, "id_none": 0, "id_total": 1,
           "id_readable": 0, "id_missed": 0, "serial_total": 0, "serial_ok": 0, "serial_wrong": 0,
           "subject_total": 0, "subject_ok": 0, "subject_wrong": 0, "subject_none": 0, "subject_missed": 0,
           "nosubj_ok": 0}
    out["kinds"] = {}
    out["failures"] = []
    if res.status != "ok":
        return out
    for tq, a in zip(truth["questions"], res.answers):
        assert tq["number"] == a.number
        exp = tq["expect"]
        if exp == "single":
            correct = a.state == "single" and a.marked == [tq["answer"]]
        elif exp == "blank":
            correct = a.state == "blank"
        elif exp == "multi":
            correct = a.state == "multi" and a.marked == tq["answer"]
        else:
            correct = False  # "flag": only acceptable when flagged
        out["q_total"] += 1
        silent = (not correct) and (not a.ambiguous)
        kd = out["kinds"].setdefault(tq["kind"], [0, 0, 0, 0])  # n, correct, flagged, silent
        kd[0] += 1
        kd[1] += bool(correct)
        kd[2] += bool(a.ambiguous)
        kd[3] += bool(silent)
        if not correct:
            out["failures"].append((tq["number"], tq["kind"], tq["expect"], tq["answer"], a.state, a.marked,
                                    a.ambiguous, a.scores))
        out["q_silent"] += silent
        if tq["kind"] == "clear" or tq["kind"] == "blank":
            out["q_clear"] += 1
            out["q_clear_ok"] += correct
            out["q_clear_flagged"] += a.ambiguous
            out["q_clear_wrong"] += not correct
            out["q_silent_clear"] += silent
        else:
            out["q_hard"] += 1
            out["q_hard_flagged"] += a.ambiguous
            out["q_hard_ok"] += correct
    if truth.get("serial"):
        out["serial_total"] += 1
        out["serial_ok"] += int(res.serial == truth["serial"])
        out["serial_wrong"] += int(res.serial not in (None, truth["serial"]))
    sid = truth["student_id"]
    out["id_readable"] = int(sid is not None)
    if res.student_id is None:
        out["id_none"] = 1
        out["id_missed"] = int(sid is not None)
    elif res.student_id == sid:
        out["id_ok"] = 1
    else:
        out["id_wrong"] = 1
    if "subject" in truth:
        subj = truth["subject"]
        if subj == "":                           # the form has no subject code: it must read as ""
            out["nosubj_ok" if res.subject == "" else "subject_wrong"] = 1
        else:
            out["subject_total"] = 1
            if res.subject is None:
                out["subject_none"] = 1
                out["subject_missed"] = int(subj is not None)
            elif res.subject == subj:
                out["subject_ok"] = 1
            else:
                out["subject_wrong"] = 1
    return out


def merge(tot: dict, ev: dict) -> None:
    for k, v in ev.items():
        if k == "kinds":
            for kk, vv in v.items():
                t = tot.setdefault("kinds", {}).setdefault(kk, [0, 0, 0, 0])
                for i in range(4):
                    t[i] += vv[i]
        elif k == "failures":
            tot.setdefault("failures", []).extend(v)
        else:
            tot[k] = tot.get(k, 0) + v


# ================================================================= batch runner (used by pytest and bench)
STANDARD_TEMPLATES = {
    "q45": lay.Template(45, 5, 8, title="Midterm"),
    "q100_2p": lay.Template(100, 5, 8, pages=2),
    "q100_3p": lay.Template(100, 5, 8, pages=3),
    "q30_abcd": lay.Template(30, 4, 6, choice_labels=list("ABCD")),
    "q100u80_2p": lay.Template(100, 5, 8, pages=2, used=80),      # 100-question form, 80 used
    "q40_nosubj": lay.Template(40, 5, 8, subject_digits=0),
}

_worker_state: dict = {}


def prepare(workdir: str, templates: Optional[dict] = None):
    """Build layouts + PDFs once.  Returns {name: (Layout, pdf_path)}."""
    os.makedirs(workdir, exist_ok=True)
    out = {}
    for name, t in (templates or STANDARD_TEMPLATES).items():
        L = lay.build_layout(t)
        out[name] = (L, build_pdf(L, os.path.join(workdir, f"{name}_{L.form_key}.pdf")))
    return out


def mk_serial_max() -> int:
    from omr import markers as mk
    return mk.SERIAL_MAX


def run_one(args):
    """args = (assets, name, cond_name, mode, seed, px_per_mm) -> dict of results (picklable)."""
    from omr.recognize import recognize
    assets, name, cond_name, mode, seed, ppm = args
    L, pdf = assets[name]
    rng = np.random.default_rng(seed)
    serial = 1 + seed % mk_serial_max() if mode != "none" else None
    frame, truth = make_frame(L, pdf, CONDS[cond_name], rng, mode=mode, serial=serial)
    kw = {} if ppm is None else {"px_per_mm": ppm}
    res = recognize(frame, **kw)
    ev = evaluate(res, truth)
    return {"name": name, "cond": cond_name, "mode": mode, "status": res.status, "truth": truth,
            "ev": ev, "ms": res.timing_ms["total"], "timing": res.timing_ms, "id": res.student_id, "subject": res.subject,
            "px_per_mm": truth.get("px_per_mm"), "seed": seed}
