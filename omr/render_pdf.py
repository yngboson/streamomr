"""Layout -> printable PDF. Draws exactly what the Layout says; no geometry decisions here."""
from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from . import markers as mk
from .layout import Layout, Text, serial_cell_rects, serial_label

_FONT_CANDIDATES = [
    ("/System/Library/Fonts/AppleSDGothicNeo.ttc", 0),
    ("/System/Library/Fonts/Supplemental/AppleGothic.ttf", None),
    ("/Library/Fonts/NanumGothic.ttf", None),
    ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", None),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
]


def register_font() -> Tuple[str, bool]:
    """Return (font name, supports_cjk). Falls back to Helvetica (Latin only)."""
    for path, idx in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                kw = {"subfontIndex": idx} if idx is not None else {}
                pdfmetrics.registerFont(TTFont("OMRFont", path, **kw))
                return "OMRFont", True
            except Exception:
                continue
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
        return "HYSMyeongJo-Medium", True
    except Exception:
        return "Helvetica", False


def render_pdf(layout: Layout, path: str, serials: Optional[Sequence[int]] = None) -> str:
    """Render the sheet.  ``serials=None``: one set, no serial (the master for photocopying).
    ``serials=[s1, s2, ...]``: one set of pages per serial, in that order (a set = one student's pages)."""
    font, cjk = register_font()
    c = canvas.Canvas(path, pagesize=(layout.page_w * mm, layout.page_h * mm))
    H = layout.page_h
    sw = layout.stroke_mm * mm

    def Y(y):
        return (H - y) * mm

    def rect(r, fill=None, stroke=False, gray_fill=None):
        c.rect(r[0] * mm, Y(r[1] + r[3]), r[2] * mm, r[3] * mm, stroke=int(stroke), fill=int(fill is not None))

    def text(t: Text):
        s = t.text if cjk else "".join(ch for ch in t.text if ord(ch) < 256)
        c.setFillGray(t.gray)
        c.setFont(font, t.size_pt)
        x, y = t.x * mm, Y(t.y)
        {"l": c.drawString, "c": c.drawCentredString, "r": c.drawRightString}[t.align](x, y, s)

    c.setTitle(f"OMR form {layout.form_key}")
    sets = [0] if serials is None else [int(x) for x in serials]
    if not sets:
        raise ValueError("serials must not be empty")
    for serial in sets:
        if serial and not 1 <= serial <= mk.SERIAL_MAX:
            raise ValueError(f"serial {serial} out of range 1..{mk.SERIAL_MAX}")
    for serial, pg in ((s_, p_) for s_ in sets for p_ in layout.pages):
        c.setLineWidth(sw)
        c.setStrokeGray(0)
        # light row bands
        for col in pg.columns:
            for b in col.bands:
                c.setFillGray(b.gray)
                rect(b.rect, fill=True)
        # frames
        c.setFillGray(1)
        for fr in (pg.info_box, pg.answer_box):
            rect(fr, stroke=True)
        rect(pg.name_box, stroke=True)
        # markers
        for m in pg.markers:
            bits = mk.marker_bits(m.id)
            k = bits.shape[0]
            cs = m.rect[2] / k
            c.setFillGray(0)
            rect(m.rect, fill=True)
            c.setFillGray(1)
            for r in range(k):
                x = 0
                while x < k:
                    if bits[r, x]:
                        x += 1
                        continue
                    x2 = x
                    while x2 < k and not bits[r, x2]:
                        x2 += 1
                    rect([m.rect[0] + x * cs, m.rect[1] + r * cs, (x2 - x) * cs, cs], fill=True)
                    x = x2
        # meta strip
        c.setFillGray(0)
        for cell in pg.meta_cells:
            if cell.bit:
                rect(cell.rect, fill=True)
        # timing tracks (header version 2)
        c.setFillGray(0)
        for tk in pg.ticks:
            rect(tk.rect, fill=True)
        # serial strip + human-readable number (nothing for serial 0)
        if serial:
            c.setFillGray(0)
            for r_, bit in zip(serial_cell_rects(), mk.encode_serial(serial)):
                if bit:
                    rect(r_, fill=True)
            text(serial_label(serial))
        # student id + subject code
        for grid in (g for g in (pg.id_grid, pg.subject_grid) if g is not None):
            for col in grid.columns:
                c.setStrokeGray(0)
                rect(col.writein, stroke=True)
                for b in col.bubbles:
                    c.circle(b.cx * mm, Y(b.cy), grid.bubble_diameter / 2 * mm, stroke=1, fill=0)
        # answer bubbles
        for col in pg.columns:
            for q in col.questions:
                for b in q.bubbles:
                    c.circle(b.cx * mm, Y(b.cy), pg.bubble_diameter / 2 * mm, stroke=1, fill=0)
        # text
        subj = [pg.subject_label, *pg.subject_grid.row_labels] if pg.subject_grid is not None else []
        for t in [pg.id_label, pg.name_label, *pg.texts, *pg.id_grid.row_labels, *subj]:
            text(t)
        for col in pg.columns:
            for t in col.headers:
                text(t)
            for q in col.questions:
                text(q.num_text)
        c.showPage()
    c.save()
    return path
