"""Image helpers for the review UI: exam page previews and highlighted warped sheets.

Everything is derived from the Layout (mm coordinates) - no geometry is hard-coded here.
"""
from __future__ import annotations

import os
from typing import Iterable, List, Optional

import cv2
import numpy as np

from .layout import Layout


def render_pdf_page_png(pdf_path: str, page: int, dpi: int = 80) -> bytes:
    import pymupdf
    doc = pymupdf.open(pdf_path)
    try:
        if not 0 <= page < len(doc):
            raise ValueError(f"page {page} out of range")
        return doc[page].get_pixmap(dpi=dpi).tobytes("png")
    finally:
        doc.close()


def cached_preview(pdf_path: str, page: int, cache_dir: str, dpi: int = 80) -> bytes:
    """PNG of one PDF page, cached on disk next to the PDF (keyed by pdf mtime)."""
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    path = os.path.join(cache_dir, f"{base}_p{page}_{dpi}_{int(os.path.getmtime(pdf_path))}.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    data = render_pdf_page_png(pdf_path, page, dpi)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return data


def question_rect_mm(layout: Layout, page: int, number: int) -> Optional[List[float]]:
    """[x0, y0, x1, y1] in mm around a question's number and bubbles, or None if not on this page."""
    pg = layout.pages[page]
    r = pg.bubble_diameter / 2
    for col in pg.columns:
        for q in col.questions:
            if q.number == number:
                xs = [b.cx for b in q.bubbles] + [q.num_text.x]
                ys = [b.cy for b in q.bubbles]
                return [min(xs) - r - 5.0, min(ys) - r - 0.8, max(xs) + r + 1.5, max(ys) + r + 0.8]
    return None


def annotate_warped(img: np.ndarray, layout: Layout, page: int, questions: Iterable[int]) -> np.ndarray:
    """BGR copy of the warped sheet with a red box around each listed (global) question number."""
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    scale = out.shape[1] / layout.page_w
    for n in questions:
        rect = question_rect_mm(layout, page, int(n))
        if rect is None:
            continue
        x0, y0, x1, y1 = [int(round(v * scale)) for v in rect]
        ov = out.copy()
        cv2.rectangle(ov, (x0, y0), (x1, y1), (60, 60, 255), -1)
        out = cv2.addWeighted(ov, 0.18, out, 0.82, 0)
        cv2.rectangle(out, (x0, y0), (x1, y1), (40, 40, 230), max(2, int(scale * 0.35)))
    return out
