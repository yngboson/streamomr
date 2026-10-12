import os

import cv2
import numpy as np
import pymupdf
import pytest

from omr import markers as mk
from omr.layout import Template, build_layout
from omr.render_pdf import render_pdf

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
DPI = 150
PXMM = DPI / 25.4

CASES = {
    "q45": Template(45, 5, 8),
    "q100_2p": Template(100, 5, 8, pages=2),
    "q100_3p": Template(100, 5, 8, pages=3),
    "q30_abcd": Template(30, 4, 6, choice_labels=list("ABCD")),
    "q100u80_2p": Template(100, 5, 8, pages=2, used=80),
    "q40_nosubj": Template(40, 5, 8, subject_digits=0),
}


@pytest.mark.parametrize("name", CASES)
def test_markers_match_layout(name, tmp_path):
    lay = build_layout(CASES[name])
    pdf = render_pdf(lay, str(tmp_path / "s.pdf"))
    doc = pymupdf.open(pdf)
    assert len(doc) == lay.total_pages
    det = cv2.aruco.ArucoDetector(mk.aruco_dictionary(), cv2.aruco.DetectorParameters())
    os.makedirs(OUT, exist_ok=True)
    for pg, page in zip(lay.pages, doc):
        pix = page.get_pixmap(dpi=DPI)
        pix.save(os.path.join(OUT, f"{name}_p{pg.index + 1}_150dpi.png"))
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = det.detectMarkers(gray)
        assert ids is not None
        found = {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}
        assert set(found) == {m.id for m in pg.markers}
        for m in pg.markers:
            cx = (m.rect[0] + m.rect[2] / 2) * PXMM
            cy = (m.rect[1] + m.rect[3] / 2) * PXMM
            dx, dy = found[m.id].mean(axis=0)
            assert abs(dx - cx) < 2 and abs(dy - cy) < 2, (m.id, dx - cx, dy - cy)
            # corner order: TL corner of detection == TL of rect (orientation upright)
            tl = found[m.id][0]
            assert abs(tl[0] - m.rect[0] * PXMM) < 2 and abs(tl[1] - m.rect[1] * PXMM) < 2


def test_bubbles_drawn_where_layout_says(tmp_path):
    """Every bubble ring is dark at its layout radius and light at its center."""
    lay = build_layout(CASES["q45"])
    doc = pymupdf.open(render_pdf(lay, str(tmp_path / "s.pdf")))
    pix = doc[0].get_pixmap(dpi=300)
    g = cv2.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n), cv2.COLOR_RGB2GRAY)
    k = 300 / 25.4
    pg = lay.pages[0]
    for col in pg.columns:
        for q in col.questions:
            for b in q.bubbles:
                cx, cy, r = b.cx * k, b.cy * k, pg.bubble_diameter / 2 * k
                assert g[int(round(cy)), int(round(cx))] > 200
                ring = min(g[int(round(cy)), int(round(cx + r)) + d] for d in (-3, -2, -1, 0, 1, 2, 3))
                assert ring < 140


def test_header_and_subject_grid_drawn_where_layout_says(tmp_path):
    """Header cells (black = ink, white = paper) and the subject-code rings sit at their layout positions."""
    lay = build_layout(CASES["q100u80_2p"])
    doc = pymupdf.open(render_pdf(lay, str(tmp_path / "s.pdf")))
    pix = doc[1].get_pixmap(dpi=300)
    g = cv2.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n), cv2.COLOR_RGB2GRAY)
    k = 300 / 25.4
    pg = lay.pages[1]
    for cell in pg.meta_cells:
        x, y, w, h = cell.rect
        v = g[int(round((y + h / 2) * k)), int(round((x + w / 2) * k))]
        assert (v < 100) == cell.bit, cell
    sg = pg.subject_grid
    assert sg is not None and len(sg.columns) == 2
    r = sg.bubble_diameter / 2 * k
    for col in sg.columns:
        for b in col.bubbles:
            assert g[int(round(b.cy * k)), int(round(b.cx * k))] > 200
            assert min(g[int(round(b.cy * k)), int(round(b.cx * k + r)) + d] for d in (-3, -2, -1, 0, 1, 2, 3)) < 140
    n_q = sum(len(c.questions) for c in pg.columns)
    assert n_q == 30            # only the used questions are drawn
