#!/usr/bin/env python
"""One PDF for testing the recognizer: a blank master sheet followed by N randomly marked answer sheets.

    python scripts/make_test_pdf.py --out out/omr_test            # 45 questions, 8-digit ID, 3-digit subject code 011, 20 sheets
    python scripts/make_test_pdf.py --out out/omr_test --seed 7 --subject 011 --count 20

Writes <out>.pdf (page 1 = blank master, pages 2.. = marked sheets, filled as vector circles over the printed bubbles)
and <out>_truth.csv (page, student_id, subject, answers 1-based per question) as the ground truth to compare scans to.
Student IDs are unique; every choice is uniformly random."""
import argparse
import csv
import os
import sys

import numpy as np
import pymupdf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from omr.layout import Template, build_layout  # noqa: E402
from omr.render_pdf import render_pdf  # noqa: E402

MM = 72.0 / 25.4
INK = (0.1, 0.1, 0.1)
FILL_OF_RADIUS = 0.9


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--out", required=True, help="output prefix")
    a.add_argument("--questions", type=int, default=45)
    a.add_argument("--choices", type=int, default=5)
    a.add_argument("--id-digits", type=int, default=8)
    a.add_argument("--subject", default="011", help="subject code marked on every sheet (its length = subject digits)")
    a.add_argument("--count", type=int, default=20)
    a.add_argument("--seed", type=int, default=2026)
    o = a.parse_args()
    t = Template(questions=o.questions, choices=o.choices, id_digits=o.id_digits, subject_digits=len(o.subject), title="OMR test")
    L = build_layout(t)
    if L.total_pages != 1:
        raise SystemExit(f"the form needs {L.total_pages} pages; this script handles one-page forms")
    os.makedirs(os.path.dirname(os.path.abspath(o.out)), exist_ok=True)
    blank = o.out + "_blank.pdf"
    render_pdf(L, blank)
    rng = np.random.default_rng(o.seed)
    ids = [f"{int(v):0{o.id_digits}d}" for v in rng.choice(10 ** o.id_digits, o.count, replace=False)]
    pg = L.pages[0]

    def dot(page, b, diameter):
        page.draw_circle((b.cx * MM, b.cy * MM), diameter / 2 * MM * FILL_OF_RADIUS, color=None, fill=INK)

    src = pymupdf.open(blank)
    doc = pymupdf.open()
    doc.insert_pdf(src)                                   # page 1: the blank master
    rows = []
    for k, sid in enumerate(ids):
        doc.insert_pdf(src)
        page = doc[-1]
        for d, ch in enumerate(sid):
            dot(page, pg.id_grid.columns[d].bubbles[int(ch)], pg.id_grid.bubble_diameter)
        for d, ch in enumerate(o.subject):
            dot(page, pg.subject_grid.columns[d].bubbles[int(ch)], pg.subject_grid.bubble_diameter)
        qs = sorted((q for col in pg.columns for q in col.questions), key=lambda q: q.number)
        ans = [int(rng.integers(o.choices)) for _ in qs]
        for q, c in zip(qs, ans):
            dot(page, q.bubbles[c], pg.bubble_diameter)
        rows.append([k + 2, sid, o.subject] + [c + 1 for c in ans])
    doc.save(o.out + ".pdf", deflate=True)
    os.remove(blank)
    with open(o.out + "_truth.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["page", "student_id", "subject"] + [f"q{i}" for i in range(1, len(qs) + 1)])
        w.writerows(rows)
    print(f"{o.out}.pdf: {1 + o.count} pages (1 blank + {o.count} marked), form_key={L.form_key}; truth: {o.out}_truth.csv")


if __name__ == "__main__":
    main()
