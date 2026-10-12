#!/usr/bin/env python
"""Generate a printable OMR sheet PDF and its layout JSON.

The sheet's header records how to read it, so the layout JSON is only a convenience (the web app and the
recognizer need no file or database).  Questions are split evenly over the pages."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from omr.layout import Template, build_layout, describe  # noqa: E402
from omr.markers import LABEL_STYLES, choice_labels  # noqa: E402
from omr.render_pdf import render_pdf  # noqa: E402


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--questions", type=int, default=45)
    a.add_argument("--choices", type=int, default=5)
    a.add_argument("--id-digits", type=int, default=8)
    a.add_argument("--used", type=int, help="questions actually used (default: all); the rest are not printed or read")
    a.add_argument("--subject-digits", type=int, default=2, help="digits of the marked subject code, 0 = none")
    a.add_argument("--pages", type=int)
    a.add_argument("--title", default="")
    a.add_argument("--labels", default="1", choices=list(LABEL_STYLES),
                   help="choice label style: 1 (1 2 3..), A (A B C..), a (a b c..) or 가 (가 나 다..)")
    a.add_argument("--min-bubble", type=float, default=4.0)
    a.add_argument("--copies", type=int, help="print N numbered sets (serials start..start+N-1); default: one plain master")
    a.add_argument("--start-serial", type=int, default=1, help="first serial of --copies (1..4095)")
    a.add_argument("--out", required=True, help="output prefix, writes <out>.pdf and <out>.layout.json")
    a.add_argument("--preview", action="store_true", help="also write <out>_p<N>.png (needs pymupdf)")
    o = a.parse_args()
    labels = choice_labels(LABEL_STYLES.index(o.labels), o.choices)
    t = Template(questions=o.questions, choices=o.choices, id_digits=o.id_digits, pages=o.pages, title=o.title,
                 choice_labels=labels, min_bubble_mm=o.min_bubble, used=o.used, subject_digits=o.subject_digits)
    lay = build_layout(t)
    os.makedirs(os.path.dirname(os.path.abspath(o.out)), exist_ok=True)
    serials = None
    if o.copies:
        serials = list(range(o.start_serial, o.start_serial + o.copies))
        if o.start_serial < 1 or serials[-1] > 4095:
            raise SystemExit("serials must stay within 1..4095")
    render_pdf(lay, o.out + ".pdf", serials=serials)
    with open(o.out + ".layout.json", "w", encoding="utf-8") as f:
        f.write(lay.to_json(indent=1))
    if o.preview:
        import pymupdf
        for i, pg in enumerate(pymupdf.open(o.out + ".pdf")):
            pg.get_pixmap(dpi=100).save(f"{o.out}_p{i + 1}.png")
    if serials:
        print(f"{len(serials)} numbered sets, serials {serials[0]}..{serials[-1]} (not recorded in any database; "
              f"use the web UI / API to track issued ranges)")
    print(f"{o.out}.pdf  form_key={lay.form_key} pages={lay.total_pages} per-page={lay.questions_per_page}  ({describe(lay)})")


if __name__ == "__main__":
    main()
