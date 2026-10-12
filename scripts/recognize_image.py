#!/usr/bin/env python
"""Recognise one photo of a printed OMR sheet: prints the result as JSON and writes a debug overlay PNG.

    python scripts/recognize_image.py photo.jpg
    python scripts/recognize_image.py old_photo.jpg --layout out/q45.layout.json     # sheet printed before the self-describing header

Sheets with the self-describing header need no layout.  --layout (repeatable) is only for LEGACY sheets
(16-bit form id strip), chosen by the form id read from the sheet.

Overlay: left = the photo with the detected marker corners (green: found, red: missing page-group hint);
right = the sheet warped to canonical orientation, every bubble coloured by the decision
(green = marked, red = multiple marks, orange = ambiguous / needs a look, grey = empty).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from omr.layout import Layout  # noqa: E402
from omr.recognize import layout_for_header, recognize  # noqa: E402
from omr.layout import header_from_form_key  # noqa: E402

GREEN, RED, ORANGE, GREY, BLUE = (60, 180, 60), (40, 40, 220), (0, 150, 255), (150, 150, 150), (220, 120, 0)


def draw_overlay(img, res, layout, ppm):
    h, w = img.shape[:2]
    left = img.copy()
    thick = max(2, int(round(max(h, w) / 600)))
    for mid, c in res.marker_corners.items():
        pts = np.array(c, np.int32)
        cv2.polylines(left, [pts], True, GREEN, thick)
        cv2.circle(left, tuple(pts[0]), thick * 3, RED, -1)  # first corner = marker's own top-left
        cv2.putText(left, str(mid), tuple(pts[0] + [thick * 2, -thick * 2]), cv2.FONT_HERSHEY_SIMPLEX,
                    max(h, w) / 1400, GREEN, thick)
    if res.homography is not None:
        outline = res.image_points([[0, 0], [210, 0], [210, 297], [0, 297]]).astype(np.int32)
        cv2.polylines(left, [outline], True, BLUE, thick)
    txt = f"{res.status}" + (f"  subject={res.subject}  id={res.student_id}" if res.status == "ok" else "") + (f"  flags={res.flags}" if res.flags else "")
    cv2.putText(left, txt, (20, int(max(h, w) / 25)), cv2.FONT_HERSHEY_SIMPLEX, max(h, w) / 1200, RED if res.status != "ok" else GREEN, thick)
    scale = 900.0 / h
    left = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if res.warped is None or res.status not in ("ok",) or layout is None:
        if res.warped is not None:
            right = cv2.cvtColor(res.warped, cv2.COLOR_GRAY2BGR)
            right = cv2.resize(right, None, fx=900.0 / right.shape[0], fy=900.0 / right.shape[0])
            return np.hstack([left, right])
        return left
    right = cv2.cvtColor(res.warped, cv2.COLOR_GRAY2BGR)
    pg = layout.pages[res.page]
    R = pg.bubble_diameter / 2 * ppm
    centers = res.bubble_centers_mm
    for q, cs in zip(res.answers, centers["answers"]):
        for k, (x, y) in enumerate(cs):
            p = (int(x * ppm), int(y * ppm))
            if k in q.marked:
                col = RED if q.state == "multi" else (ORANGE if q.ambiguous else GREEN)
                cv2.circle(right, p, int(R), col, -1 if not q.ambiguous else 2)
                cv2.circle(right, p, int(R), col, 2)
            elif q.ambiguous and q.scores[k] > 0.1:
                cv2.circle(right, p, int(R), ORANGE, 2)
            else:
                cv2.circle(right, p, 2, GREY, -1)
    Ri = pg.id_grid.bubble_diameter / 2 * ppm
    for digits, rows in ((res.id_digits, centers["id"]), (res.subject_digits, centers.get("subject") or [])):
        for d, cs in zip(digits, rows):
            for k, (x, y) in enumerate(cs):
                p = (int(x * ppm), int(y * ppm))
                if d.value == k:
                    cv2.circle(right, p, int(Ri), GREEN, -1)
                elif d.ambiguous and d.scores[k] > 0.1 or d.state == "multi" and d.scores[k] > 0.3:
                    cv2.circle(right, p, int(Ri), ORANGE, 2)
                else:
                    cv2.circle(right, p, 2, GREY, -1)
    x, y, ww, hh = pg.name_box
    cv2.rectangle(right, (int(x * ppm), int(y * ppm)), (int((x + ww) * ppm), int((y + hh) * ppm)), BLUE, 2)
    right = cv2.resize(right, None, fx=900.0 / right.shape[0], fy=900.0 / right.shape[0], interpolation=cv2.INTER_AREA)
    return np.hstack([left, right])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--layout", action="append", default=[], help="legacy layout JSON (repeatable; chosen by the form id read from the sheet)")
    ap.add_argument("--px-per-mm", type=float, default=6.0)
    ap.add_argument("--max-side", type=int, default=2600, help="downscale photos larger than this (long side, px)")
    ap.add_argument("--out", help="overlay PNG path (default: <image>.debug.png)")
    ap.add_argument("--name-crop", help="also write the name-box crop to this PNG")
    ap.add_argument("--compact", action="store_true", help="omit per-bubble scores/centres from the printed JSON")
    o = ap.parse_args()

    layouts = {}
    for p in o.layout:
        with open(p, encoding="utf-8") as f:
            L = Layout.from_json(f.read())
        layouts[L.form_id] = L
        if L.form_id is None:
            sys.exit(f"{p} is not a legacy layout (sheets with the new header need no layout)")
    img = cv2.imread(o.image, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"cannot read {o.image}")
    if max(img.shape[:2]) > o.max_side:
        s = o.max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    res = recognize(img, layouts.get, px_per_mm=o.px_per_mm)
    d = res.to_dict()
    if o.compact:
        d.pop("bubble_centers_mm", None)
        for a in d["answers"]:
            a.pop("scores", None)
        for a in d["id_digits"] + d["subject_digits"]:
            a.pop("scores", None)
    print(json.dumps(d, ensure_ascii=False, indent=1))
    out = o.out or os.path.splitext(o.image)[0] + ".debug.png"
    if res.status == "ok":
        layout = layouts.get(res.form_id) if res.form_id is not None else layout_for_header(header_from_form_key(res.form_key))
    else:
        layout = None
    cv2.imwrite(out, draw_overlay(img, res, layout, o.px_per_mm))
    if o.name_crop and res.name_crop is not None:
        cv2.imwrite(o.name_crop, res.name_crop)
    print(f"overlay written to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
