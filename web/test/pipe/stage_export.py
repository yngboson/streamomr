"""Dump Python intermediates of frames (warped, smooth, bubble centres, darkness D, ...) for the stage parity test (stage.js).

    .venv/bin/python web/test/pipe/stage_export.py <set> <idx> [idx...]      -> fixtures/pipe/stage/<set>_<idx>.{json,*.bin}
"""
import json, os, sys
import cv2
import numpy as np
from common import *  # noqa
from omr import recognize as R, illum, layout as lay, ink as inkmod, markers as mk, detect as dt, tracks as trk

setname = sys.argv[1]
man = json.load(open(f"{FIX}/{setname}.manifest.json"))
os.makedirs(f"{FIX}/stage", exist_ok=True)


def dump(base, name, arr):
    arr = np.ascontiguousarray(arr)
    open(f"{base}.{name}.bin", "wb").write(arr.tobytes())
    return {"file": os.path.basename(base) + f".{name}.bin", "shape": list(arr.shape), "dtype": str(arr.dtype)}


for idx in map(int, sys.argv[2:]):
    m = man[idx]
    gray = np.frombuffer(open(f"{FIX}/{m['file']}", "rb").read(), np.uint8).reshape(m["h"], m["w"])
    base = f"{FIX}/stage/{setname}_{idx}"
    meta = {"set": setname, "idx": idx, "arrays": {}}
    found, dup, dinfo = dt.find_markers_scaled(gray)
    meta["found"] = [[int(i), c.tolist()] for i, c in found.items()]
    meta["dup"] = bool(dup)
    # replicate _recognize up to the scores
    page = sorted(found)[0] // 4
    mm = R._marker_corners_mm(page)
    ids = sorted(found)
    src = np.concatenate([found[i] for i in ids]); dstp = np.concatenate([mm[i] for i in ids]) * 6.0
    Hm, _ = cv2.findHomography(src, dstp, 0)
    meta["H"] = Hm.tolist()
    warped = cv2.warpPerspective(gray, Hm, (int(round(lay.PAGE_W * 6)), int(round(lay.PAGE_H * 6))), flags=cv2.INTER_LINEAR, borderValue=255)
    meta["arrays"]["warped"] = dump(base, "warped", warped)
    full = R.recognize(gray, None)
    corner_d = {}
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), Hm).reshape(-1, 2)
    for n_, i in enumerate(ids):
        corner_d[mk.CORNERS[i % 4]] = ((proj - dstp)[4 * n_:4 * n_ + 4].mean(0) / 6.0).tolist()
    meta["corner_d"] = corner_d
    # header on the plain warp
    hdr, info = R._read_header(warped, 6.0)
    meta["header"] = None if hdr is None else dict(questions=hdr.questions, used=hdr.used, choices=hdr.choices, id_digits=hdr.id_digits, subject_digits=hdr.subject_digits,
                                                      pages=hdr.pages, page=hdr.page, labels=hdr.labels, bubble=hdr.bubble, version=hdr.version)
    meta["header_info"] = clean(info)
    flat = R._flatten_header(warped, 6.0)
    meta["arrays"]["header_flat"] = dump(base, "header_flat", flat)
    layout = R.layout_for_header(hdr)
    pg = layout.pages[page]
    # tracks on the plain warp
    tf = trk.measure(warped, 6.0, {k: np.array(v) for k, v in corner_d.items()}) if pg.ticks else None
    warped2 = warped
    if tf is not None:
        meta["tracks"] = {"c": clean(tf.c), "fn": {s: {"u": clean(u), "v": clean(v)} for s, (u, v) in tf._fn.items()}, "sides_ok": clean(tf.sides_ok),
                          "max_disp": tf.max_disp(), "found": {s: [clean(t.d), clean(t.resp)] for s, t in tf.tracks.items()}}
        if tf.max_disp() >= trk.FLAT_MM:
            warped2 = trk.rectify(warped, tf, 6.0)
    meta["tracks_none"] = tf is None
    meta["arrays"]["warped2"] = dump(base, "warped2", warped2)
    smooth = cv2.GaussianBlur(warped2, (0, 0), 1.5)
    meta["arrays"]["smooth"] = dump(base, "smooth", smooth)
    qs = sorted((q for col in pg.columns for q in col.questions), key=lambda q: q.number)
    nc = layout.template.choices
    cent = [(b.cx, b.cy) for q in qs for b in q.bubbles]
    g = pg.id_grid; sg = pg.subject_grid
    icols = list(g.columns) + (list(sg.columns) if sg is not None else [])
    icent = [(b.cx, b.cy) for col in icols for b in col.bubbles]
    adiag = {}
    coef, ainfo = R._fit_alignment(smooth, [(cent, pg.bubble_diameter), (icent, g.bubble_diameter)], 6.0, extrapolated=bool(full.missing_corners) and tf is None,
                                   strict=tf is not None and tf is not trk.FLAT, diag=adiag)
    meta["coef"] = None if coef is None else coef.tolist()
    meta["align_info"] = clean(ainfo)
    meta["cent"] = clean(cent); meta["icent"] = clean(icent)
    meta["dia"] = [pg.bubble_diameter, g.bubble_diameter]
    meta["flags"] = dict(missing=bool(full.missing_corners), strict=bool(tf is not None))
    centA = R._apply_alignment(cent, coef); centI = R._apply_alignment(icent, coef)
    meta["centA"] = clean(centA); meta["centI"] = clean(centI)
    DA, UA = illum.bubble_darkness(smooth, centA, pg.bubble_diameter, 6.0, "E")
    DI, UI = illum.bubble_darkness(smooth, centI, g.bubble_diameter, 6.0, "E")
    meta["arrays"]["DA"] = dump(base, "DA", DA); meta["arrays"]["DI"] = dump(base, "DI", DI)
    meta["UA"] = clean(UA.astype(int)); meta["UI"] = clean(UI.astype(int))
    black = float(np.clip(full.quality.get("meta_black", 0.9), R.META_MIN_CONTRAST, 1.0))
    meta["black"] = black
    inkA_, inkI_ = inkmod.bubble_ink(DA), inkmod.bubble_ink(DI)
    rows = list(inkA_.reshape(-1, nc)) + list(inkI_.reshape(-1, 10))
    ia = inkmod.choose_floor(rows, black)
    meta["ia"] = dict(floor=ia.floor, source=ia.source, sep=ia.separation)
    meta["inkA"] = clean(inkA_); meta["inkI"] = clean(inkI_)
    # smooth + paper samples + centers px etc for illum internals
    v = illum.paper_samples(smooth, centA, pg.bubble_diameter, 6.0)
    meta["v"] = clean(v)
    json.dump(meta, open(base + ".json", "w"))
    print("dumped", base)
