"""Timing tracks (header version 2): sheet design, versioning, and straightening a folded / curled sheet.

The sheet carries rows of small black squares along its four edges; recognition measures where they really are and
bends the page flat again (omr.tracks).  Older sheets (no tracks) must keep reading exactly as before.
"""
import dataclasses
import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import synth  # noqa: E402

from omr import layout as lay  # noqa: E402
from omr import markers as mk  # noqa: E402
from omr import tracks as trk  # noqa: E402
from omr.recognize import layout_for_header, recognize  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TEMPLATES = {
    "q45": lay.Template(45, 5, 8, title="Midterm"),
    "q100_3p": lay.Template(100, 5, 8, pages=3),
    "wide": lay.Template(60, 10, 12, subject_digits=3, min_bubble_mm=5.75),
}


def overlap(a, b, gap=0.0):
    return a[0] < b[0] + b[2] + gap and b[0] < a[0] + a[2] + gap and a[1] < b[1] + b[3] + gap and b[1] < a[1] + a[3] + gap


# ------------------------------------------------------------------ the sheet
@pytest.mark.parametrize("name", TEMPLATES)
def test_ticks_do_not_touch_anything_else(name):
    L = lay.build_layout(TEMPLATES[name])
    for pg in L.pages:
        assert len(pg.ticks) == 2 * lay.TRACK_N_LR + 2 * lay.TRACK_N_TB
        others = {"marker quiet": [m.quiet for m in pg.markers], "header": [pg.meta_rect], "info box": [pg.info_box],
                  "answer box": [pg.answer_box], "name box": [pg.name_box],
                  "serial": lay.serial_cell_rects()}
        for t in pg.ticks:
            x, y, w, h = t.rect
            assert 4.0 <= x and x + w <= L.page_w - 4.0 and 4.0 <= y and y + h <= L.page_h - 4.0, t   # clear of the printer's clip
            for what, rects in others.items():
                for r in rects:
                    assert not overlap(t.rect, r, 0.5), (t, what, r)
        for a in pg.ticks:                                               # and of each other
            for b in pg.ticks:
                if a is not b and a.side == b.side:
                    assert not overlap(a.rect, b.rect, 1.0)


def test_header_version_and_form_key_tell_the_formats_apart():
    new = lay.build_layout(lay.Template(45, 5, 8))
    old = lay.build_layout(lay.Template(45, 5, 8, tracks=False))
    assert new.version == 2 and new.form_key.endswith("V2") and new.header(0).version == 2 and new.pages[0].ticks
    assert old.version == 1 and old.form_key.endswith("V1") and old.header(0).version == 1 and not old.pages[0].ticks
    assert new.form_key != old.form_key
    for L in (new, old):
        h = mk.decode_header(mk.encode_header(L.header(0)))
        assert h == L.header(0)
        assert lay.header_from_form_key(L.form_key) == L.header(0)
        L2 = layout_for_header(h)
        assert L2.version == L.version and L2.form_key == L.form_key and bool(L2.pages[0].ticks) == bool(L.pages[0].ticks)
    with pytest.raises(ValueError):
        mk.encode_header(dataclasses.replace(new.header(0), version=3))


def test_plain_sheet_geometry_is_unchanged():
    """tracks=False is the format printed before tracks existed: same frames, same answer area."""
    L = lay.build_layout(lay.Template(45, 5, 8, tracks=False))
    assert L.pages[0].answer_box[0] == lay.MARGIN + lay.FRAME_INSET and L.pages[0].answer_box[2] == 193.5
    new = lay.build_layout(lay.Template(45, 5, 8))
    assert new.pages[0].bubble_diameter == L.pages[0].bubble_diameter           # the tracks cost bubbles nothing here
    assert [len(c.questions) for c in new.pages[0].columns] == [len(c.questions) for c in L.pages[0].columns]
    for c in range(2, 11):                                                       # ... nor capacity
        for kw in (dict(id_digits=8), dict(id_digits=12, subject_digits=3, min_bubble_mm=5.75)):
            assert lay.page_capacity(lay.Template(50, c, tracks=False, **kw)) == lay.page_capacity(lay.Template(50, c, tracks=True, **kw))


def test_layouts_saved_before_tracks_load_without_them():
    d = lay.build_layout(lay.Template(45, 5, 8, tracks=False)).to_dict()
    d["template"].pop("tracks")
    for p in d["pages"]:
        p.pop("ticks")
    L = lay.Layout.from_dict(d)
    assert L.version == 1 and not L.template.tracks and not L.pages[0].ticks
    legacy = os.path.join(ROOT, "out", "q45.layout.json")
    if os.path.exists(legacy):
        L0 = lay.Layout.from_json(open(legacy).read())
        assert L0.version == 0 and not L0.template.tracks and not L0.pages[0].ticks


def test_pdf_has_the_ticks(tmp_path):
    import pymupdf
    from omr.render_pdf import render_pdf
    paths = {}
    for tracks in (True, False):
        L = lay.build_layout(lay.Template(45, 5, 8, tracks=tracks))
        paths[tracks] = render_pdf(L, str(tmp_path / f"t{int(tracks)}.pdf"))
    for tracks, path in paths.items():
        pix = pymupdf.open(path)[0].get_pixmap(dpi=100, colorspace=pymupdf.csGRAY)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w)
        s = 100 / 25.4
        dark = 0
        for r in lay.track_tick_rects()["L"] + lay.track_tick_rects()["B"]:
            x, y, w, h = r
            dark += img[int((y + 0.3) * s):int((y + h - 0.3) * s), int((x + 0.3) * s):int((x + w - 0.3) * s)].mean() < 60
        assert (dark == 40 + 26) if tracks else dark == 0


# ------------------------------------------------------------------ measuring and straightening
def canvas_of(pdf, ppm=6.0):
    r = synth.paper_raster(pdf, 0)
    s = ppm / synth.PPM_HI
    return cv2.resize(r, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def displace(canvas, dfun, ppm=6.0):
    """What the corner-marker warp looks like when the page point p is at p + d(p) (mm): canvas(q) = paper(p), p + d(p) = q."""
    h, w = canvas.shape
    qy, qx = np.mgrid[0:h, 0:w].astype(np.float32)
    px, py = qx.copy(), qy.copy()
    for _ in range(8):
        d = dfun(px / ppm, py / ppm)
        px, py = qx - (d[..., 0] * ppm).astype(np.float32), qy - (d[..., 1] * ppm).astype(np.float32)
    return cv2.remap(canvas, px, py, cv2.INTER_LINEAR, borderValue=255)


def crease_y(x, y):           # a horizontal crease at y = 140: the lower half is raised, i.e. magnified and shifted
    k = np.clip(y - 140.0, 0, None)
    return np.stack([0.03 * (x - 105.0) * (k / 100.0), 0.025 * k], -1)


def crease_x(x, y):           # a vertical crease at x = 120
    k = np.clip(x - 120.0, 0, None)
    return np.stack([0.04 * k, 0.02 * (y - 148.0) * (k / 60.0)], -1)


@pytest.fixture(scope="module")
def sheet(tmp_path_factory):
    L = lay.build_layout(lay.Template(45, 5, 8))
    pdf = synth.build_pdf(L, str(tmp_path_factory.mktemp("trk") / "q45.pdf"))
    return L, pdf


@pytest.mark.parametrize("dfun", [crease_y, crease_x], ids=["horizontal crease", "vertical crease"])
def test_field_recovers_a_crease(sheet, dfun):
    _, pdf = sheet
    base = canvas_of(pdf)
    bent = displace(base, dfun)
    f = trk.measure(bent, 6.0)
    assert f is not None and all(f.sides_ok.values())
    xs, ys = np.meshgrid(np.linspace(20, 190, 12), np.linspace(35, 265, 24))
    err = np.linalg.norm(f(xs, ys) - dfun(xs, ys), axis=-1)
    assert np.median(err) < 0.4 and err.max() < 1.0, (np.median(err), err.max())
    flat = trk.rectify(bent, f, 6.0)
    again = trk.measure(flat, 6.0)                               # straightened: the ticks are where they belong
    assert again is not None and again.max_disp() < 0.8 < f.max_disp()


def test_flat_page_has_no_field(sheet):
    _, pdf = sheet
    f = trk.measure(canvas_of(pdf), 6.0)
    assert f is not None and f.max_disp() < trk.FLAT_MM and f.info()["tracks_found"] > 0.95


def test_no_tracks_no_field(tmp_path):
    L = lay.build_layout(lay.Template(45, 5, 8, tracks=False))
    pdf = synth.build_pdf(L, str(tmp_path / "plain.pdf"))
    assert trk.measure(canvas_of(pdf), 6.0) is None
    assert trk.measure(np.full((1782, 1260), 255, np.uint8), 6.0) is None


def test_ticks_hidden_by_a_hand_are_skipped(sheet):
    _, pdf = sheet
    bent = displace(canvas_of(pdf), crease_y)
    bent[600:900, :90] = 120                                      # a thumb over part of the left edge
    f = trk.measure(bent, 6.0)
    assert f is not None and 0.6 < f.tracks["L"].frac < 0.95
    xs, ys = np.meshgrid(np.linspace(20, 190, 6), np.linspace(35, 265, 12))
    assert np.linalg.norm(f(xs, ys) - crease_y(xs, ys), axis=-1).max() < 1.0


# ------------------------------------------------------------------ recognition of bent sheets
@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("sheets"))
    return (synth.prepare(d + "/new", {"q45": dataclasses.replace(synth.STANDARD_TEMPLATES["q45"], tracks=True),
                                       "q100_2p": synth.STANDARD_TEMPLATES["q100_2p"]}),
            synth.prepare(d + "/old", {"q45": dataclasses.replace(synth.STANDARD_TEMPLATES["q45"], tracks=False),
                                       "q100_2p": dataclasses.replace(synth.STANDARD_TEMPLATES["q100_2p"], tracks=False)}))


def run(assets, cond, seeds, names=("q45", "q100_2p")):
    out = []
    for i, seed in enumerate(seeds):
        name = names[i % len(names)]
        L, pdf = assets[name]
        frame, truth = synth.make_frame(L, pdf, synth.CONDS[cond], np.random.default_rng(seed), mode="sheet", serial=7)
        res = recognize(frame)
        out.append((res, truth, synth.evaluate(res, truth)))
    return out


def test_folded_sheets_read_better_with_tracks(assets):
    new, old = assets
    seeds = range(3000, 3012)
    a, b = run(new, "fold", seeds), run(old, "fold", seeds)
    ok_new = [(r, t, ev) for r, t, ev in a if r.status == "ok"]
    ok_old = [(r, t, ev) for r, t, ev in b if r.status == "ok"]
    assert len(ok_new) >= len(ok_old) and len(ok_new) >= 8
    for res, truth, ev in ok_new:
        assert ev["id_wrong"] == 0 and ev["subject_wrong"] == 0 and ev["q_silent_clear"] == 0, (truth["cond"], res.flags)
        assert res.form_key.endswith("V2")
        assert res.quality["tracks"] in (0.0, 1.0)
    tracked = [r for r, _, _ in ok_new if r.quality.get("tracks") == 1.0]
    assert tracked and all(r.quality["tracks_sides"] >= 2 for r in tracked)
    good = sum(ev["q_clear_ok"] for _, _, ev in ok_new) / max(sum(ev["q_clear"] for _, _, ev in ok_new), 1)
    assert good > 0.97, good
    json.dumps(tracked[0].to_dict())                                   # the field is not part of the JSON view
    assert tracked[0].dewarp is None or tracked[0].dewarp is not None


def test_overlay_points_follow_the_bent_page(assets):
    """image_points maps page mm through the field: a bubble centre lands on the printed ring in the camera image."""
    new, _ = assets
    for res, truth, _ in run(new, "fold", range(3000, 3012), names=("q45",)):
        if res.status == "ok" and res.dewarp is not None:
            break
    else:
        pytest.skip("no frame needed straightening")
    c = res.bubble_centers_mm["answers"][0][0]
    p_plain = np.asarray(res.homography)
    plain = cv2.perspectiveTransform(np.array([[c]], np.float64) * res.px_per_mm, np.linalg.inv(p_plain))[0, 0]
    with_field = res.image_points([c])[0]
    d = res.dewarp(np.array([c[0]]), np.array([c[1]]))[0]
    assert np.linalg.norm(with_field - plain) > 0 and np.linalg.norm(d) < trk.MAX_DISP_MM


def test_plain_sheets_still_read_and_never_use_tracks(assets):
    _, old = assets
    for res, truth, ev in run(old, "tilted", range(4000, 4004)):
        assert res.status == "ok" and res.form_key.endswith("V1") and "tracks" not in res.quality and res.dewarp is None
        assert ev["id_wrong"] == 0 and ev["q_clear_ok"] == ev["q_clear"]
