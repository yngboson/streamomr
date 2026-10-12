"""Recognition tests on synthetic camera frames (fixed seeds, so deterministic).

The synthetic generator (tests/synth.py) rasterises the real PDF, draws human-like marks (plus hard cases with
ground truth) and simulates a 1080p / phone camera at ~2-3.5 px/mm.  scripts/bench.py runs a much larger set.
"""
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
from omr.recognize import hard_flags, recognize  # noqa: E402


@pytest.fixture(scope="session")
def assets(tmp_path_factory):
    return synth.prepare(str(tmp_path_factory.mktemp("sheets")))


def plain_frame(raster, ppm=3.2, pad_mm=12):
    """Paper raster -> BGR frame at ppm px/mm with a grey desk border, no camera effects."""
    s = ppm / synth.PPM_HI
    small = cv2.resize(raster, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    pad = int(pad_mm * ppm)
    small = cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=90)
    return cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)


def marked_frame(assets, name, page=0, seed=0, hard_rate=0.0):
    L, pdf = assets[name]
    img, truth = synth.mark_sheet(synth.paper_raster(pdf, page), L, page, np.random.default_rng(seed), hard_rate,
                                  id_hard=False)
    return plain_frame(img), truth, L


# ------------------------------------------------------------------ accuracy by condition
N_PER_COND = 6
SEEDS = {c: 100 * (i + 1) for i, c in enumerate(synth.CONDS)}
SPARSE_CONDS = [c for c in synth.CONDS if c.startswith('sp_')]


@pytest.fixture(scope="session")
def results(assets):
    """{cond: [(result, truth)]}; frames cycle through the 4 templates."""
    names = list(assets)
    out = {}
    for cn in synth.MAIN_CONDS:
        rows = []
        for i in range(N_PER_COND):
            L, pdf = assets[names[i % len(names)]]
            frame, truth = synth.make_frame(L, pdf, synth.CONDS[cn], np.random.default_rng(SEEDS[cn] + i))
            rows.append((recognize(frame), truth))
        out[cn] = rows
    return out


@pytest.mark.parametrize("cond", synth.MAIN_CONDS)
def test_condition(results, cond):
    rows = results[cond]
    read = [(r, t) for r, t in rows if r.status == "ok"]
    # a sheet frame may be skipped (streaming retries next frame) but most must be read
    assert len(read) >= N_PER_COND - 1, [r.status for r, _ in rows]
    tot = {}
    for r, t in read:
        synth.merge(tot, synth.evaluate(r, t))
    assert tot["q_silent"] == 0, tot["failures"]           # wrong AND not flagged
    assert tot["id_wrong"] == 0
    assert tot["subject_wrong"] == 0                        # a wrong subject is worse than none
    assert tot["q_clear_ok"] / tot["q_clear"] >= 0.995
    assert tot["q_clear_flagged"] / tot["q_clear"] <= 0.02  # clear marks should not be nagging the user
    # the deliberately hard half-filled marks must never come out silently as an answer
    amb = tot["kinds"].get("ambiguous")
    if amb:
        assert amb[3] == 0


def test_hard_cases_flagged_or_right(results):
    tot = {}
    for rows in results.values():
        for r, t in rows:
            if r.status == "ok":
                synth.merge(tot, synth.evaluate(r, t))
    assert tot["q_hard"] > 100
    assert tot["q_hard_flagged"] / tot["q_hard"] > 0.1   # hard cases do get flagged
    assert tot["id_wrong"] == 0 and tot["subject_wrong"] == 0


def test_student_id_read_when_clear(assets):
    for name in assets:
        frame, truth, _ = marked_frame(assets, name, seed=3)
        r = recognize(frame)
        assert r.status == "ok" and r.student_id == truth["student_id"], (name, r.flags)
        assert all(d.value is not None and not d.ambiguous for d in r.id_digits)


# ------------------------------------------------------------------ orientation
@pytest.mark.parametrize("k", [1, 2, 3])
def test_any_inplane_rotation(assets, k):
    frame, truth, _ = marked_frame(assets, "q45", seed=5)
    base = recognize(frame)
    r = recognize(np.ascontiguousarray(np.rot90(frame, k)))
    assert base.status == r.status == "ok"
    assert [a.marked for a in r.answers] == [a.marked for a in base.answers]
    assert r.student_id == base.student_id == truth["student_id"]


def test_answers_match_truth_clean(assets):
    for name in assets:
        frame, truth, L = marked_frame(assets, name, page=L_last(assets, name), seed=7)
        r = recognize(frame)
        assert r.status == "ok"
        for tq, a in zip(truth["questions"], r.answers):
            if tq["expect"] == "single":
                assert a.marked == [tq["answer"]] and not a.ambiguous
            else:
                assert a.state == "blank"


def L_last(assets, name):
    return assets[name][0].total_pages - 1


def test_gray_input_accepted(assets):
    frame, truth, _ = marked_frame(assets, "q45", seed=2)
    r = recognize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    assert r.status == "ok" and r.student_id == truth["student_id"]


# ------------------------------------------------------------------ non-ok statuses
def test_no_markers_is_no_sheet():
    rng = np.random.default_rng(0)
    img = (rng.random((1080, 1920, 3)) * 255).astype(np.uint8)
    r = recognize(img)
    assert r.status == "no_sheet" and r.answers == [] and r.student_id is None


# ------------------------------------------------------------------ self-describing sheets
def test_sheets_describe_themselves(assets):
    """No resolver, no database: form identity, page count and the question split come from the header."""
    for name, (L, pdf) in assets.items():
        for page in range(L.total_pages):
            frame, truth, _ = marked_frame(assets, name, page=page, seed=20 + page)
            r = recognize(frame)
            assert r.status == "ok", (name, page, r.status, r.flags)
            assert r.form_key == L.form_key and r.form_id is None and r.page == page and r.total_pages == L.total_pages
            assert [a.number for a in r.answers] == [q.number for c in L.pages[page].columns for q in c.questions]
            assert len(r.subject_digits) == L.template.subject_digits and len(r.id_digits) == L.template.id_digits


def test_unused_questions_are_never_read(assets):
    L, _ = assets["q100u80_2p"]
    seen = []
    for page in range(2):
        frame, truth, _ = marked_frame(assets, "q100u80_2p", page=page, seed=3)
        r = recognize(frame)
        seen += [a.number for a in r.answers]
        assert r.status == "ok"
    assert seen == list(range(1, 81))


def test_subject_read_when_clear(assets):
    for name in assets:
        frame, truth, L = marked_frame(assets, name, seed=3)
        r = recognize(frame)
        assert r.status == "ok" and r.subject == truth["subject"], (name, r.subject, truth["subject"], r.flags)
        if L.template.subject_digits:
            assert len(r.subject) == L.template.subject_digits and not hard_flags(r.flags)
            assert len(r.bubble_centers_mm["subject"]) == L.template.subject_digits
        else:
            assert r.subject == "" and r.subject_digits == []


@pytest.mark.parametrize("kind", ["double", "blank", "stray", "erased", "half"])
def test_hard_subject_marks_are_never_misread(assets, kind):
    """A damaged subject digit leaves the subject unread (None), reads the right one, or yields a best guess flagged
    `subject_guessed` (needs review) - but is never wrong without a flag."""
    L, pdf = assets["q45"]
    n_none = 0
    for seed in range(8):
        rng = np.random.default_rng(500 + seed)
        img = synth.paper_raster(pdf, 0).copy()
        pg = L.pages[0]
        g = pg.subject_grid
        digits = [int(x) for x in rng.integers(0, 10, len(g.columns))]
        bad = int(rng.integers(len(g.columns)))
        for ci, col in enumerate(g.columns):
            d = digits[ci]
            R = g.bubble_diameter / 2
            if ci == bad and kind == "blank":
                continue
            if ci == bad and kind == "half":
                synth.draw_ink(img, col.bubbles[d].cx, col.bubbles[d].cy, R, rng, "chord")
                continue
            synth.draw_ink(img, col.bubbles[d].cx, col.bubbles[d].cy, R, rng, "fill")
            if ci == bad:
                o = (d + 1 + int(rng.integers(8))) % 10
                synth.draw_ink(img, col.bubbles[o].cx, col.bubbles[o].cy, R, rng,
                               "fill" if kind == "double" else "stray" if kind == "stray" else "erased")
        r = recognize(plain_frame(img))
        assert r.status == "ok"
        truth = "".join(map(str, digits))
        assert r.subject in (None, truth) or r.subject_guessed, (kind, seed, r.subject, truth)   # a wrong value is only ever a FLAGGED guess
        if kind in ("double", "blank", "half"):
            assert (r.subject is None or r.subject_guessed) and r.needs_review
            n_none += 1
        if r.subject is None:
            assert any(f in r.flags for f in ("subject_blank", "subject_unreadable"))
    if kind in ("double", "blank", "half"):
        assert n_none == 8


def test_blank_subject_is_flagged_blank_not_unreadable(assets):
    L, pdf = assets["q45"]
    frame = plain_frame(synth.paper_raster(pdf, 0))
    r = recognize(frame)
    assert r.status == "ok" and r.subject is None and "subject_blank" in r.flags and "subject_unreadable" not in r.flags


# ------------------------------------------------------------------ legacy sheets (printed before the header existed)
LEGACY_LAYOUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "q45.layout.json")
LEGACY_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "q45.pdf")


@pytest.mark.skipif(not (os.path.exists(LEGACY_LAYOUT) and os.path.exists(LEGACY_PDF)), reason="no legacy sheet in out/")
def test_legacy_sheet_needs_its_layout_and_still_reads():
    L = lay.Layout.from_json(open(LEGACY_LAYOUT, encoding="utf-8").read())
    assert L.version == 0 and L.form_id is not None
    img, truth = synth.mark_sheet(synth.paper_raster(LEGACY_PDF, 0), L, 0, np.random.default_rng(9), 0.0, id_hard=False)
    frame = plain_frame(img)
    r = recognize(frame)                                   # no resolver: the layout cannot be derived
    assert r.status == "unknown_form" and r.form_id == L.form_id and r.answers == []
    r = recognize(frame, lambda fid: L if fid == L.form_id else None)
    assert r.status == "ok" and r.form_key == L.form_key == f"F{L.form_id:04X}" and r.form_id == L.form_id
    assert r.subject == "" and r.subject_digits == [] and r.student_id == truth["student_id"]
    ev = synth.evaluate(r, truth)
    assert ev["q_silent"] == 0 and ev["q_clear_ok"] == ev["q_clear"] and ev["subject_wrong"] == 0


def test_bad_meta_checksum(assets):
    L, pdf = assets["q45"]
    img = synth.paper_raster(pdf, 0).copy()
    for i in (12, 13, 14):                              # flip three data cells of the header: beyond what the CRC can repair
        x, y, w, h = L.pages[0].meta_cells[i].rect
        x0, y0, x1, y1 = [int(v * synth.PPM_HI) for v in (x, y, x + w, y + h)]
        img[y0:y1, x0:x1] = 255 if L.pages[0].meta_cells[i].bit else 0
    r = recognize(plain_frame(img))
    assert r.status == "bad_meta" and r.answers == []


def test_page_from_strip_must_match_markers(assets):
    L, pdf = assets["q100_2p"]
    a = synth.paper_raster(pdf, 0).copy()
    b = synth.paper_raster(pdf, 1)
    y0, y1 = int(7 * synth.PPM_HI), int(19 * synth.PPM_HI)
    x0, x1 = int(20 * synth.PPM_HI), int(190 * synth.PPM_HI)
    a[y0:y1, x0:x1] = b[y0:y1, x0:x1]                    # page-2 header under page-1 markers
    r = recognize(plain_frame(a))
    assert r.status == "bad_meta" and "page_mismatch" in r.flags


def test_mixed_page_groups_inconsistent(assets):
    L, pdf = assets["q100_2p"]
    f = np.hstack([plain_frame(synth.paper_raster(pdf, 0)), plain_frame(synth.paper_raster(pdf, 1))])
    r = recognize(f)
    assert r.status == "inconsistent" and r.answers == []
    assert set(r.marker_ids) == {0, 1, 2, 3, 4, 5, 6, 7}


def _cover(frame, L, corner_idx, s=3.2, r_mm=14):
    """Thumb-coloured blob over one corner marker of the plain (no camera) frame made by plain_frame()."""
    q = L.pages[0].markers[corner_idx].rect
    cx, cy = int((q[0] + 8 + 12) * s), int((q[1] + 8 + 12) * s)
    cv2.circle(frame, (cx, cy), int(r_mm * s), (120, 150, 210), -1)


@pytest.mark.parametrize("hidden", [0, 1, 2, 3])
def test_one_hidden_marker_still_reads_from_three(assets, hidden):
    frame, truth, L = marked_frame(assets, "q45", seed=11 + hidden)
    _cover(frame, L, hidden)
    r = recognize(frame)
    assert r.status == "ok", r.status
    assert r.missing_corners == [mk.CORNERS[hidden]] and r.quality["markers_used"] == 3
    assert "corner_hidden" in r.flags
    ev = synth.evaluate(r, truth)
    assert ev["q_silent"] == 0 and ev["id_wrong"] == 0 and ev["q_clear_ok"] == ev["q_clear"], ev["failures"]
    assert r.student_id == truth["student_id"]
    assert hard_flags(r.flags) == []   # a hidden corner alone is informational, not a reason for human review


@pytest.mark.parametrize("pair", [(1, 2), (0, 2), (0, 3)])
def test_two_hidden_markers_is_partial_not_a_result(assets, pair):
    """2 markers = 8 points is not stable enough for a homography over a hand-held (bowed) sheet: refuse."""
    frame, _, L = marked_frame(assets, "q45")
    for k in pair:
        _cover(frame, L, k)
    r = recognize(frame)
    assert r.status == "partial"
    assert r.answers == [] and r.student_id is None
    assert r.missing_corners == [mk.CORNERS[k] for k in range(4) if k in pair]
    assert all(len(c) == 4 for c in r.marker_corners.values())


def test_three_markers_geometry_is_checked(assets):
    """A third marker that does not fit the sheet geometry (wrong place) must not give a result."""
    frame, _, L = marked_frame(assets, "q45")
    _cover(frame, L, 2)
    h, w = frame.shape[:2]
    q = L.pages[0].markers[2].rect
    s = 3.2
    # a genuine-looking BR marker pasted far from where the sheet geometry puts it
    _paste_marker(frame, mk.marker_id(0, "BR"), int((q[0] + 12 - 30) * s), int((q[1] + 12 - 44) * s), px=int(16 * s))
    r = recognize(frame)
    assert r.status != "ok" and r.answers == []


@pytest.mark.parametrize("ppm", [2.4, 3.2, 4.5, 6.5])
def test_streak_suppression_is_scale_independent(assets, ppm):
    """The streak-removing opening must follow the marker size, not a magic pixel count."""
    L, pdf = assets["q45"]
    rng = np.random.default_rng(31)
    raster = synth.print_defects(synth.paper_raster(pdf, 0), 0, rng, p_band=0.0)
    img, truth = synth.mark_sheet(raster, L, 0, np.random.default_rng(5), 0.0, id_hard=False)
    r = recognize(plain_frame(img, ppm=ppm))
    assert r.status == "ok", (ppm, r.status, r.marker_ids)
    ev = synth.evaluate(r, truth)
    assert ev["q_silent"] == 0 and ev["id_wrong"] == 0


@pytest.mark.parametrize("cond", synth.DEFECT_CONDS)
def test_print_defects_and_hand_no_silent_errors(assets, cond):
    """Grey streaky print and a thumb over a corner are NORMAL conditions: most frames read, none wrongly."""
    names = list(assets)
    tot, n_ok, n = {}, 0, 10
    for i in range(n):
        L, pdf = assets[names[i % len(names)]]
        frame, truth = synth.make_frame(L, pdf, synth.CONDS[cond], np.random.default_rng(4100 + i), serial=1 + i)
        r = recognize(frame)
        if r.status == "ok":
            n_ok += 1
            synth.merge(tot, synth.evaluate(r, truth))
        else:
            assert r.answers == [] and r.student_id is None
    assert n_ok >= 8, n_ok
    assert tot["q_silent"] == 0, tot["failures"]
    assert tot["id_wrong"] == 0 and tot["serial_wrong"] == 0
    assert tot["q_clear_ok"] / tot["q_clear"] >= 0.99


@pytest.mark.parametrize("mode", ["none", "thumb2", "cut"])
def test_negative_frames_never_yield_a_result(assets, mode):
    names = list(assets)
    for i in range(4):
        L, pdf = assets[names[i % len(names)]]
        cond = synth.CONDS["tilted" if i % 2 else "phone"]
        frame, truth = synth.make_frame(L, pdf, cond, np.random.default_rng(900 + i), mode=mode)
        r = recognize(frame)
        if mode == "cut" and truth["n_inside"] >= 3:       # one corner cut off = one marker hidden: a result is fine ...
            if r.status == "ok":
                ev = synth.evaluate(r, truth)              # ... but then it must be right
                assert ev["q_silent"] == 0 and ev["id_wrong"] == 0
            continue
        assert r.status in truth["expect_status"] and r.status != "ok", (mode, i, r.status)
        assert r.answers == [] and r.student_id is None


def test_decoy_markers_cannot_produce_a_result(assets):
    """Background full of marker-like decoys (relaxed border tolerance, streak-suppression passes): never a wrong read."""
    names = list(assets)
    for i in range(8):
        L, pdf = assets[names[i % len(names)]]
        frame, truth = synth.make_frame(L, pdf, synth.CONDS["phone"], np.random.default_rng(1700 + i), mode="none")
        r = recognize(frame)
        assert r.status in ("no_sheet", "partial", "inconsistent") and r.student_id is None and r.answers == []


# ------------------------------------------------------------------ output contract
def test_result_contract(assets):
    frame, truth, L = marked_frame(assets, "q45", seed=9)
    r = recognize(frame, px_per_mm=6.0)
    d = r.to_dict()
    json.dumps(d)                                   # fully serialisable, no image arrays
    assert "warped" not in d and "name_crop" not in d
    assert r.warped.shape == (int(round(297 * 6)), int(round(210 * 6)))
    nb = L.pages[0].name_box
    assert abs(r.name_crop.shape[1] - nb[2] * 6) <= 1 and abs(r.name_crop.shape[0] - nb[3] * 6) <= 1
    assert set(r.marker_corners) == {0, 1, 2, 3}
    assert r.quality["reproj_max_mm"] < 0.5 and r.quality["sharpness"] > 0
    assert {"detect", "warp", "score", "total"} <= set(r.timing_ms)
    # overlay helper: page-space point -> source pixels; the TL marker corner must land on the detected corner
    p = r.image_points([L.pages[0].markers[0].rect[:2]])[0]
    assert np.linalg.norm(p - np.array(r.marker_corners[0][0])) < 2.0
    # per-question detail for the UI
    assert len(r.answers) == 45 and len(r.id_digits) == 8 and len(r.subject_digits) == 2 and r.subject == truth["subject"]
    assert r.answers[0].confidence > 0.5 and len(r.answers[0].scores) == 5


def test_recognize_is_pure(assets):
    frame, _, _ = marked_frame(assets, "q45", seed=4)
    a, b = recognize(frame).to_dict(), recognize(frame).to_dict()
    a.pop("timing_ms"), b.pop("timing_ms")
    assert a == b


def test_no_sheet_frame_is_fast():
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur((rng.random((1080, 1920, 3)) * 255).astype(np.uint8), (0, 0), 3)
    recognize(img)
    ts = sorted(recognize(img).timing_ms["total"] for _ in range(7))
    assert ts[3] < 150   # generous: ~15-30 ms on a laptop; this only guards against an accidental 10x


def _paste_marker(frame, mid, x, y, px=64):
    m = cv2.cvtColor(mk.marker_image(mid, px=px, quiet_cells=1), cv2.COLOR_GRAY2BGR)
    frame[y:y + m.shape[0], x:x + m.shape[1]] = m


def test_foreign_marker_outside_sheet_is_ignored_inside_is_not(assets):
    L, pdf = assets["q45"]
    raster = synth.paper_raster(pdf, 0)
    frame = plain_frame(raster, pad_mm=40)
    _paste_marker(frame, 21, 20, 20)                     # on the desk, far from the sheet
    r = recognize(frame)
    assert r.status == "ok" and r.quality["foreign_markers"] == 1
    frame2 = plain_frame(raster, pad_mm=40)
    h, w = frame2.shape[:2]
    _paste_marker(frame2, 21, w // 2, h // 2)            # on the sheet itself
    assert recognize(frame2).status == "inconsistent"


@pytest.mark.parametrize("cond", SPARSE_CONDS)
def test_sparse_pages_no_silent_errors(assets, cond):
    """Pages with few/no real marks (ghosts only, light pencil only, dark light) must not let residue pose as ink."""
    names = list(assets)
    tot = {}
    for i in range(8):
        L, pdf = assets[names[i % len(names)]]
        frame, truth = synth.make_frame(L, pdf, synth.CONDS[cond], np.random.default_rng(7000 + i))
        r = recognize(frame)
        if r.status == "ok":
            synth.merge(tot, synth.evaluate(r, truth))
    assert tot["q_total"] > 0
    assert tot["q_silent"] == 0, tot["failures"]
    assert tot["id_wrong"] == 0
    if cond == "sp_pencil":   # light pencil marks must be read, or at worst flagged - never dropped
        assert tot["q_clear"] and tot["kinds"]["clear"][1] + tot["kinds"]["clear"][2] >= tot["kinds"]["clear"][0] - 0


@pytest.mark.parametrize("serial", [0, 77])
def test_hand_over_serial_strip_is_not_flagged_unreadable(assets, serial):
    """Strip blank/occluded (no guard pattern visible) is "absent", not "present but garbled": no flag."""
    L, pdf = assets["q45"]
    raster = synth.paper_raster(pdf, 0).copy()
    if serial:
        synth.draw_serial(raster, serial)
    img, truth = synth.mark_sheet(raster, L, 0, np.random.default_rng(5), 0.0, id_hard=False)
    frame = plain_frame(img)
    rects = lay.serial_cell_rects()
    x0 = min(r[0] for r in rects); x1 = max(r[0] + r[2] for r in rects)
    y0 = min(r[1] for r in rects); y1 = max(r[1] + r[3] for r in rects)
    s, pad = 3.2, int(12 * 3.2)
    cv2.rectangle(frame, (int((x0 - 3) * s) + pad, int((y0 - 3) * s) + pad), (int((x1 + 3) * s) + pad, int((y1 + 3) * s) + pad),
                  (120, 150, 210), -1)                      # a hand over the footer
    r = recognize(frame)
    assert r.status == "ok" and r.serial is None and "serial_unreadable" not in r.flags
