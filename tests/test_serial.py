"""Per-printed-copy serial numbers: codec, recognition, stream identity + missed-sheet check, DB, PDF, E2E."""
import copy
import dataclasses
import os
import random
import sys

import cv2
import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import flow  # noqa: E402
import synth  # noqa: E402
from test_db import make_result  # noqa: E402

from app.main import create_app  # noqa: E402
from omr import identity  # noqa: E402
from omr import markers as mk  # noqa: E402
from omr.layout import Template  # noqa: E402
from omr.recognize import recognize  # noqa: E402
from omr.service import OmrService  # noqa: E402
from omr.stream import StreamSession  # noqa: E402

DT = 0.125


# =============================================================== codec / identity
def test_serial_roundtrip_and_bitflips():
    for s in list(range(1, 4096, 37)) + [1, 4095]:
        bits = mk.encode_serial(s)
        assert len(bits) == mk.SERIAL_CELLS and mk.decode_serial(bits) == s
    rnd = random.Random(3)
    for _ in range(60):
        bits = mk.encode_serial(rnd.randint(1, 4095))
        for i in range(len(bits)):
            b = list(bits)
            b[i] = not b[i]
            assert mk.decode_serial(b) is None      # guards + CRC-8 catch every single-bit flip
    for bad in (0, 4096, -1):
        with pytest.raises(ValueError):
            mk.encode_serial(bad)
    assert mk.decode_serial([False] * mk.SERIAL_CELLS) is None


def test_serial_strip_geometry_fits():
    from omr import layout as lay
    r = lay.serial_cell_rects()
    assert len(r) == mk.SERIAL_CELLS and all(c[2] >= 3.0 for c in r)
    right = r[-1][0] + r[-1][2]
    assert right <= lay.PAGE_W - lay.MARGIN - lay.MARKER_MM - lay.QUIET_MM + 1e-6
    assert r[0][0] > lay.MARGIN + lay.MARKER_MM + lay.QUIET_MM + 40      # leaves room for the footer text


def test_serial_strip_clear_of_markers():
    from omr import layout as lay
    xs = lay.serial_cell_rects()
    box = (min(r[0] for r in xs), xs[0][1], max(r[0] + r[2] for r in xs), xs[0][1] + xs[0][3])
    lab = lay.serial_label(4095)
    lab_box = (lab.x - lay._text_width_mm(lab.text, lab.size_pt), lab.y - lab.size_pt * 0.3528, lab.x, lab.y)
    for corner, (mx, my, mw, mh) in lay.marker_rects(0).items():
        for b in (box, lab_box):
            dx = max(mx - b[2], b[0] - (mx + mw), 0)
            dy = max(my - b[3], b[1] - (my + mh), 0)
            assert (dx * dx + dy * dy) ** 0.5 >= 6.0 - 1e-6, (corner, b)


def test_sheet_key_is_string_keyed():
    k = identity.make_sheet_key
    assert k(12, "345") != k(123, "45")            # concatenation with a separator, no numeric collisions
    assert k(1, "1001") != k(2, "1001") and k(0, "1001") == k(0, "1001")
    assert k(1, "1001", "07") != k(1, "1001", "08") and k(1, "1001", "07") != k(1, "1001", "")   # the subject is part of the key
    assert k(1, "1001") == k(1, "1001", "")             # forms without subject code: as before
    assert k(1, "10", "017") != k(1, "100", "17")        # separator between the parts
    assert identity.sheet_key({"serial": 3, "student_id": None, "subject": None,
                               "id_digits": [{"value": 1}, {"value": None}], "subject_digits": [{"value": None}, {"value": 7}]}) \
        == k(3, "1?", "?7")                                # unread digits of either code are '?'
    assert identity.sheet_key({"serial": 3, "student_id": "12", "subject": "07"}) == k(3, "12", "07")


# =============================================================== recognition
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    assets = synth.prepare(str(tmp_path_factory.mktemp("sheets")), {"q30": synth.STANDARD_TEMPLATES["q30_abcd"]})
    L, pdf = assets["q30"]
    return {"L": L, "pdf": pdf}


@pytest.mark.parametrize("cond", ["clean", "tilted", "phone", "lowres", "dark"])
def test_serial_read_at_normal_conditions(world, cond):
    ok = 0
    n = 6
    for i in range(n):
        serial = 100 + 37 * i
        frame, tr = synth.make_frame(world["L"], world["pdf"], synth.CONDS[cond], np.random.default_rng(500 + i),
                                     serial=serial)
        r = recognize(frame)
        if r.status != "ok":
            continue
        assert r.serial in (None, serial)                         # never a wrong number
        ok += r.serial == serial
        if r.serial == serial:
            assert "serial_unreadable" not in r.flags
    assert ok >= n - 1


def test_no_serial_is_quiet_and_unreadable_serial_is_soft(world):
    L, pdf = world["L"], world["pdf"]
    for i in range(4):
        frame, _ = synth.make_frame(L, pdf, synth.CONDS["tilted"], np.random.default_rng(700 + i))
        r = recognize(frame)
        if r.status == "ok":
            assert r.serial is None and "serial_unreadable" not in r.flags
    # damage the strip on the paper: still an ok frame, serial None, informational flag only
    base, _ = synth.make_frame(L, pdf, synth.CONDS["clean"], np.random.default_rng(800), page=0, serial=77)
    ref = recognize(base)
    assert ref.status == "ok" and ref.serial == 77
    raster = synth.paper_raster(pdf, 0).copy()
    marked, tr = synth.mark_sheet(raster, L, 0, np.random.default_rng(1), 0.0)
    synth.draw_serial(marked, 77)
    from omr import layout as lay
    x, y, w, h = lay.serial_cell_rects()[8]
    marked[int(y * synth.PPM_HI):int((y + h) * synth.PPM_HI), int(x * synth.PPM_HI):int((x + w) * synth.PPM_HI)] ^= 255
    cond = synth.CONDS["clean"]
    # reuse the camera path by temporarily serving the damaged raster
    synth._raster_cache[(os.path.abspath(pdf), 0)], saved = marked, synth._raster_cache.get((os.path.abspath(pdf), 0))
    try:
        orig = synth.mark_sheet
        synth.mark_sheet = lambda raster, *a, **k: (raster, tr)
        frame, _ = synth.make_frame(L, pdf, cond, np.random.default_rng(801), page=0)
    finally:
        synth.mark_sheet = orig
        synth._raster_cache[(os.path.abspath(pdf), 0)] = saved
    r = recognize(frame)
    assert r.status == "ok" and r.serial is None and "serial_unreadable" in r.flags
    assert not r.needs_review or r.student_id is None or any(a.ambiguous or a.state != "single" for a in r.answers)


# =============================================================== stream
@pytest.fixture(scope="module")
def sheets(world):
    L, pdf = world["L"], world["pdf"]
    f, _ = synth.make_frame(L, pdf, synth.CONDS["clean"], np.random.default_rng(1), page=0)
    A = recognize(f)
    assert A.status == "ok"
    return {"A": A}


def variant(r, serial, id_unreadable=True):
    r = copy.copy(r)
    r.serial = serial
    if id_unreadable:
        r.student_id = None
        r.id_digits = [dataclasses.replace(d, value=None) for d in r.id_digits]
    return r


def run(session, seq):
    return [session.feed(r, i * DT) for i, r in enumerate(seq)]


def ncommits(evs):
    return sum(1 for e in evs if "commit" in e)


def test_pile_swap_same_answers_unreadable_id_different_serial_two_commits(sheets):
    a, b = variant(sheets["A"], 11), variant(sheets["A"], 12)
    s = StreamSession()
    assert ncommits(run(s, [a] * 7 + [b] * 7)) == 2           # zero interruption frames between them
    # without serials the same pile swap is indistinguishable: today's behaviour (one commit)
    s0 = StreamSession()
    assert ncommits(run(s0, [variant(sheets["A"], 0)] * 7 + [variant(sheets["A"], 0)] * 7)) == 1


def test_same_serial_flicker_one_commit(sheets):
    a = variant(sheets["A"], 21, id_unreadable=False)
    flick = copy.copy(a)
    flick.answers = [dataclasses.replace(x) for x in a.answers]
    flick.answers[0].marked = [(a.answers[0].marked[0] + 1) % 4] if a.answers[0].marked else [0]
    flick.answers[0].state = "single"
    s = StreamSession()
    evs = run(s, [a] * 5 + [flick] * 2 + [a] * 2 + [flick] * 3)
    assert ncommits(evs) == 1


def test_photocopied_master_same_serial_different_ids_degrade_to_student_id(sheets):
    a = variant(sheets["A"], 0, id_unreadable=False)
    b = copy.copy(a)
    b.student_id = "99999999"
    assert ncommits(run(StreamSession(), [a] * 6 + [b] * 6)) == 2
    same_serial_a, same_serial_b = variant(a, 5, False), variant(b, 5, False)
    assert ncommits(run(StreamSession(), [same_serial_a] * 6 + [same_serial_b] * 6)) == 2


def test_missed_sheet_reported(sheets):
    A = sheets["A"]
    a, b, c = (variant(A, n) for n in (31, 32, 33))
    empty = recognize(np.full((480, 640, 3), 200, np.uint8), lambda f: None)
    s = StreamSession()
    evs = run(s, [a] * 6 + [b] * 2 + [c] * 6)                  # b flashes by (2 ok frames), then c is committed
    assert ncommits(evs) == 2
    assert [m["serial"] for m in s.missed()] == [32]
    # a sheet still in view and not yet K-stable is not "missed" yet
    s2 = StreamSession()
    run(s2, [a] * 6 + [b] * 2)
    assert s2.missed() == [] and [m["serial"] for m in s2.missed(final=True)] == [32]
    # re-showing it until it commits clears the report
    s3 = StreamSession()
    run(s3, [a] * 6 + [b] * 2 + [empty] * 6 + [b] * 5)
    assert s3.missed(final=True) == []
    # a one-frame glimpse is not reported
    s4 = StreamSession()
    run(s4, [a] * 6 + [b] + [c] * 6)
    assert s4.missed(final=True) == []


# =============================================================== DB
@pytest.fixture
def svc(tmp_path):
    return OmrService(str(tmp_path / "data"))


@pytest.fixture
def exam(svc):
    return svc.create_exam("Midterm", subject="07", questions=10, choices=4, id_digits=4)


def res(svc, exam, serial, **kw):
    r = make_result(svc, exam, **kw)
    r["serial"] = serial
    return r


def test_same_id_different_serial_is_conflict_even_with_identical_answers(svc, exam):
    o1 = svc.ingest(res(svc, exam, 5))
    assert o1.kind == "new"
    assert svc.ingest(res(svc, exam, 5)).kind == "duplicate"              # same printed copy rescanned
    o3 = svc.ingest(res(svc, exam, 6))                                    # another printed copy, same ID, same answers
    assert o3.kind == "conflict"
    c = svc.list_conflicts(exam["id"])
    assert len(c) == 1 and c[0]["label"] == "같은 수험번호, 다른 시트" and c[0]["different_sheets"]
    assert sorted({s["serial"] for s in c[0]["scans"]}) == [5, 6]
    assert c[0]["diff"][0]["different_sheet"] and not c[0]["minor"]
    # moving the second copy to its real owner clears the conflict
    o4 = svc.resolve_reassign(c[0]["id"], o3.scan_id, "1002")
    assert o4.kind == "new" and svc.list_conflicts(exam["id"]) == []


def test_serial_zero_behaves_as_before(svc, exam):
    assert svc.ingest(res(svc, exam, 0)).kind == "new"
    assert svc.ingest(res(svc, exam, 0)).kind == "duplicate"
    assert svc.ingest(make_result(svc, exam)).kind == "duplicate"         # key absent entirely
    assert svc.ingest(res(svc, exam, 0, marks={1: [1]})).kind == "conflict"
    assert svc.list_conflicts(exam["id"])[0]["different_sheets"] is False


def test_follow_human_routing_uses_sheet_key(svc, exam):
    a = svc.ingest(res(svc, exam, 9, student=None))
    assert a.kind == "unassigned"
    assert svc.assign_scan(a.scan_id, "2002").kind == "new"
    again = svc.ingest(res(svc, exam, 9, student=None))                   # same printed copy shown again
    assert again.kind == "duplicate" and again.student_id == "2002"
    other = svc.ingest(res(svc, exam, 10, student=None))                  # a different copy is not auto-routed
    assert other.kind == "unassigned"


def test_scan_columns_and_database_file(svc, exam, tmp_path):
    o = svc.ingest(res(svc, exam, 7))
    import sqlite3
    row = sqlite3.connect(svc.db_path).execute("SELECT serial, sheet_key, subject_raw FROM scans WHERE id=?", (o.scan_id,)).fetchone()
    assert row[0] == 7 and row[1] == identity.make_sheet_key(7, "1001", "07") and row[2] == "07"
    # a new database file: the pre-redesign omr.sqlite3 is never opened, let alone modified
    assert os.path.basename(svc.db_path) == "streamomr.sqlite3"
    old = tmp_path / "old"
    old.mkdir()
    (old / "omr.sqlite3").write_bytes(b"user data of the previous version")
    OmrService(str(old))
    assert (old / "omr.sqlite3").read_bytes() == b"user data of the previous version"
    assert (old / "streamomr.sqlite3").exists()


def test_print_runs_continue_without_reuse_per_form(svc, exam):
    t = svc.exam_layout(exam["id"]).template
    runs = lambda n, **kw: svc.create_print_run(t, n, **kw)["serials"]
    assert runs(5) == [1, 2, 3, 4, 5]
    assert runs(3) == [6, 7, 8]
    with pytest.raises(ValueError):
        runs(3, start=7)
    with pytest.raises(ValueError):
        runs(10, start=4090)
    assert runs(2, start=20) == [20, 21]
    # another form has its own numbering; the same form under another title shares the numbering (identity = form_key)
    other = dataclasses.replace(t, questions=20)
    assert svc.create_print_run(other, 2)["serials"] == [1, 2]
    assert svc.create_print_run(dataclasses.replace(t, title="another title"), 1)["serials"] == [22]


def test_serial_report_issued_vs_scanned(svc):
    e = svc.create_exam("Two pages", subject="07", questions=20, choices=4, id_digits=4, pages=2)
    svc.create_print_run(svc.exam_layout(e["id"]).template, 6)
    svc.ingest(res(svc, e, 1, page=0, student="1001"))
    svc.ingest(res(svc, e, 1, page=1, student="1001"))
    svc.ingest(res(svc, e, 2, page=0, student="1002"))                    # page 2 of copy 2 never scanned
    svc.ingest(res(svc, e, 4, page=0, student="1004"))
    svc.ingest(res(svc, e, 99, page=0, student="1099"))                   # not issued here
    rep = svc.dashboard(e["id"])["serials"]
    assert rep["n_issued"] == 6 and rep["n_scanned"] == 3
    assert rep["unscanned"] == [[3, 3], [5, 6]] and rep["n_unscanned"] == 3
    assert {p["serial"]: (p["pages_missing"], p["students"]) for p in rep["partial"]} == \
        {2: ([2], ["1002"]), 4: ([2], ["1004"]), 99: ([2], ["1099"])}
    assert rep["unissued"] == [99]


def test_serial_report_spans_the_subjects_of_one_form(svc):
    """One print run is handed out to several subjects: a copy scanned under subject 08 is not 'unscanned' for 07."""
    t = Template(10, 4, 4)
    e7, e8 = svc.create_exam("a", t, subject="07"), svc.create_exam("b", t, subject="08")
    svc.create_print_run(t, 4)
    svc.ingest(res(svc, e7, 1, student="1001"))
    svc.ingest(res(svc, e8, 2, student="1002"))
    for e in (e7, e8):
        rep = svc.serial_report(e["id"])
        assert rep["n_issued"] == 4 and rep["n_scanned"] == 2 and rep["unscanned"] == [[3, 4]]


# =============================================================== PDF
def test_numbered_pdf_copies_have_right_serials_and_pages(tmp_path):
    app = create_app(str(tmp_path))
    c = TestClient(app)
    form = {"questions": 60, "choices": 4, "id_digits": 4, "pages": 2, "title": "N"}
    run = c.post("/api/forms/print-runs", json={**form, "copies": 3, "start": 5}).json()
    assert run["serials"] == [5, 6, 7]
    r = c.get(run["pdf_url"])
    assert c.get(run["pdf_url"]).content == r.content                       # re-fetching does not issue anything
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    pdf = tmp_path / "n.pdf"
    pdf.write_bytes(r.content)
    doc = pymupdf.open(str(pdf))
    assert len(doc) == 3 * 2
    got = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w)
        res_ = recognize(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), px_per_mm=4.0)     # no resolver, no database
        assert res_.status == "ok", (i, res_.status)
        got.append((res_.serial, res_.page))
        text = page.get_text()
        assert f"No. {5 + i // 2:04d}" in text
    assert got == [(5, 0), (5, 1), (6, 0), (6, 1), (7, 0), (7, 1)]
    # the master PDF carries no serial, and the next numbered run continues after the issued range
    master = pymupdf.open(stream=c.get("/api/forms/pdf?questions=60&choices=4&id_digits=4&pages=2&title=N").content, filetype="pdf")
    assert len(master) == 2 and "No. " not in master[0].get_text()
    assert c.post("/api/forms/print-runs", json={**form, "copies": 2}).json()["serials"] == [8, 9]
    assert c.post("/api/forms/print-runs", json={**form, "copies": 2, "start": 6}).status_code == 400
    assert c.post("/api/forms/print-runs", json={**form, "copies": 0}).status_code == 422
    assert c.get("/api/forms/print-runs/999/pdf").status_code == 404
    assert c.get("/api/exams").json() == []                                  # printing never creates an exam


# =============================================================== end to end 100q x 2p
def test_e2e_numbered_copies_100q_2p(tmp_path):
    app = create_app(str(tmp_path))
    c = TestClient(app)
    svc = app.state.service
    form = {"questions": 100, "choices": 5, "id_digits": 8, "pages": 2, "title": "E2E"}
    f = svc.form_pdf(form)
    L, pdf = f["layout"], f["path"]
    assert c.post("/api/forms/print-runs", json={**form, "copies": 4}).status_code == 200   # serials 1..4 issued
    a1, a2 = flow.random_answers(L, 1), flow.random_answers(L, 2)
    S1, S2 = "10000001", "10000002"
    seed = [3000]

    def frame(sid, page, ans, serial):
        seed[0] += 1
        return flow.sheet_jpeg(L, pdf, page, sid, ans, seed[0], serial=serial)

    batch = [(S1, 0, a1, 1), (S1, 1, a1, 1),               # copy 1 complete
             (S2, 0, a2, 2),                               # copy 2: page 2 never scanned
             (S1, 0, a1, 3)]                               # copy 3 with a mismarked ID == S1, identical answers
    res_ = flow.upload(c, [frame(*b) for b in batch])
    assert all(r["status"] == "ok" and r["summary"]["serial"] == b[3] for r, b in zip(res_, batch))
    kinds = [r["outcome"]["kind"] for r in res_]
    assert kinds == ["new", "new", "new", "conflict"]
    eid = res_[0]["outcome"]["exam_id"]
    cf = c.get(f"/api/exams/{eid}/conflicts").json()["conflicts"]
    assert len(cf) == 1 and cf[0]["label"] == "같은 수험번호, 다른 시트"
    d = c.get(f"/api/exams/{eid}/dashboard").json()
    sr = d["serials"]
    assert sr["n_issued"] == 4 and sr["unscanned"] == [[4, 4]]
    assert {p["serial"]: p["pages_missing"] for p in sr["partial"]} == {2: [2], 3: [2]}


def test_websocket_reports_missed_serial(tmp_path):
    app = create_app(str(tmp_path))
    c = TestClient(app)
    f = app.state.service.form_pdf(questions=30, choices=4, id_digits=6)
    L, pdf = f["layout"], f["path"]
    j = {n: flow.sheet_jpeg(L, pdf, 0, f"10000{n}", flow.random_answers(L, n), 40 + n, serial=n) for n in (1, 2, 3)}
    empty = synth.make_frame(L, pdf, synth.CONDS["clean"], np.random.default_rng(9), mode="none")[0]
    ok, buf = cv2.imencode(".jpg", empty)
    e_b = buf.tobytes()
    seen_missed, commits = [], 0
    with c.websocket_connect("/ws/scan") as ws:
        for b in [j[1]] * 6 + [j[2]] * 2 + [j[3]] * 6 + [e_b] * 2:
            ws.send_bytes(b)
            ev = ws.receive_json()
            commits += "commit" in ev
            if ev.get("missed_serials"):
                seen_missed = ev["missed_serials"]
    assert commits == 2 and seen_missed == [2]


def test_replay_summary_lists_missed_serials(tmp_path):
    import replay_video
    f = OmrService(str(tmp_path / "data")).form_pdf(questions=30, choices=4, id_digits=6)
    L, pdf = f["layout"], f["path"]
    cond = synth.CONDS["clean"]
    f = {n: synth.make_frame(L, pdf, cond, np.random.default_rng(60 + n), page=0, serial=n)[0] for n in (1, 2, 3)}
    empty = synth.make_frame(L, pdf, cond, np.random.default_rng(99), mode="none")[0]
    d = tmp_path / "photos"
    d.mkdir()
    for i, fr in enumerate([f[1]] * 6 + [f[2]] * 2 + [f[3]] * 6 + [empty] * 6):
        cv2.imwrite(str(d / f"{i:03d}.png"), fr)
    s = replay_video.main([str(d), "--dry-run"])         # no database, no exam, no layout
    assert [c["serial"] for c in s["commits"]] == [1, 3]
    assert s["missed_serials"] == [2]
