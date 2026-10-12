"""Self-describing sheets: exams (= form + subject code) appear by themselves; the subject code routes results;
sheets with an unreadable subject wait in a holding exam until a human moves them; several scanners can work on
one data folder without anyone creating an exam first."""
import dataclasses
import os
import sqlite3
import sys
import threading

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(__file__))
import flow  # noqa: E402
import synth  # noqa: E402
from test_db import result_for_layout  # noqa: E402

from app.main import create_app  # noqa: E402
from omr.layout import Template, build_layout  # noqa: E402
from omr.recognize import recognize  # noqa: E402
from omr.service import OmrService  # noqa: E402
from omr.stream import StreamSession  # noqa: E402

T = Template(20, 4, 4)
L = build_layout(T)


@pytest.fixture
def svc(tmp_path):
    return OmrService(str(tmp_path / "data"))


def res(subject, student="1001", page=0, **kw):
    return result_for_layout(L, subject, page, student, **kw)


def by_subject(svc):
    return {e["subject"]: e for e in svc.list_exams()}


# ------------------------------------------------------------------ exams appear with the first scan
def test_scans_create_their_exam_and_route_by_subject(svc):
    assert svc.list_exams() == []                                  # nothing was created beforehand
    a = svc.ingest(res("07", "1001"))
    b = svc.ingest(res("08", "1001"))                              # the same student number in another subject
    c = svc.ingest(res("07", "1002"))
    assert [x.kind for x in (a, b, c)] == ["new", "new", "new"]
    ex = by_subject(svc)
    assert set(ex) == {"07", "08"} and all(e["form_key"] == L.form_key and e["origin"] == "auto" for e in ex.values())
    assert [ex["07"]["title"], ex["08"]["title"]] == ["과목 07", "과목 08"]
    assert a.exam_id == c.exam_id == ex["07"]["id"] != b.exam_id == ex["08"]["id"]
    assert (a.subject, b.subject) == ("07", "08")
    assert svc.list_students(ex["07"]["id"]) == ["1001", "1002"] and svc.list_students(ex["08"]["id"]) == ["1001"]
    assert svc.ingest(res("08", "1001")).kind == "duplicate"        # within the subject, as always
    # an answer key per subject
    svc.set_answer_key(ex["07"]["id"], [0] * 20)
    assert svc.dashboard(ex["07"]["id"])["students"][0]["score"] == 20 and svc.dashboard(ex["08"]["id"])["students"][0]["score"] is None
    assert svc.exam_overview(ex["07"]["id"])["form_desc"].startswith("20문항")


def test_precreated_exam_is_used_and_keeps_its_title(svc):
    e = svc.create_exam("국어", T, subject="07")
    svc.set_answer_key(e["id"], [1] * 20)
    o = svc.ingest(res("07"))
    assert o.exam_id == e["id"] and len(svc.list_exams()) == 1
    ex = svc.get_exam(e["id"])
    assert ex["title"] == "국어" and ex["origin"] == "manual" and len(ex["answer_key"]) == 20


def test_form_without_subject_code_has_one_exam(svc):
    lay = build_layout(Template(10, 4, 4, subject_digits=0))
    o = svc.ingest(result_for_layout(lay, "", 0, "5"))
    assert o.kind == "new" and o.subject == ""
    e = svc.get_exam(o.exam_id)
    assert e["subject"] == "" and e["title"] == "시험"
    bad = result_for_layout(lay, "07", 0, "5")
    assert svc.ingest(bad).kind == "rejected"                       # a subject cannot be read on a form without one


def test_partly_read_subject_is_treated_as_unreadable(svc):
    for s in ("0", "007", "0x", "", None):
        o = svc.ingest(res(s, f"9{len(str(s))}"))
        assert o.kind == "unassigned" and o.subject is None, s
    assert [e["subject"] for e in svc.list_exams()] == [None]


# ------------------------------------------------------------------ unreadable subject: holding exam + human move
def test_unreadable_subject_waits_and_a_human_moves_it(svc):
    o = svc.ingest(res(None, "1001", marks={1: [1]}))
    assert o.kind == "unassigned" and o.subject is None and o.reason == "subject_unreadable"
    hold = svc.get_exam(o.exam_id)
    assert hold["subject"] is None and hold["title"] == "과목코드 미확인" and hold["form_key"] == L.form_key
    ov = svc.exam_overview(hold["id"])
    assert ov["n_unassigned"] == 1 and ov["n_students"] == 0 and ov["subject_label"] == "과목코드 미확인"
    assert [s["scan_id"] for s in svc.list_unassigned(hold["id"])] == [o.scan_id]
    d = svc.unassigned_detail(hold["id"])
    assert len(d) == 1 and d[0]["subject_raw"] is None
    with pytest.raises(ValueError):
        svc.assign_scan(o.scan_id, "1001")                          # the subject has to be settled first
    with pytest.raises(ValueError):
        svc.route_scan(o.scan_id, None, "1001")
    for bad in ("7", "007", "ab"):
        with pytest.raises(ValueError):
            svc.route_scan(o.scan_id, bad)
    assert svc.list_students(hold["id"]) == [] and len(svc.list_unassigned(hold["id"])) == 1   # failed moves changed nothing

    m = svc.route_scan(o.scan_id, "07")
    assert m.kind == "new" and m.subject == "07" and m.student_id == "1001" and m.scan_id != o.scan_id
    target = svc.get_exam(m.exam_id)
    assert target["subject"] == "07" and target["title"] == "과목 07"
    assert svc.list_unassigned(hold["id"]) == [] and svc.exam_overview(hold["id"])["n_unassigned"] == 0
    assert svc.student_summary(target["id"], "1001")["answers"][1]["marked"] == [1]
    # nothing was mutated: the source scan is still there, marked as moved; the copy points back to it
    kinds = {r["scan_id"]: r["kind"] for r in svc.recent_scans(None, 10)}
    assert m.scan_id in kinds and o.scan_id not in kinds             # global list: the moved-away source is not listed twice
    assert [r["kind"] for r in svc.recent_scans(hold["id"])] == ["moved"]
    c = sqlite3.connect(svc.db_path)
    assert c.execute("SELECT from_scan_id, to_scan_id, from_exam_id, to_exam_id FROM scan_moves").fetchall() == \
        [(o.scan_id, m.scan_id, hold["id"], target["id"])]
    assert c.execute("SELECT moved_from, subject_raw FROM scans WHERE id=?", (m.scan_id,)).fetchone() == (o.scan_id, None)
    assert c.execute("SELECT COUNT(*) FROM actions WHERE kind='move'").fetchone()[0] == 1
    c.close()
    assert svc.scan_count() == 1 and svc.scan_count(hold["id"]) == 1 and svc.exam_overview(target["id"])["n_scans"] == 1
    with pytest.raises(ValueError):
        svc.route_scan(o.scan_id, "08")                              # already moved: move the copy instead
    again = svc.route_scan(m.scan_id, "08")                          # a mistake can be corrected from the copy
    assert again.subject == "08" and svc.list_students(target["id"]) == []


def test_move_with_student_and_duplicates(svc):
    a = svc.ingest(res(None, None, marks={1: [2]}))                 # nothing readable at all
    b = svc.ingest(res(None, "1001", marks={1: [2]}))
    first = svc.route_scan(b.scan_id, "07")
    assert first.kind == "new"
    m = svc.route_scan(a.scan_id, "07", "1001")                      # student entered by hand: the same answers -> confirms
    assert m.kind == "duplicate" and m.student_id == "1001" and m.submission_id == first.submission_id
    c = svc.ingest(res(None, None, marks={1: [3]}))
    n = svc.route_scan(c.scan_id, "07", " 2002 ")
    assert n.kind == "new" and n.student_id == "2002"
    raw = sqlite3.connect(svc.db_path).execute("SELECT student_id_raw FROM scans WHERE id=?", (n.scan_id,)).fetchone()[0]
    assert raw is None                                               # what was READ stays as read; the human decision is an assignment
    # no subject, no change of subject: only the student of a scan that already has its subject
    s = svc.route_scan(n.scan_id, None, "2003")
    assert s.student_id == "2003"
    assert svc.route_scan(n.scan_id, "07", "2004").student_id == "2004"      # same subject + student = plain assignment
    with pytest.raises(ValueError):
        svc.route_scan(n.scan_id, "07")                              # nothing to do
    with pytest.raises(KeyError):
        svc.route_scan(9999, "07")
    nos = svc.ingest(result_for_layout(build_layout(Template(10, 4, 4, subject_digits=0)), "", 0, "5"))
    with pytest.raises(ValueError):
        svc.route_scan(nos.scan_id, "07")                            # form without subject code


def test_rescan_follows_the_subject_decision(svc):
    first = svc.ingest(res(None, "1001", marks={1: [1]}))
    moved = svc.route_scan(first.scan_id, "07")
    again = svc.ingest(res(None, "1001", marks={1: [1]}))            # the teacher shows the same sheet again
    assert again.kind == "duplicate" and again.subject == "07" and again.student_id == "1001"
    assert again.exam_id == moved.exam_id and again.submission_id == moved.submission_id
    hold = first.exam_id
    assert svc.list_unassigned(hold) == []                           # it never reaches the queue
    c = sqlite3.connect(svc.db_path)
    assert c.execute("SELECT COUNT(*) FROM scan_moves WHERE note='auto-follow'").fetchone()[0] == 1
    c.close()
    # a different reading of the same-looking sheet is not followed
    other = svc.ingest(res(None, "1001", marks={1: [3]}))
    assert other.kind == "unassigned" and other.exam_id == hold
    # nor is it when the earlier decisions disagree (no guessing)
    x1, x2 = svc.ingest(res(None, "1009")), svc.ingest(res(None, "1009"))
    svc.route_scan(x1.scan_id, "07")
    svc.route_scan(x2.scan_id, "08")
    assert svc.ingest(res(None, "1009")).kind == "unassigned"
    # a human decision about the student travels with the sheet
    y = svc.ingest(res(None, None, marks={1: [2]}))
    ym = svc.route_scan(y.scan_id, "07", "3003")
    assert svc.ingest(res(None, None, marks={1: [2]})).student_id == "3003" == ym.student_id


def test_moves_never_mutate_scans(svc):
    o = svc.ingest(res(None, "1001"))
    snap = sqlite3.connect(svc.db_path).execute("SELECT * FROM scans ORDER BY id").fetchall()
    svc.route_scan(o.scan_id, "07", "55")
    after = sqlite3.connect(svc.db_path).execute("SELECT * FROM scans ORDER BY id").fetchall()
    assert after[:len(snap)] == snap and len(after) == len(snap) + 1
    c = sqlite3.connect(svc.db_path)
    for sql in ("UPDATE scan_moves SET note='x'", "DELETE FROM scan_moves", "UPDATE scans SET page=3"):
        with pytest.raises(sqlite3.DatabaseError):
            c.execute(sql)
    c.close()


def test_move_leaves_other_students_of_the_old_exam_alone(svc):
    """A scan read as subject 08 by mistake is moved to 07: only that scan changes subject."""
    a = svc.ingest(res("08", "1001", marks={1: [1]}))
    b = svc.ingest(res("08", "1002", marks={1: [2]}))
    svc.route_scan(a.scan_id, "07")
    ex = by_subject(svc)
    assert svc.list_students(ex["08"]["id"]) == ["1002"] and svc.list_students(ex["07"]["id"]) == ["1001"]
    assert svc.get_submission(a.submission_id)["status"] == "empty"
    assert svc.get_submission(b.submission_id)["status"] == "ok"


# ------------------------------------------------------------------ legacy sheets
def test_legacy_sheet_needs_a_registered_layout(tmp_path):
    lay = build_layout(Template(10, 4, 4, subject_digits=0))
    lay.form_id, lay.version, lay.form_key = 0x1234, 0, "F1234"
    r = result_for_layout(lay, "", 0, "77")
    r["form_id"], r["form_key"] = 0x1234, "F1234"
    svc = OmrService(str(tmp_path / "d"))
    o = svc.ingest(r)
    assert o.kind == "rejected" and "legacy" in o.reason and svc.list_exams() == []
    assert svc.resolve_layout(0x1234) is None
    assert svc.register_legacy_layout(lay) == "F1234"
    assert svc.resolve_layout(0x1234).form_id == 0x1234
    o = svc.ingest(r)
    e = svc.get_exam(o.exam_id)
    assert o.kind == "new" and e["form_key"] == "F1234" and e["subject"] == "" and e["title"] == "시험"
    assert svc.ingest(r).kind == "duplicate"
    # the registration is stored: a new service object on the same folder still knows the form
    assert OmrService(str(tmp_path / "d")).resolve_layout(0x1234).form_key == "F1234"
    with pytest.raises(ValueError):
        svc.register_legacy_layout(build_layout(T))                  # self-describing layouts need no registration


# ------------------------------------------------------------------ printing is stateless
def test_form_pdf_needs_no_database_rows(svc):
    a = svc.form_pdf(T)
    b = svc.form_pdf(dataclasses.asdict(T))
    assert a["path"] == b["path"] and os.path.getsize(a["path"]) > 1000 and a["form_key"] == L.form_key
    assert svc.form_pdf(T, title="x")["path"] != a["path"]            # texts differ -> another file, same form_key
    c = sqlite3.connect(svc.db_path)
    assert [c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("exams", "scans", "print_runs")] == [0, 0, 0]
    c.close()
    with pytest.raises(TypeError):
        svc.form_pdf(T, form_id=3)                                    # removed option


# ------------------------------------------------------------------ several scanners, one data folder, no exam created first
@pytest.fixture(scope="module")
def sheets(tmp_path_factory):
    d = tmp_path_factory.mktemp("sheets")
    pdf = synth.build_pdf(L, str(d / "f.pdf"))
    answers = {q: q % 4 for q in range(1, 21)}

    def result(student, subject, seed):
        fixed = {"student_id": student, "answers": answers, "subject": subject}
        frame, _ = synth.make_frame(L, pdf, synth.CONDS["clean"], np.random.default_rng(seed), page=0, fixed=fixed)
        r = recognize(frame)
        assert r.status == "ok"
        return r, frame

    empty, _ = synth.make_frame(L, pdf, synth.CONDS["clean"], np.random.default_rng(1), mode="none")
    return {"result": result, "empty": recognize(empty), "L": L, "pdf": pdf, "answers": answers}


def feed_all(svc, plan, empty, out):
    """One scanner: its own StreamSession, sheets shown one after the other, every commit ingested."""
    s = StreamSession()
    t = 0.0
    for r, frame in plan:
        for fr in [r] * 5 + [empty] * 6:
            ev = s.feed(fr, t, payload=frame)
            t += 0.125
            if "commit" in ev:
                c = ev["commit"]
                out.append(svc.ingest(c["result"], images={"original": c["payload"]}, source="stream"))


def test_two_scanners_one_data_folder_no_exam_created_first(tmp_path, sheets):
    svc = OmrService(str(tmp_path))
    assert svc.list_exams() == []
    mk = sheets["result"]
    plan_a = [mk("1001", "07", 11), mk("1002", "07", 12), mk("1003", None, 13)]
    plan_b = [mk("2001", "08", 21), mk("2002", "08", 22), mk("1004", "07", 23)]
    out_a, out_b = [], []
    ts = [threading.Thread(target=feed_all, args=(svc, p, sheets["empty"], o)) for p, o in ((plan_a, out_a), (plan_b, out_b))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert [o.kind for o in out_a] == ["new", "new", "unassigned"] and [o.kind for o in out_b] == ["new", "new", "new"]
    ex = by_subject(svc)
    assert set(ex) == {"07", "08", None}
    assert svc.list_students(ex["07"]["id"]) == ["1001", "1002", "1004"]       # both scanners fed subject 07
    assert svc.list_students(ex["08"]["id"]) == ["2001", "2002"]
    assert [s["scan_id"] for s in svc.list_unassigned(ex[None]["id"])] == [out_a[2].scan_id]
    # the unreadable one is moved by a human; the dashboard of 07 then shows it
    svc.route_scan(out_a[2].scan_id, "07")
    assert svc.list_students(ex["07"]["id"]) == ["1001", "1002", "1003", "1004"]
    assert svc.dashboard(ex["07"]["id"])["exam"]["n_students"] == 4


def test_two_uploads_and_the_routing_api(tmp_path, sheets):
    app = create_app(str(tmp_path))
    c1, c2 = TestClient(app), TestClient(app)                      # two browsers
    assert c1.get("/api/exams").json() == []
    jp = lambda student, subject, seed: flow.sheet_jpeg(sheets["L"], sheets["pdf"], 0, student, sheets["answers"], seed, subject=subject)
    r1 = flow.upload(c1, [jp("1001", "07", 31), jp("1002", None, 32)])
    r2 = flow.upload(c2, [jp("2001", "08", 33)])
    assert [r["outcome"]["kind"] for r in r1 + r2] == ["new", "unassigned", "new"]
    assert r1[1]["outcome"]["subject"] is None and r1[1]["outcome"]["reason"] in ("subject_blank", "subject_unreadable")
    assert r1[1]["summary"]["subject"] is None and r1[0]["summary"]["subject"] == "07"
    exams = {e["subject"]: e for e in c2.get("/api/exams").json()}
    assert set(exams) == {"07", "08", None}
    hold = exams[None]
    assert hold["subject_label"] == "과목코드 미확인" and hold["n_unassigned"] == 1 and hold["n_scans"] == 1
    d = c1.get(f"/api/exams/{hold['id']}/unassigned").json()
    s = d["scans"][0]
    assert len(s["subject_digits"]) == 2 and len(s["id_digits"]) == 4
    assert d["exam"]["subject"] is None and sorted(s["subjects"]) == ["07", "08"]
    sid = s["scan_id"]
    assert c1.post(f"/api/scans/{sid}/route", json={}).status_code == 400
    assert c1.post(f"/api/scans/{sid}/route", json={"subject": "7"}).status_code == 400
    assert c1.post(f"/api/scans/{sid}/assign", json={"student_id": "1"}).status_code == 400      # subject first
    assert c1.post("/api/scans/9999/route", json={"subject": "07"}).status_code == 404
    r = c1.post(f"/api/scans/{sid}/route", json={"subject": "07"})
    assert r.status_code == 200 and r.json()["kind"] == "new" and r.json()["subject"] == "07" and r.json()["student_id"] == "1002"
    assert c1.post(f"/api/scans/{sid}/route", json={"subject": "08"}).status_code == 400          # already moved
    rows = c2.get(f"/api/exams/{exams['07']['id']}/dashboard").json()["students"]
    assert [x["student_id"] for x in rows] == ["1001", "1002"]
    assert c2.get(f"/api/exams/{hold['id']}/unassigned").json()["scans"] == []
    recent = c2.get("/api/scans/recent?limit=10").json()
    assert sorted((x["subject"], x["student_id"]) for x in recent) == [("07", "1001"), ("07", "1002"), ("08", "2001")]


# ------------------------------------------------------------------ best guesses never become a confirmed student / subject
def test_guessed_subject_goes_to_the_holding_exam_and_guessed_id_is_unassigned_with_a_suggestion(svc):
    r = res("07", "1001")
    r.update(subject_guessed=True, flags=["subject_guessed"], needs_review=True)
    o = svc.ingest(r)
    assert o.kind == "unassigned" and o.subject is None and o.reason == "subject_guessed"
    r = res("07", "1002")
    r.update(id_guessed=True, flags=["id_guessed"], needs_review=True)
    o = svc.ingest(r)
    assert o.kind == "unassigned" and o.subject == "07"                    # right exam, no student
    assert svc.list_students(o.exam_id) == []
    info = svc.unassigned_detail(o.exam_id)[0]
    assert info["suggested_id"] == "1002" and info["suggested_subject"] is None
    svc.assign_scan(o.scan_id, "1002")                                      # a human confirms the suggestion
    assert svc.list_students(o.exam_id) == ["1002"]


def test_guessed_answers_are_graded_and_exported_as_guessed(svc):
    r = res("07", "1001", marks={2: []}, amb=(2, 3))
    r["answers"][1].update(guess=3, guessed=True)             # q2: uncertain blank with a guess
    r["answers"][2].update(state="multi", marked=[0, 1], guess=1, guessed=True)   # (legacy stored guess: ignored)
    o = svc.ingest(r)
    svc.set_answer_key(o.exam_id, [0, 3, 1] + [0] * 17)
    st = svc.student_summary(o.exam_id, "1001")
    assert st["answers"][2]["marked"] == [3] and st["answers"][2]["guessed"] and st["answers"][2]["ambiguous"]
    assert st["answers"][3]["marked"] == [0, 1] and st["answers"][3]["state"] == "multi" and not st["answers"][3]["guessed"]
    assert st["guessed_questions"] == [2] and st["multi_questions"] == [3] and st["status"] == "needs_review"
    g = svc.grade(o.exam_id, "1001")
    assert g["score"] == 19 and g["provisional"]                        # the multi row is wrong against a single-choice key
    row = svc.export_csv(o.exam_id).splitlines()
    assert "guessed:2" in row[1] and "multi:3" in row[1] and ",4,1|2," in row[1]
    svc.set_answer_key(o.exam_id, [0, 3, [[0, 1]]] + [0] * 17)            # a set key: right only for exactly that set
    assert svc.grade(o.exam_id, "1001")["score"] == 20
