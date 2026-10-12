"""DB / policy / grading tests.  No camera pipeline: results are built as RecognitionResult-like dicts."""
import json
import sqlite3
import threading

import numpy as np
import pytest

from omr.layout import Template, build_layout
from omr.service import OmrService


@pytest.fixture
def svc(tmp_path):
    return OmrService(str(tmp_path / "data"))


def make_result(svc, exam, page=0, student="1001", marks=None, amb=(), conf=0.95, status="ok", flags=(), subject=None):
    """marks: {question: [choice,...]} (default: choice 0 for every question)."""
    lay = svc.exam_layout(exam["id"])
    pg = lay.pages[page]
    marks = marks or {}
    return result_for_layout(lay, exam["subject"] if subject is None else (None if subject == "?" else subject), page, student, marks, amb, conf, status, flags)


KEEP = object()


def result_for_layout(lay, subject, page=0, student="1001", marks=None, amb=(), conf=0.95, status="ok", flags=()):
    """A recognition result as `recognize` would return it for a sheet of layout `lay` (no exam has to exist)."""
    pg = lay.pages[page]
    marks = marks or {}
    answers = []
    for q in range(pg.first_question, pg.first_question + pg.n_questions):
        m = marks.get(q, [0])
        answers.append({"number": q, "state": "blank" if not m else "single" if len(m) == 1 else "multi",
                        "marked": list(m), "labels": [], "confidence": 0.5 if q in amb else conf,
                        "ambiguous": q in amb, "scores": []})
    return {"status": status, "form_key": lay.form_key, "form_id": lay.form_id, "page": page, "total_pages": lay.total_pages,
            "subject": subject, "subject_digits": [], "student_id": student, "answers": answers,
            "flags": list(flags) + (["subject_unreadable"] if subject is None else []), "id_digits": [], "quality": {},
            "needs_review": student is None or subject is None or bool(amb)}


def scans_snapshot(svc):
    c = sqlite3.connect(svc.db_path)
    try:
        return c.execute("SELECT * FROM scans ORDER BY id").fetchall()
    finally:
        c.close()


@pytest.fixture
def exam(svc):
    return svc.create_exam("Midterm", subject="07", questions=10, choices=4, id_digits=4)


# ------------------------------------------------------------------ exams = (form, subject)
def test_exam_is_a_form_subject_pair(svc):
    t = Template(20, 5, 6)
    e1 = svc.create_exam("A", t, subject="07")
    e2 = svc.create_exam("A", t, subject="08")
    assert e1["form_key"] == e2["form_key"] == build_layout(t).form_key and (e1["subject"], e2["subject"]) == ("07", "08")
    assert svc.ingest(make_result(svc, e1)).exam_id == e1["id"]
    assert svc.ingest(make_result(svc, e2)).exam_id == e2["id"]
    with pytest.raises(ValueError):
        svc.create_exam("again", t, subject="07")           # already there
    with pytest.raises(ValueError):
        svc.create_exam("x", t, subject="7")                 # wrong number of digits
    with pytest.raises(ValueError):
        svc.create_exam("x", t, subject="ab")
    with pytest.raises(ValueError):
        svc.create_exam("x", t)                              # this form has a subject code: it is needed
    with pytest.raises(ValueError):
        svc.create_exam("x", Template(20, 5, 6, subject_digits=0), subject="07")
    nos = svc.create_exam("nosubject", Template(20, 5, 6, subject_digits=0))
    assert nos["subject"] == ""


def test_exam_titles_and_rename(svc):
    e = svc.create_exam("", Template(10, 4, 4), subject="07")
    assert e["title"] == "과목 07" and e["origin"] == "manual"
    assert svc.rename_exam(e["id"], "  국어  ")["title"] == "국어"
    with pytest.raises(ValueError):
        svc.rename_exam(e["id"], "  ")
    with pytest.raises(KeyError):
        svc.rename_exam(999, "x")


# ------------------------------------------------------------------ ingest policy
def test_new_and_duplicates(svc, exam):
    o = svc.ingest(make_result(svc, exam))
    assert o.kind == "new" and o.confirm_count == 1
    for i in range(5):
        d = svc.ingest(make_result(svc, exam, conf=0.8 + i * 0.01))  # confidences differ, content same
        assert d.kind == "duplicate" and d.submission_id == o.submission_id
    sub = svc.get_submission(o.submission_id)
    assert sub["confirm_count"] == 6 and sub["status"] == "ok" and len(sub["scans"]) == 6
    assert svc.scan_count() == 6  # duplicates are still stored


def test_conflict_diff_and_provisional_rule(svc, exam):
    a = svc.ingest(make_result(svc, exam, marks={3: [1]}))
    b = svc.ingest(make_result(svc, exam, marks={3: [2], 5: [3]}, amb=(7,)))  # worse: has an ambiguous answer
    assert b.kind == "conflict" and not b.minor
    sub = svc.get_submission(a.submission_id)
    assert sub["status"] == "conflict" and sub["adopted_scan_id"] == a.scan_id  # fewer doubtful answers wins
    qs = {d["question"] for d in sub["diff"][0]["questions"]}
    assert qs == {3, 5}  # q7 differs only in ambiguity flag, not in decision content
    assert len(sub["scans"]) == 2
    # a better third scan (all confident, different content) takes over provisionally
    c = svc.ingest(make_result(svc, exam, marks={3: [2]}, conf=0.99))
    assert c.kind == "conflict"
    assert svc.get_submission(a.submission_id)["adopted_scan_id"] == c.scan_id
    # re-seeing any known version is a duplicate, not a new conflict
    assert svc.ingest(make_result(svc, exam, marks={3: [2]})).kind == "duplicate"


def test_tie_prefers_higher_confidence_then_newer(svc, exam):
    a = svc.ingest(make_result(svc, exam, marks={1: [1]}, conf=0.90))
    svc.ingest(make_result(svc, exam, marks={1: [2]}, conf=0.97))
    assert svc.get_submission(a.submission_id)["adopted_scan_id"] != a.scan_id
    e2 = svc.create_exam("T", subject="08", questions=10, choices=4)
    a = svc.ingest(make_result(svc, e2, marks={1: [1]}, conf=0.9))
    b = svc.ingest(make_result(svc, e2, marks={1: [2]}, conf=0.9))
    assert svc.get_submission(a.submission_id)["adopted_scan_id"] == b.scan_id  # newer


def test_minor_conflict(svc, exam):
    svc.ingest(make_result(svc, exam, marks={4: [1]}, amb=(4,)))
    o = svc.ingest(make_result(svc, exam, marks={4: [2]}))  # confident read of the doubtful question
    assert o.kind == "conflict" and o.minor
    sub = svc.get_submission(o.submission_id)
    assert sub["minor"] and sub["status"] == "conflict"
    assert sub["adopted_scan_id"] == o.scan_id
    # an unrelated further difference makes it non-minor
    o2 = svc.ingest(make_result(svc, exam, marks={4: [2], 6: [3]}))
    assert not o2.minor


def test_resolve_adopt_and_rescans(svc, exam):
    a = svc.ingest(make_result(svc, exam, marks={2: [1]}))
    b = svc.ingest(make_result(svc, exam, marks={2: [2]}, amb=(9,)))
    sid = a.submission_id
    sub = svc.resolve_adopt(sid, b.scan_id)
    assert sub["status"] == "ok" and sub["adopted_scan_id"] == b.scan_id and sub["adopted_by"] == "human"
    assert svc.ingest(make_result(svc, exam, marks={2: [2]}, amb=(9,))).kind == "duplicate"
    # matches the rejected scan -> differs from the adopted -> reopens, human choice stays provisional
    o = svc.ingest(make_result(svc, exam, marks={2: [1]}))
    assert o.kind == "conflict"
    sub = svc.get_submission(sid)
    assert sub["status"] == "conflict" and sub["adopted_scan_id"] == b.scan_id
    with pytest.raises(ValueError):
        svc.resolve_adopt(sid, 9999)


def test_resolve_by_reassign_to_other_student(svc, exam):
    a = svc.ingest(make_result(svc, exam, student="1001", marks={1: [1]}, ))
    b = svc.ingest(make_result(svc, exam, student="1001", marks={1: [2]}))  # actually student 1002
    sid = a.submission_id
    sub = svc.get_submission(sid)
    assert all(s["name_crop_path"] is None or isinstance(s["name_crop_path"], str) for s in sub["scans"])
    o = svc.resolve_reassign(sid, b.scan_id, "1002")
    assert o.kind == "new" and o.student_id == "1002"
    old = svc.get_submission(sid)
    assert old["status"] == "ok" and [s["scan_id"] for s in old["scans"]] == [a.scan_id]
    assert svc.student_summary(exam["id"], "1002")["pages_present"] == [0]
    assert svc.student_summary(exam["id"], "1002")["answers"][1]["marked"] == [2]
    # reassigning the only scan of a submission empties it
    svc.assign_scan(a.scan_id, "1003")
    assert svc.get_submission(sid)["status"] == "empty"
    assert svc.list_students(exam["id"]) == ["1002", "1003"]


def test_conflict_exposes_name_crops(svc, exam):
    crop = np.full((20, 40), 200, np.uint8)
    a = svc.ingest(make_result(svc, exam, marks={1: [1]}), images={"name_crop": crop, "warped": crop})
    svc.ingest(make_result(svc, exam, marks={1: [2]}), images={"name_crop": crop})
    sub = svc.get_submission(a.submission_id)
    paths = [s["name_crop_path"] for s in sub["scans"]]
    assert len(set(paths)) == 2 and all(p and p.endswith(".png") for p in paths)
    import os
    assert all(os.path.exists(p) for p in paths)
    assert svc.list_conflicts(exam["id"])[0]["id"] == a.submission_id


def test_unassigned_then_assign_joins_policy(svc, exam):
    u = svc.ingest(make_result(svc, exam, student=None, marks={1: [1]}))
    assert u.kind == "unassigned" and [s["scan_id"] for s in svc.list_unassigned(exam["id"])] == [u.scan_id]
    o = svc.assign_scan(u.scan_id, "2001")
    assert o.kind == "new" and svc.list_unassigned(exam["id"]) == []
    # an identical unreadable sheet follows the human decision straight away (see routing tests)
    u2 = svc.ingest(make_result(svc, exam, student=None, marks={1: [1]}))
    assert u2.kind == "duplicate" and u2.student_id == "2001"
    # a differing one stays in the queue; assigning it by hand -> conflict
    u3 = svc.ingest(make_result(svc, exam, student=None, marks={1: [3]}))
    assert u3.kind == "unassigned"
    assert svc.assign_scan(u3.scan_id, "2001").kind == "conflict"
    assert svc.assign_scan(u3.scan_id, "2001").kind == "unchanged"
    # back to the queue
    assert svc.assign_scan(u3.scan_id, None).kind == "unassigned"
    assert svc.get_submission(o.submission_id)["status"] == "ok"
    assert len(svc.list_unassigned(exam["id"])) == 1


def test_rejected(svc, exam):
    assert svc.ingest(make_result(svc, exam, status="partial")).kind == "rejected"
    assert svc.ingest(make_result(svc, exam, status="no_sheet")).kind == "rejected"
    bad = make_result(svc, exam)
    bad["form_key"] = "garbage"
    assert svc.ingest(bad).kind == "rejected"
    bad = make_result(svc, exam)
    bad["form_id"], bad["form_key"] = 0xABCD, "FABCD"          # a legacy sheet whose layout was never registered
    o = svc.ingest(bad)
    assert o.kind == "rejected" and "legacy" in o.reason
    bad = make_result(svc, exam)
    bad["page"] = 3
    assert svc.ingest(bad).kind == "rejected"
    bad = make_result(svc, exam)
    bad["answers"] = bad["answers"][:-1]
    assert svc.ingest(bad).kind == "rejected"
    assert svc.scan_count() == 0


def test_scans_are_never_mutated(svc, exam):
    results = [make_result(svc, exam, marks={1: [1]}), make_result(svc, exam, marks={1: [2]}),
               make_result(svc, exam, student=None), make_result(svc, exam, student="77")]
    outs = [svc.ingest(r) for r in results]
    snap = scans_snapshot(svc)
    svc.assign_scan(outs[2].scan_id, "1001")
    svc.resolve_adopt(outs[0].submission_id, outs[1].scan_id)
    svc.resolve_reassign(outs[0].submission_id, outs[0].scan_id, "555")
    svc.override_answer(outs[3].submission_id, 1, [2])
    svc.ingest(make_result(svc, exam, marks={1: [1]}))
    after = scans_snapshot(svc)
    assert after[:len(snap)] == snap and len(after) == len(snap) + 1
    c = sqlite3.connect(svc.db_path)
    with pytest.raises(sqlite3.DatabaseError):
        c.execute("UPDATE scans SET page=5")
    with pytest.raises(sqlite3.DatabaseError):
        c.execute("DELETE FROM scans")
    c.close()


def test_concurrent_ingest(svc, exam):
    errs = []
    kinds = []

    def work(i):
        try:
            kinds.append(svc.ingest(make_result(svc, exam, student="9", conf=0.5 + i / 100)).kind)
        except Exception as e:  # pragma: no cover
            errs.append(e)
    ts = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs
    assert kinds.count("new") == 1 and kinds.count("duplicate") == 15
    c = sqlite3.connect(svc.db_path)
    assert c.execute("SELECT COUNT(*) FROM submissions").fetchone()[0] == 1
    assert c.execute("SELECT confirm_count FROM submissions").fetchone()[0] == 16
    c.close()


# ------------------------------------------------------------------ multi-page grouping
@pytest.mark.parametrize("q,pages", [(45, 1), (100, 2), (100, 3)])
def test_multipage_grouping(svc, q, pages):
    e = svc.create_exam(f"E{q}", subject="01", questions=q, pages=pages if pages > 1 else None)
    lay = svc.exam_layout(e["id"])
    assert lay.total_pages == pages
    for p in range(pages - 1):
        svc.ingest(make_result(svc, e, page=p, student="42"))
        st = svc.student_summary(e["id"], "42")
        assert st["status"] == "incomplete" and st["pages_missing"] == list(range(p + 1, pages))
    svc.ingest(make_result(svc, e, page=pages - 1, student="42"))
    st = svc.student_summary(e["id"], "42")
    assert st["status"] == "complete" and st["pages_missing"] == []
    assert sorted(st["answers"]) == list(range(1, q + 1))  # global numbering
    # conflict on one page dominates
    if pages > 1:
        svc.ingest(make_result(svc, e, page=0, student="42", marks={1: [3]}))
        assert svc.student_summary(e["id"], "42")["status"] == "conflict"


def test_needs_review_status_and_override(svc, exam):
    o = svc.ingest(make_result(svc, exam, marks={3: [1]}, amb=(3,)))
    st = svc.student_summary(exam["id"], "1001")
    assert st["status"] == "needs_review" and st["doubtful_questions"] == [3]
    svc.override_answer(o.submission_id, 3, [2], note="checked by eye")
    st = svc.student_summary(exam["id"], "1001")
    assert st["status"] == "complete" and st["answers"][3]["marked"] == [2] and st["answers"][3]["source"] == "override"
    svc.override_answer(o.submission_id, 3, None)  # remove override
    assert svc.student_summary(exam["id"], "1001")["answers"][3]["source"] == "scan"
    with pytest.raises(ValueError):
        svc.override_answer(o.submission_id, 99, [0])
    with pytest.raises(ValueError):
        svc.override_answer(o.submission_id, 1, [7])


# ------------------------------------------------------------------ key + grading + csv
def test_answer_key_from_scan_and_grading(svc, exam):
    k = svc.ingest(make_result(svc, exam, student="0000", marks={1: [1], 2: [2], 3: [], 4: [0, 1]}, amb=(2,)))
    res = svc.answer_key_from_scan(k.scan_id)
    assert res["skipped"] == [3] and res["ambiguous"] == [2]
    key = svc.get_exam(exam["id"])["answer_key"]
    assert key[1] == [[1]] and key[4] == [[0, 1]] and 3 not in key
    # student: q1 right, q2 wrong, q4 multi exactly right, q5.. default choice 0 vs key 0 right
    svc.ingest(make_result(svc, exam, student="1", marks={1: [1], 2: [3], 4: [0, 1]}))
    g = svc.grade(exam["id"], "1")
    assert g["per_question"][1]["correct"] and not g["per_question"][2]["correct"] and g["per_question"][4]["correct"]
    assert g["max_score"] == 9 and g["score"] == 8 and not g["provisional"]


def test_grading_rules(svc):
    exam = svc.create_exam("labels", subject="01", questions=10, choices=4, choice_labels=list("ABCD"))
    svc.set_answer_key(exam["id"], {1: 0, 2: [0, 2], 3: [[1, 2]], 4: "D", 5: 1}, points={5: 2.5})
    marks = {1: [], 2: [2], 3: [1, 2], 4: [3], 5: [1, 2]}
    o = svc.ingest(make_result(svc, exam, student="5", marks=marks, amb=(2,)))
    g = svc.grade(exam["id"], "5")
    pq = g["per_question"]
    assert not pq[1]["correct"]            # blank = wrong
    assert pq[2]["correct"]                # any-of
    assert pq[3]["correct"]                # exact set
    assert pq[4]["correct"]                # label key
    assert not pq[5]["correct"]            # multi where key is single
    assert g["max_score"] == 1 + 1 + 1 + 1 + 2.5 and g["score"] == 3
    assert g["provisional"] and g["reasons"] == ["ambiguous"]
    svc.override_answer(o.submission_id, 2, [2])       # resolved by human -> no longer ambiguous
    assert not svc.grade(exam["id"], "5")["provisional"]
    with pytest.raises(ValueError):
        svc.set_answer_key(exam["id"], {1: 9})
    with pytest.raises(ValueError):
        svc.set_answer_key(exam["id"], {11: 0})
    e2 = svc.create_exam("nokey", subject="01", questions=5)
    with pytest.raises(ValueError):
        svc.grade(e2["id"], "1")


def test_grading_incomplete_is_provisional(svc):
    e = svc.create_exam("two", subject="01", questions=100, pages=2)
    svc.set_answer_key(e["id"], [0] * 100)
    svc.ingest(make_result(svc, e, page=0, student="8"))
    g = svc.grade(e["id"], "8")
    assert g["score"] == 50 and g["max_score"] == 100 and g["provisional"] and "incomplete" in g["reasons"]


def test_export_csv(svc):
    e = svc.create_exam("csv", subject="01", questions=6, choices=4, choice_labels=list("ABCD"))
    svc.set_answer_key(e["id"], {1: 0, 2: 1, 3: 2, 4: 3, 5: 0, 6: 0})
    svc.ingest(make_result(svc, e, student="1", marks={1: [0], 2: [1], 3: [2], 4: [3], 5: [0], 6: [0]}))
    svc.ingest(make_result(svc, e, student="2", marks={2: [3], 3: [], 4: [1, 2]}, amb=(5,)))
    import csv, io
    rows = list(csv.DictReader(io.StringIO(svc.export_csv(e["id"]))))
    assert [r["student_id"] for r in rows] == ["1", "2"]
    r1, r2 = rows
    assert r1["status"] == "complete" and r1["score"] == "6" and r1["max_score"] == "6" and r1["provisional"] == "false"
    assert [r1[f"q{i}"] for i in range(1, 7)] == list("ABCDAA")
    assert r2["status"] == "needs_review" and r2["score"] == "3" and r2["provisional"] == "true"
    assert [r2[f"q{i}"] for i in range(1, 7)] == ["A", "D", "", "B|C", "A", "A"]
    assert "doubtful:4,5" in r2["flags"]  # multi-mark and ambiguous


# ------------------------------------------------------------------ human routing is remembered
def test_rescan_follows_reassign(svc, exam):
    a = svc.ingest(make_result(svc, exam, student="1001", marks={1: [1]}))
    b = svc.ingest(make_result(svc, exam, student="1001", marks={1: [2]}))
    svc.resolve_reassign(a.submission_id, b.scan_id, "1002")
    # teacher rescans the pile
    ra = svc.ingest(make_result(svc, exam, student="1001", marks={1: [1]}))
    rb = svc.ingest(make_result(svc, exam, student="1001", marks={1: [2]}))
    assert ra.kind == "duplicate" and ra.student_id == "1001"
    assert rb.kind == "duplicate" and rb.student_id == "1002"
    assert svc.get_submission(a.submission_id)["status"] == "ok"
    assert svc.list_conflicts(exam["id"]) == []
    c = sqlite3.connect(svc.db_path)
    assert c.execute("SELECT COUNT(*) FROM actions WHERE kind='follow_routing'").fetchone()[0] == 1
    c.close()


def test_rescan_follows_assignment_of_unreadable_id(svc, exam):
    def res():
        r = make_result(svc, exam, student=None, marks={1: [1]})
        r["id_digits"] = [{"index": 0, "state": "single", "value": 2}, {"index": 1, "state": "blank", "value": None}]
        return r
    u = svc.ingest(res())
    o = svc.assign_scan(u.scan_id, "2000")
    again = svc.ingest(res())
    assert again.kind == "duplicate" and again.student_id == "2000" and again.submission_id == o.submission_id
    assert svc.list_unassigned(exam["id"]) == []
    # a different partial reading is NOT matched
    other = res()
    other["id_digits"][0]["value"] = 3
    assert svc.ingest(other).kind == "unassigned"
    # different content with the same reading is not matched either
    r = res()
    r["answers"][0]["marked"], r["answers"][0]["state"] = [3], "single"
    assert svc.ingest(r).kind == "unassigned"


def test_ambiguous_routing_falls_back(svc, exam):
    def res():
        r = make_result(svc, exam, student=None)
        r["id_digits"] = [{"index": 0, "state": "blank", "value": None}]
        return r
    u1, u2 = svc.ingest(res()), svc.ingest(res())
    svc.assign_scan(u1.scan_id, "3001")
    svc.assign_scan(u2.scan_id, "3002")          # same-looking sheets routed to different students
    assert svc.ingest(res()).kind == "unassigned"  # no guessing
    # same for a readable id routed two ways
    a1 = svc.ingest(make_result(svc, exam, student="4000"))
    a2 = svc.ingest(make_result(svc, exam, student="4000"))
    svc.assign_scan(a1.scan_id, "4001")
    svc.assign_scan(a2.scan_id, "4002")
    o = svc.ingest(make_result(svc, exam, student="4000"))
    assert o.kind == "new" and o.student_id == "4000"   # normal path, not routed to either


def test_original_saved_as_jpeg(svc, exam):
    img = np.full((60, 80), 128, np.uint8)
    o = svc.ingest(make_result(svc, exam), images={"original": img, "warped": img})
    sub = svc.get_submission(o.submission_id)["scans"][0]
    assert sub["original_path"].endswith(".jpg") and sub["warped_path"].endswith(".png")


def test_recent_scans(svc, exam):
    svc.ingest(make_result(svc, exam, student="1001"))
    svc.ingest(make_result(svc, exam, student="1001"))
    rs = svc.recent_scans(exam["id"], limit=5)
    assert [r["kind"] for r in rs] == ["duplicate", "new"] and rs[0]["student_id"] == "1001" and rs[0]["scan_id"] > rs[1]["scan_id"]
    assert rs[0]["subject"] == "07" and rs[0]["exam_id"] == exam["id"]
    assert len(svc.recent_scans(exam["id"], limit=1)) == 1
    assert [r["scan_id"] for r in svc.recent_scans(None, limit=5)] == [r["scan_id"] for r in rs]      # all exams
