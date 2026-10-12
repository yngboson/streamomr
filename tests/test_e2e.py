"""End-to-end: synthetic camera frames -> /api/upload -> grouping, statuses, grading, CSV."""
import csv
import io
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(__file__))
import flow  # noqa: E402

from app.main import create_app  # noqa: E402

CASES = [("q45_1p", dict(questions=45), 1),
         ("q100_2p", dict(questions=100, pages=2), 2),
         ("q100_3p", dict(questions=100, pages=3), 3)]


def score(answers, truth):
    return sum(1 for q, a in answers.items() if truth.get(q) == a)


@pytest.mark.parametrize("name,params,pages", CASES, ids=[c[0] for c in CASES])
def test_exam_end_to_end(tmp_path, name, params, pages):
    app = create_app(str(tmp_path))
    c = TestClient(app)
    # no exam is created first: only the printed form exists (stateless PDF); the exam appears with the first scan
    svc = app.state.service
    form = svc.form_pdf(title=name, choices=5, id_digits=8, **params)
    L, pdf = form["layout"], form["path"]
    assert c.get("/api/exams").json() == []
    assert L.total_pages == pages
    nq = sum(L.questions_per_page)
    last = pages - 1
    page_qs = {pg.index: range(pg.first_question, pg.first_question + pg.n_questions) for pg in L.pages}

    key = flow.random_answers(L, 100)                      # the teacher's answers
    a1, a2, a3 = (flow.random_answers(L, s) for s in (1, 2, 3))
    # make each student's sheet agree with the key on roughly half of the questions so scores differ
    for a, mod in ((a1, 1), (a2, 2), (a3, 3)):
        for q in a:
            if q % mod == 0:
                a[q] = key[q]
    a3_alt = dict(a3)
    for q in (1, 2, 3):                                    # the conflicting rescan of page 0
        a3_alt[q] = (a3[q] + 1) % 5
    S1, S2, S3, S4 = "10000001", "10000002", "10000003", "10000004"

    seed = [1000]

    def frame(sid, page, answers):
        seed[0] += 1
        return flow.sheet_jpeg(L, pdf, page, sid, answers, seed[0])

    batch = []                                              # (label, jpeg)
    for p in range(pages):
        batch.append((("S1", p), frame(S1, p, a1)))
    for p in range(pages - 1 if pages > 1 else 1):         # S2 lacks the last page (when there are several)
        batch.append((("S2", p), frame(S2, p, a2)))
    for p in range(pages):
        batch.append((("S3", p), frame(S3, p, a3)))
    batch.append((("S1dup", 0), frame(S1, 0, a1)))          # duplicate rescan
    batch.append((("S3alt", 0), frame(S3, 0, a3_alt)))      # conflicting rescan
    # an unreadable ID: S2's missing last page (several pages) or a brand-new student (single page)
    un_ans, un_page = (a2, last) if pages > 1 else (flow.random_answers(L, 4), 0)
    batch.append((("unreadable", un_page), frame(None, un_page, un_ans)))

    res = flow.upload(c, [b for _, b in batch])
    out = {lab: r for (lab, _), r in zip(batch, res)}
    for lab, r in out.items():
        assert r["status"] == "ok", (lab, r)
    kinds = {lab: r["outcome"]["kind"] for lab, r in out.items()}
    exams = c.get("/api/exams").json()
    assert len(exams) == 1 and exams[0]["subject"] == "07" and exams[0]["title"] == "과목 07" and exams[0]["form_key"] == L.form_key
    eid = exams[0]["id"]
    assert exams[0]["total_pages"] == pages and exams[0]["n_questions"] == params["questions"]
    assert kinds[("S1dup", 0)] == "duplicate" and out[("S1dup", 0)]["outcome"]["confirm_count"] == 2
    assert kinds[("S3alt", 0)] == "conflict"
    assert kinds[("unreadable", un_page)] == "unassigned"
    assert all(kinds[("S1", p)] == "new" for p in range(pages))
    assert out[("S1", 0)]["outcome"]["student_id"] == S1

    # ---- grouping and statuses (no key yet)
    d = c.get(f"/api/exams/{eid}/dashboard").json()
    rows = {s["student_id"]: s for s in d["students"]}
    assert set(rows) == {S1, S2, S3}
    assert rows[S1]["status"] == "complete" and rows[S1]["pages_present"] == list(range(1, pages + 1))
    assert rows[S3]["status"] == "conflict"
    if pages > 1:
        assert rows[S2]["status"] == "incomplete" and rows[S2]["pages_missing"] == [pages]
    else:
        assert rows[S2]["status"] == "complete"
    assert all(r["score"] is None for r in rows.values())
    assert d["exam"]["n_conflicts"] == 1 and d["exam"]["n_unassigned"] == 1 and d["exam"]["n_students"] == 3

    # ---- unassigned queue: reading, suggestions, manual assignment
    un = c.get(f"/api/exams/{eid}/unassigned").json()["scans"]
    assert len(un) == 1 and un[0]["page"] == un_page
    if pages > 1:
        assert S2 in un[0]["suggestions"] and S1 not in un[0]["suggestions"]
    target = S2 if pages > 1 else S4
    r = c.post(f"/api/scans/{un[0]['scan_id']}/assign", json={"student_id": target})
    assert r.status_code == 200 and r.json()["kind"] == "new"
    rows = {s["student_id"]: s for s in c.get(f"/api/exams/{eid}/dashboard").json()["students"]}
    assert rows[S2]["status"] == "complete"
    assert (S4 in rows) == (pages == 1)
    assert c.get(f"/api/exams/{eid}/unassigned").json()["scans"] == []

    # ---- answer key + grading
    assert c.put(f"/api/exams/{eid}/answer-key", json={"key": flow.accepted_key(key)}).status_code == 200
    rows = {s["student_id"]: s for s in c.get(f"/api/exams/{eid}/dashboard").json()["students"]}
    assert rows[S1]["score"] == score(a1, key) and rows[S1]["max_score"] == nq and rows[S1]["provisional"] is False
    assert rows[S2]["score"] == score(a2, key) and rows[S2]["provisional"] is False
    # S3 is in conflict: provisional; its score is one of the two readings
    assert rows[S3]["provisional"] is True and rows[S3]["status"] == "conflict"
    assert rows[S3]["score"] in (score(a3, key), score(a3_alt, key))
    # resolve in favour of the first scan -> exact score, complete
    cf = c.get(f"/api/exams/{eid}/conflicts").json()["conflicts"]
    assert len(cf) == 1 and cf[0]["student_id"] == S3 and cf[0]["page"] == 0
    first = out[("S3", 0)]["outcome"]["scan_id"]
    assert c.post(f"/api/submissions/{cf[0]['id']}/adopt", json={"scan_id": first}).status_code == 200
    rows = {s["student_id"]: s for s in c.get(f"/api/exams/{eid}/dashboard").json()["students"]}
    assert rows[S3]["status"] == "complete" and rows[S3]["score"] == score(a3, key) and rows[S3]["provisional"] is False

    # ---- student detail agrees with the ground truth
    st = c.get(f"/api/exams/{eid}/students/{S1}").json()
    assert [q["marked"] for q in st["questions"]] == [[a1[q]] for q in range(1, nq + 1)]
    assert sum(1 for q in st["questions"] if q["correct"]) == score(a1, key)
    assert len(st["scans"]) == pages + 1                    # one scan per page plus the duplicate rescan

    # ---- CSV
    r = c.get(f"/api/exams/{eid}/export.csv")
    assert r.content[:3] == b"\xef\xbb\xbf"
    table = list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))
    byid = {row["student_id"]: row for row in table}
    expected = {S1, S2, S3} | ({S4} if pages == 1 else set())
    assert set(byid) == expected
    labels = L.template.choice_labels
    for sid, ans in ((S1, a1), (S3, a3)):
        row = byid[sid]
        assert row["status"] == "complete" and int(row["score"]) == score(ans, key) and int(row["max_score"]) == nq
        assert row["provisional"] == "false" and row["pages_present"] == ",".join(map(str, range(1, pages + 1)))
        assert [row[f"q{q}"] for q in range(1, nq + 1)] == [labels[ans[q]] for q in range(1, nq + 1)]
    assert list(table[0].keys())[-1] == f"q{nq}"
