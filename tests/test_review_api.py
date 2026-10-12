import os
import sqlite3
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(__file__))
import flow  # noqa: E402

from app.main import create_app  # noqa: E402

LABELS = ["A", "B", "C", "D"]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    app = create_app(data)
    c = TestClient(app)
    svc = app.state.service
    exam, L, pdf = flow.make_exam(c, svc, exam_title="퀴즈 <b>", questions=12, choices=4, id_digits=6, labels="A")
    ans_a = flow.random_answers(L, 1)
    ans_b = flow.random_answers(L, 2)
    ans_b2 = dict(ans_b)
    ans_b2[3] = (ans_b[3] + 1) % 4
    ans_b2[7] = (ans_b[7] + 2) % 4
    frames = [flow.sheet_jpeg(L, pdf, 0, "111111", ans_a, 10),
              flow.sheet_jpeg(L, pdf, 0, "222222", ans_b, 11),
              flow.sheet_jpeg(L, pdf, 0, "222222", ans_b2, 12),
              flow.sheet_jpeg(L, pdf, 0, None, ans_a, 13)]
    res = flow.upload(c, frames)
    kinds = [r["outcome"]["kind"] for r in res]
    assert kinds == ["new", "new", "conflict", "unassigned"], kinds
    return dict(app=app, c=c, svc=svc, exam=exam, id=exam["id"], data=data, ans_a=ans_a, ans_b=ans_b, ans_b2=ans_b2,
                res=res, L=L, pdf=pdf)


def test_exam_overview_and_validation(env):
    c, i = env["c"], env["id"]
    e = c.get(f"/api/exams/{i}").json()
    assert (e["n_questions"], e["total_pages"], e["choices"], e["id_digits"]) == (12, 1, 4, 6)
    assert e["n_students"] == 2 and e["n_conflicts"] == 1 and e["n_unassigned"] == 1 and e["has_key"] is False
    assert e["choice_labels"] == LABELS and e["pdf_url"] and e["preview_url"]
    assert c.get("/api/exams/999").status_code == 404
    assert any(x["id"] == i and x["n_scans"] == 4 for x in c.get("/api/exams").json())
    assert e["subject"] == "07" and e["subject_label"] == "과목 07" and e["title"] == "퀴즈 <b>" and "문항" in e["form_desc"]
    assert c.post("/api/exams", json={"questions": 0, "subject": "07"}).status_code == 422
    assert c.post("/api/exams", json={"questions": 10, "labels": "Z", "subject": "07"}).status_code == 422
    assert c.post("/api/exams", json={"questions": 2, "pages": 3, "subject": "07"}).status_code == 400
    assert c.post("/api/exams", json={"questions": 10, "subject": "07"}).status_code == 200            # a new form: fine
    assert c.post("/api/exams", json={"questions": 10, "subject": "07"}).status_code == 400            # same (form, subject) again
    assert c.post("/api/exams", json={"questions": 10, "subject": "7"}).status_code == 400
    assert c.post("/api/exams", json={"questions": 10}).status_code == 400                             # subject is needed
    # removed options are errors, not silently ignored
    for old in ({"questions_per_page": 2}, {"choice_labels": ["A", "B"]}, {"form_id": 5}):
        assert c.post("/api/exams", json={"questions": 10, "subject": "09", **old}).status_code == 422
    # rename
    assert c.patch(f"/api/exams/{i}", json={"title": "기말"}).json()["title"] == "기말"
    assert c.patch(f"/api/exams/{i}", json={"title": ""}).status_code == 422
    assert c.patch("/api/exams/999", json={"title": "x"}).status_code == 404
    c.patch(f"/api/exams/{i}", json={"title": "퀴즈 <b>"})


def test_preview(env):
    c, i = env["c"], env["id"]
    r = c.get(f"/api/exams/{i}/preview.png")
    assert r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert c.get(f"/api/exams/{i}/preview.png").content == r.content           # cached
    assert os.path.isdir(os.path.join(env["data"], "forms", "preview"))
    assert c.get(f"/api/exams/{i}/preview.png?page=5").status_code == 404
    assert c.get("/api/exams/999/preview.png").status_code == 404


def test_dashboard_and_csv(env):
    c, i = env["c"], env["id"]
    d = c.get(f"/api/exams/{i}/dashboard").json()
    rows = {s["student_id"]: s for s in d["students"]}
    assert set(rows) == {"111111", "222222"}
    assert rows["111111"]["status"] == "complete" and rows["222222"]["status"] == "conflict"
    assert rows["111111"]["score"] is None and rows["111111"]["pages_present"] == [1]
    assert d["exam"]["n_conflicts"] == 1
    assert c.get("/api/exams/999/dashboard").status_code == 404
    r = c.get(f"/api/exams/{i}/export.csv")
    assert r.status_code == 200 and r.content.startswith(b"\xef\xbb\xbf") and "attachment" in r.headers["content-disposition"]
    assert r.headers["content-type"].startswith("text/csv")
    assert c.get("/api/exams/999/export.csv").status_code == 404


def test_images_endpoint(env):
    c, svc = env["c"], env["svc"]
    sid = env["res"][0]["outcome"]["scan_id"]
    for kind in ("original", "warped", "name_crop"):
        r = c.get(f"/api/scans/{sid}/image/{kind}")
        assert r.status_code == 200 and r.headers["content-type"].startswith("image/"), kind
    small = c.get(f"/api/scans/{sid}/image/warped?w=300")
    assert small.status_code == 200 and small.headers["content-type"] == "image/jpeg"
    hl = c.get(f"/api/scans/{sid}/image/warped?w=600&highlight=1,3")
    plain = c.get(f"/api/scans/{sid}/image/warped?w=600")
    assert hl.status_code == 200 and hl.content != plain.content
    assert c.get(f"/api/scans/{sid}/image/warped?highlight=a,b").status_code == 400
    assert c.get(f"/api/scans/{sid}/image/bogus").status_code == 400
    assert c.get("/api/scans/99999/image/original").status_code == 404
    assert c.get(f"/api/scans/{sid}/image/../../etc/passwd").status_code in (404, 400)
    assert c.get("/api/scans/abc/image/original").status_code == 422
    # a path stored in the DB that points outside the scans folder is never served
    secret = os.path.join(env["data"], "secret.txt")
    with open(secret, "w") as f:
        f.write("secret")
    con = sqlite3.connect(svc.db_path)
    con.execute("DROP TRIGGER scans_no_update")
    con.execute("UPDATE scans SET original_path=? WHERE id=?", (secret, sid))
    con.commit()
    con.close()
    r = c.get(f"/api/scans/{sid}/image/original")
    assert r.status_code == 404 and b"secret" not in r.content


def test_conflict_listing_and_resolution(env):
    c, i = env["c"], env["id"]
    d = c.get(f"/api/exams/{i}/conflicts").json()
    assert len(d["conflicts"]) == 1
    cf = d["conflicts"][0]
    assert cf["student_id"] == "222222" and len(cf["scans"]) == 2 and cf["diff"]
    assert sorted(q["question"] for q in cf["diff"][0]["questions"]) == [3, 7]
    assert all("original_path" not in s and "warped_path" not in s for s in cf["scans"])
    assert all(s["has_warped"] and s["has_name_crop"] for s in cf["scans"])
    sub = cf["id"]
    other = next(s["scan_id"] for s in cf["scans"] if not s["adopted"])
    assert c.get("/api/exams/999/conflicts").status_code == 404
    assert c.post("/api/submissions/9999/adopt", json={"scan_id": other}).status_code == 404
    assert c.post(f"/api/submissions/{sub}/adopt", json={"scan_id": 9999}).status_code == 400
    assert c.post(f"/api/submissions/{sub}/adopt", json={}).status_code == 422
    assert c.post(f"/api/submissions/{sub}/reassign", json={"scan_id": other, "student_id": ""}).status_code == 422
    assert c.post(f"/api/submissions/{sub}/reassign", json={"scan_id": 9999, "student_id": "5"}).status_code == 400
    r = c.post(f"/api/submissions/{sub}/adopt", json={"scan_id": other})
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["adopted_scan_id"] == other
    assert c.get(f"/api/exams/{i}/conflicts").json()["conflicts"] == []
    assert c.get(f"/api/exams/{i}/dashboard").json()["exam"]["n_conflicts"] == 0


def test_reassign_conflict_scan(env):
    """'Different students': build a fresh conflict and move one of its scans to another student id."""
    c = env["c"]
    svc = env["svc"]
    ex, L, pdf = flow.make_exam(c, svc, questions=6, choices=4, id_digits=6)
    a = flow.random_answers(L, 5)
    b = {q: (v + 1) % 4 for q, v in a.items()}
    res = flow.upload(c, [flow.sheet_jpeg(L, pdf, 0, "333333", a, 20), flow.sheet_jpeg(L, pdf, 0, "333333", b, 21)])
    assert [r["outcome"]["kind"] for r in res] == ["new", "conflict"]
    cf = c.get(f"/api/exams/{ex['id']}/conflicts").json()["conflicts"][0]
    r = c.post(f"/api/submissions/{cf['id']}/reassign", json={"scan_id": res[1]["outcome"]["scan_id"], "student_id": "444444"})
    assert r.status_code == 200 and r.json()["kind"] == "new" and r.json()["student_id"] == "444444"
    dash = c.get(f"/api/exams/{ex['id']}/dashboard").json()
    st = {s["student_id"]: s["status"] for s in dash["students"]}
    assert set(st) == {"333333", "444444"} and "conflict" not in st.values() and "incomplete" not in st.values()
    assert dash["exam"]["n_conflicts"] == 0


def test_unassigned_flow(env):
    c, i = env["c"], env["id"]
    d = c.get(f"/api/exams/{i}/unassigned").json()
    assert len(d["scans"]) == 1
    s = d["scans"][0]
    assert len(s["id_digits"]) == 6 and any(x["value"] is None for x in s["id_digits"])
    assert "original_path" not in s and s["has_name_crop"] and "111111" in s["all_students"]
    assert c.get("/api/exams/999/unassigned").status_code == 404
    assert c.post("/api/scans/9999/assign", json={"student_id": "1"}).status_code == 404
    # same content as student 111111's sheet -> assigning there confirms (duplicate)
    r = c.post(f"/api/scans/{s['scan_id']}/assign", json={"student_id": "111111"})
    assert r.status_code == 200 and r.json()["kind"] == "duplicate"
    assert c.get(f"/api/exams/{i}/unassigned").json()["scans"] == []
    r = c.post(f"/api/scans/{s['scan_id']}/assign", json={"student_id": None})
    assert r.json()["kind"] == "unassigned"
    assert len(c.get(f"/api/exams/{i}/unassigned").json()["scans"]) == 1
    c.post(f"/api/scans/{s['scan_id']}/assign", json={"student_id": "111111"})


def test_answer_key_and_student_detail(env):
    c, i, a = env["c"], env["id"], env["ans_a"]
    k = c.get(f"/api/exams/{i}/answer-key").json()
    assert k["has_key"] is False and k["key"] == {}
    # invalid keys
    assert c.put(f"/api/exams/{i}/answer-key", json={"key": {"1": [[9]]}}).status_code == 400
    assert c.put(f"/api/exams/{i}/answer-key", json={"key": {"99": [[0]]}}).status_code == 400
    assert c.put(f"/api/exams/{i}/answer-key", json={"key": {"1": [[0]]}, "points": {"2": 1}}).status_code == 400
    assert c.put("/api/exams/999/answer-key", json={"key": {}}).status_code == 404
    key = flow.accepted_key(a)
    key["1"] = [[a[1]], [(a[1] + 1) % 4]]          # two accepted answers
    key["2"] = [sorted([a[2], (a[2] + 1) % 4])]    # must mark exactly this pair -> student 111111 is wrong
    r = c.put(f"/api/exams/{i}/answer-key", json={"key": key, "points": {"1": 2.5}})
    assert r.status_code == 200 and r.json()["has_key"] and r.json()["n_keyed"] == 12 and r.json()["points"] == {"1": 2.5}
    d = c.get(f"/api/exams/{i}/dashboard").json()
    row = next(s for s in d["students"] if s["student_id"] == "111111")
    assert row["score"] == 12.5 and row["max_score"] == 13.5 and row["provisional"] is False
    # student detail
    assert c.get(f"/api/exams/{i}/students/nobody").status_code == 404
    assert c.get("/api/exams/999/students/111111").status_code == 404
    st = c.get(f"/api/exams/{i}/students/111111").json()
    qs = {q["number"]: q for q in st["questions"]}
    assert qs[1]["correct"] is True and qs[1]["earned"] == 2.5 and qs[2]["correct"] is False and qs[2]["accepted"] == [sorted([a[2], (a[2] + 1) % 4])]
    assert st["grade"]["score"] == 12.5 and st["pages"][0]["present"] and len(st["scans"]) >= 1
    assert "original_path" not in st["scans"][0]
    # override: make Q2 the exact pair -> correct
    sub = qs[2]["submission_id"]
    pair = sorted([a[2], (a[2] + 1) % 4])
    assert c.post(f"/api/submissions/{sub}/override", json={"question": 2, "marked": pair}).status_code == 200
    st = c.get(f"/api/exams/{i}/students/111111").json()
    q2 = next(q for q in st["questions"] if q["number"] == 2)
    assert q2["correct"] is True and q2["source"] == "override" and st["grade"]["score"] == 13.5
    assert c.post(f"/api/submissions/{sub}/override", json={"question": 2, "marked": None}).status_code == 200
    q2 = next(q for q in c.get(f"/api/exams/{i}/students/111111").json()["questions"] if q["number"] == 2)
    assert q2["correct"] is False and q2["source"] == "scan"
    # override errors
    assert c.post(f"/api/submissions/{sub}/override", json={"question": 99, "marked": [0]}).status_code == 400
    assert c.post(f"/api/submissions/{sub}/override", json={"question": 2, "marked": [7]}).status_code == 400
    assert c.post("/api/submissions/9999/override", json={"question": 2, "marked": [0]}).status_code == 404
    # csv now has scores
    text = c.get(f"/api/exams/{i}/export.csv").content.decode("utf-8-sig")
    lines = text.strip().split("\n")
    assert lines[0].startswith("student_id,status,score") and any(l.startswith("111111,") for l in lines)


def test_answer_key_from_scan(env):
    c, i = env["c"], env["id"]
    sid = env["res"][0]["outcome"]["scan_id"]
    r = c.post(f"/api/exams/{i}/answer-key/from-scan", json={"scan_ids": [sid]})
    assert r.status_code == 200
    body = r.json()
    for q, ch in env["ans_a"].items():
        assert body["result"]["key"][str(q)] == [ch]
    assert body["exam"]["n_keyed"] == 12
    assert c.post(f"/api/exams/{i}/answer-key/from-scan", json={"scan_ids": [sid], "apply": False}).status_code == 200
    assert c.post(f"/api/exams/{i}/answer-key/from-scan", json={"scan_ids": [99999]}).status_code == 404
    assert c.post(f"/api/exams/{i}/answer-key/from-scan", json={"scan_ids": []}).status_code == 422
    other = c.post("/api/exams", json={"questions": 5, "subject": "07"}).json()
    assert c.post(f"/api/exams/{other['id']}/answer-key/from-scan", json={"scan_ids": [sid]}).status_code == 400
    assert c.get(f"/api/exams/{other['id']}/answer-key").json()["has_key"] is False   # untouched
    # clearing the key
    assert c.put(f"/api/exams/{i}/answer-key", json={"key": {}}).json()["has_key"] is False
    assert next(s for s in c.get(f"/api/exams/{i}/dashboard").json()["students"] if s["student_id"] == "111111")["score"] is None


def test_scan_detail_endpoint(env):
    c = env["c"]
    sid = env["res"][0]["outcome"]["scan_id"]
    d = c.get(f"/api/scans/{sid}/detail").json()
    assert d["ppm"] > 0 and d["page_w"] > 0 and len(d["answers"]) == 12 and len(d["id"]) == 6
    a = d["answers"][0]
    assert a["q"] == 1 and len(a["centers"]) == len(a["scores"]) == 4 and "doubt" in a
    assert all(len(p) == 2 for p in a["centers"])
    assert c.get("/api/scans/99999/detail").status_code == 404


def test_scan_detail_real_photo(tmp_path):
    """A scan made from a real photo: the detail data and a crisp, non-trivial sheet image are served."""
    photo = os.path.join(os.path.dirname(__file__), "..", "realasset", "mixed_ink_pencil_id.jpg")
    if not os.path.exists(photo):
        pytest.skip("no real photo")
    import numpy as np
    import cv2
    app = create_app(str(tmp_path))
    c = TestClient(app)
    with open(photo, "rb") as f:
        r = c.post("/api/upload", files=[("files", ("a.jpg", f.read(), "image/jpeg"))]).json()["results"][0]
    sid = r["outcome"]["scan_id"]
    d = c.get(f"/api/scans/{sid}/detail").json()
    im = c.get(f"/api/scans/{sid}/image/warped")
    assert im.status_code == 200 and im.headers["content-type"].startswith("image/") and len(im.content) > 50_000
    arr = cv2.imdecode(np.frombuffer(im.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert abs(arr.shape[1] / d["page_w"] - d["ppm"]) < 0.5          # canvas pixels / mm matches the centres' unit
    assert len(d["answers"]) == 45 and len(d["id"]) == 5 and len(d["subject"]) == 2
    xs = [p[0] for a in d["answers"] for p in a["centers"]]
    ys = [p[1] for a in d["answers"] for p in a["centers"]]
    assert 0 < min(xs) and max(xs) < d["page_w"] and 0 < min(ys) and max(ys) * d["ppm"] < arr.shape[0]


def test_detail_js_shared_block_in_sync():
    """The detail view exists twice (server static file, inlined in standalone/app.js); keep them identical."""
    root = os.path.join(os.path.dirname(__file__), "..")
    def block(p):
        s = open(os.path.join(root, p), encoding="utf-8").read()
        return s[s.index("/* OMR-DETAIL-SHARED BEGIN"):s.index("/* OMR-DETAIL-SHARED END */")]
    assert block("app/static/detail.js") == block("standalone/app.js")
