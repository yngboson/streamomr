import os
import sys

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(__file__))
import synth  # noqa: E402
import flow  # noqa: E402

from app.main import create_app  # noqa: E402


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    app = create_app(str(tmp_path_factory.mktemp("data")))
    client = TestClient(app)
    svc = app.state.service
    exam, L, pdf = flow.make_exam(client, svc, exam_title="Quiz", questions=30, choices=4, id_digits=6)

    def jpeg(seed, mode="sheet", subject="07"):
        if mode == "sheet":
            sid = f"{seed:06d}"
            return flow.sheet_jpeg(L, pdf, 0, sid, flow.random_answers(L, seed), seed, subject=subject), {"student_id": sid}
        frame, truth = synth.make_frame(L, pdf, synth.CONDS["clean"], np.random.default_rng(seed), mode=mode)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes(), truth

    return {"app": app, "client": client, "exam": exam, "svc": svc, "jpeg": jpeg}


def scan_sequence(ws, frames):
    events = []
    for b in frames:
        ws.send_bytes(b)
        events.append(ws.receive_json())
    return events


def test_pdf_and_exam_list(env):
    c, e = env["client"], env["exam"]
    r = c.get(e["pdf_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content[:4] == b"%PDF"
    assert [x["id"] for x in c.get("/api/exams").json()] == [e["id"]]
    assert c.get("/api/exams/999/pdf").status_code == 404
    assert c.post("/api/exams", json={"questions": 3}).status_code == 400            # the subject code is needed
    assert c.post("/api/exams", json={"questions": 0, "subject": "01"}).status_code == 422
    assert c.get("/").history or c.get("/", follow_redirects=False).status_code in (302, 307)
    assert "StreamOMR" in c.get("/static/index.html").text
    assert "ws/scan" in c.get("/static/scan.js").text


def test_forms_are_stateless_and_numbered_runs_are_per_form(env):
    c, svc = env["client"], env["svc"]
    q = "questions=100&used=80&pages=2&choices=5&labels=A&id_digits=8&subject_digits=2&title=T"
    n_exams = len(c.get("/api/exams").json())
    r = c.get(f"/api/forms/pdf?{q}")
    assert r.status_code == 200 and r.content[:4] == b"%PDF" and "Q100U80C5D8S2P2L1B0V2" in r.headers["content-disposition"]
    assert c.get(f"/api/forms/pdf?{q}").content == r.content
    assert len(c.get("/api/exams").json()) == n_exams                         # printing created nothing in the database
    info = c.post("/api/forms/info", json={"questions": 100, "used": 80, "pages": 2, "labels": "A"}).json()
    assert info["form_key"] == "Q100U80C5D8S2P2L1B0V2" and info["total_pages"] == 2 and info["questions_per_page"] == [50, 30]
    assert "80문항" in info["description"]
    pv = c.get(f"/api/forms/preview.png?{q}&page=1")
    assert pv.status_code == 200 and pv.content[:4] == b"\x89PNG" and c.get(f"/api/forms/preview.png?{q}&page=2").status_code == 404
    assert c.get("/api/forms/pdf?questions=0").status_code in (400, 422)
    assert c.get("/api/forms/pdf?questions=100&used=200").status_code == 400
    # numbered runs are recorded per FORM, continue after the previous one and never overlap
    body = {"questions": 100, "used": 80, "pages": 2, "labels": "A", "copies": 3}
    r1 = c.post("/api/forms/print-runs", json=body).json()
    r2 = c.post("/api/forms/print-runs", json={**body, "copies": 2}).json()
    assert r1["serials"] == [1, 2, 3] and r2["serials"] == [4, 5] and r1["form_key"] == r2["form_key"]
    assert c.post("/api/forms/print-runs", json={**body, "start": 2}).status_code == 400
    other = c.post("/api/forms/print-runs", json={"questions": 40, "copies": 2}).json()
    assert other["serials"] == [1, 2]                                         # another form: its own numbering
    pdf = c.get(r1["pdf_url"])
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    import pymupdf
    assert len(pymupdf.open(stream=pdf.content, filetype="pdf")) == 3 * 2     # 3 sets x 2 pages
    assert c.get("/api/forms/print-runs/999/pdf").status_code == 404


def test_ws_commit_then_duplicate(env):
    c, e, svc = env["client"], env["exam"], env["svc"]
    a, _ = env["jpeg"](11)
    empty, _ = env["jpeg"](12, "none")
    before = svc.scan_count(e["id"])
    with c.websocket_connect(f"/ws/scan?exam_id={e['id']}") as ws:
        evs = scan_sequence(ws, [empty, a, a, a, a, a])
        commits = [ev["commit"] for ev in evs if "commit" in ev]
        assert len(commits) == 1
        assert commits[0]["outcome"]["kind"] == "new"
        assert evs[0]["state"] == "idle" and evs[-1]["state"] == "cooldown"
        assert evs[3]["overlay"]["bubbles"] and evs[3]["w"] == 1920
        sid = commits[0]["summary"]["student_id"]
        assert sid and commits[0]["outcome"]["student_id"] == sid
        # sheet leaves, then comes back: committed again, the DB says duplicate
        evs2 = scan_sequence(ws, [empty] * 5 + [a] * 4)
        c2 = [ev["commit"] for ev in evs2 if "commit" in ev]
        assert len(c2) == 1 and c2[0]["outcome"]["kind"] == "duplicate"
    assert svc.scan_count(e["id"]) == before + 2
    recent = c.get(f"/api/exams/{e['id']}/scans").json()
    assert recent[0]["student_id"] == sid and recent[0]["kind"] == "duplicate" and recent[1]["kind"] == "new"


def test_ws_single_frame_no_commit_and_bad_frame(env):
    c, e, svc = env["client"], env["exam"], env["svc"]
    b, _ = env["jpeg"](21)
    empty, _ = env["jpeg"](22, "none")
    n = svc.scan_count(e["id"])
    with c.websocket_connect("/ws/scan") as ws:
        evs = scan_sequence(ws, [empty, b, empty, empty])
        assert not any("commit" in ev for ev in evs)
        ws.send_bytes(b"not a jpeg")
        assert ws.receive_json()["error"]
    assert svc.scan_count(e["id"]) == n


def test_ws_other_exam_restriction(env):
    c, e = env["client"], env["exam"]
    other = c.post("/api/exams", json={"questions": 10, "subject": "07"}).json()
    b, _ = env["jpeg"](31)
    with c.websocket_connect(f"/ws/scan?exam_id={other['id']}") as ws:
        evs = scan_sequence(ws, [b] * 4)
    assert not any("commit" in ev for ev in evs)
    assert "another form" in evs[-1]["hint"]
    # same form, another subject: ignored when the scanner is bound to subject 08
    with c.websocket_connect(f"/ws/scan?exam_id={e['id']}&subject=08") as ws:
        evs = scan_sequence(ws, [b] * 4)
    assert not any("commit" in ev for ev in evs) and "another subject" in evs[-1]["hint"]
    with pytest.raises(Exception):
        with c.websocket_connect("/ws/scan?exam_id=9999") as ws:
            ws.receive_json()


def test_scan_page_needs_no_exam_and_creates_it(tmp_path):
    """A fresh data dir with NO exam: scanning a sheet creates the exam of its subject."""
    app = create_app(str(tmp_path))
    c = TestClient(app)
    svc = app.state.service
    form = svc.form_pdf(questions=30, choices=4, id_digits=6)
    L, pdf = form["layout"], form["path"]
    assert c.get("/api/exams").json() == []
    a = flow.sheet_jpeg(L, pdf, 0, "123456", flow.random_answers(L, 1), 1, subject="42")
    empty, _ = synth.make_frame(L, pdf, synth.CONDS["clean"], np.random.default_rng(2), mode="none")
    ok, buf = cv2.imencode(".jpg", empty)
    with c.websocket_connect("/ws/scan") as ws:
        evs = scan_sequence(ws, [buf.tobytes(), a, a, a, a])
    commit = next(ev["commit"] for ev in evs if "commit" in ev)
    assert commit["outcome"]["kind"] == "new" and commit["outcome"]["subject"] == "42"
    assert commit["summary"]["subject"] == "42" and commit["summary"]["student_id"] == "123456"
    exams = c.get("/api/exams").json()
    assert [(e["subject"], e["title"], e["n_scans"]) for e in exams] == [("42", "과목 42", 1)]
    assert c.get("/api/scans/recent").json()[0]["subject"] == "42"


def test_upload(env):
    c, e = env["client"], env["exam"]
    b, _ = env["jpeg"](41)
    r = c.post("/api/upload", files=[("files", ("a.jpg", b, "image/jpeg")), ("files", ("bad.jpg", b"xx", "image/jpeg"))])
    assert r.status_code == 200
    res = r.json()["results"]
    assert res[0]["status"] == "ok" and res[0]["outcome"]["kind"] == "new" and res[0]["outcome"]["exam_id"] == e["id"]
    assert res[1]["status"] == "unreadable_image"
    r2 = c.post("/api/upload", files=[("files", ("a.jpg", b, "image/jpeg"))]).json()["results"][0]
    assert r2["outcome"]["kind"] == "duplicate"
    scans = c.get(f"/api/exams/{e['id']}/scans").json()
    assert scans[0]["source"] == "upload"


def test_ws_reports_received_frame_size_and_dumps_frames(env, tmp_path):
    c = env["client"]
    b, _ = env["jpeg"](41)
    h, w = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR).shape[:2]
    with c.websocket_connect("/ws/scan?debug_frames=2") as ws:
        for _ in range(3):
            ws.send_bytes(b)
            ev = ws.receive_json()
            assert (ev["w"], ev["h"]) == (w, h)
            assert ev["state"] != "idle" or ev["status"] != "no_sheet"
    import glob
    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(c.app.state.data_dir, "debug_frames", "*")))
    assert len(files) == 2 and all(f.endswith(f"_{w}x{h}.jpg") for f in files)


def test_ws_rejects_tiny_frame_with_visible_error(env):
    c = env["client"]
    tiny = cv2.imencode(".jpg", np.zeros((1, 1, 3), np.uint8))[1].tobytes()
    with c.websocket_connect("/ws/scan") as ws:
        ws.send_bytes(tiny)
        ev = ws.receive_json()
        assert "too small" in ev["error"] and (ev["w"], ev["h"]) == (1, 1)
