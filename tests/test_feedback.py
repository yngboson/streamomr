"""Commit feedback severity (red / amber / green border + sound) and the detail view's correction round trip."""
import os

import cv2
import pytest

from omr.identity import severity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIXED = os.path.join(ROOT, "realasset", "mixed_ink_pencil_id.jpg")


def _r(flags=(), sid="1234", answers=(), review=False):
    return {"flags": list(flags), "student_id": sid, "id_digits": [{}] * 4, "needs_review": review,
            "answers": [dict(a) for a in answers]}


def test_severity_levels():
    assert severity(_r()) == "ok"
    assert severity(_r(flags=["corner_hidden", "serial_unreadable"])) == "ok"        # informational only
    assert severity(_r(answers=[{"state": "single", "ambiguous": True}])) == "minor"
    assert severity(_r(answers=[{"state": "multi", "ambiguous": False}])) == "minor"
    assert severity(_r(flags=["guessed_answers"], review=True)) == "minor"
    for f in ("id_unreadable", "id_guessed", "subject_unreadable", "subject_guessed"):
        assert severity(_r(flags=[f])) == "severe"
    assert severity(_r(sid=None)) == "severe"                                         # ID area present, no ID
    assert severity({"flags": [], "student_id": None, "id_digits": [], "answers": []}) == "ok"   # form without ID


@pytest.mark.skipif(not os.path.exists(MIXED), reason="real photo not present")
def test_detail_correction_round_trip(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import create_app
    c = TestClient(create_app(str(tmp_path)))
    with open(MIXED, "rb") as f:
        out = c.post("/api/upload", files={"files": ("m.jpg", f, "image/jpeg")}).json()
    o = (out if isinstance(out, list) else out["results"])[0]
    assert o["summary"]["severity"] in ("ok", "minor")
    sid = o["outcome"]["scan_id"]
    d = c.get(f"/api/scans/{sid}/detail").json()
    assert d["submission_id"] and d["scan_id"] == sid
    q = d["answers"][0]["q"]
    assert c.post(f"/api/submissions/{d['submission_id']}/override", json={"question": q, "marked": [2]}).json()["ok"]
    a = next(x for x in c.get(f"/api/scans/{sid}/detail").json()["answers"] if x["q"] == q)
    assert a["marked"] == [2] and a["fixed"] and not a["doubt"]
