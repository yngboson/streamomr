import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import synth  # noqa: E402
import replay_video  # noqa: E402

from omr.service import OmrService  # noqa: E402


def make_video(tmp_path):
    svc = OmrService(str(tmp_path / "data"))
    form = svc.form_pdf(questions=30, choices=4, id_digits=6)
    L, pdf = form["layout"], form["path"]
    cond = synth.CONDS["clean"]
    sheets, seed = [], 1
    while len(sheets) < 2:
        f, tr = synth.make_frame(L, pdf, cond, np.random.default_rng(seed), page=0,
                                 fixed={"student_id": f"{seed:06d}", "answers": {q: q % 4 for q in range(1, 31)}, "subject": "0%d" % seed})
        seed += 1
        sheets.append((f, tr))
    exam = {"id": None}
    empty, _ = synth.make_frame(L, pdf, cond, np.random.default_rng(99), mode="none")
    seq = [empty] * 4 + [sheets[0][0]] * 8 + [empty] * 7 + [sheets[1][0]] * 8 + [empty] * 4
    path = str(tmp_path / "v.mp4")
    w = replay_video.open_writer(path, 8, (1920, 1080))
    for f in seq:
        w.write(f)
    w.release()
    return svc, exam, path, [s[1]["student_id"] for s in sheets]


def test_replay_dry_run_and_real(tmp_path):
    svc, exam, path, ids = make_video(tmp_path)
    ann = str(tmp_path / "ann.mp4")
    lines = []
    s = replay_video.main([path, "--data", str(tmp_path / "data"), "--dry-run", "--fps", "8", "--annotate", ann])
    assert len(s["commits"]) == 2 and s["frames"] >= 30
    assert [c["student_id"] for c in s["commits"]] == ids
    assert all(c["outcome"] == "dry-run" for c in s["commits"])
    assert svc.scan_count() == 0
    assert os.path.getsize(ann) > 1000
    s2 = replay_video.main([path, "--data", str(tmp_path / "data")])
    assert [c["outcome"] for c in s2["commits"]] == ["new", "new"]
    assert svc.scan_count() == 2
    assert [c["subject"] for c in s2["commits"]] == ["01", "02"]
    # the exams were created by the scans; --exam-id then restricts a replay to one subject
    ex = {e["subject"]: e["id"] for e in svc.list_exams()}
    assert set(ex) == {"01", "02"}
    s3 = replay_video.main([path, "--data", str(tmp_path / "data"), "--exam-id", str(ex["02"]), "--dry-run"])
    assert [c["student_id"] for c in s3["commits"]] == [ids[1]]


def test_replay_folder_of_photos(tmp_path):
    svc, exam, path, ids = make_video(tmp_path)
    cap = cv2.VideoCapture(path)
    d = tmp_path / "photos"
    d.mkdir()
    for i in range(30):
        ok, f = cap.read()
        if not ok:
            break
        cv2.imwrite(str(d / f"{i:03d}.jpg"), f)
    s = replay_video.main([str(d), "--data", str(tmp_path / "data"), "--dry-run"])
    assert len(s["commits"]) == 2


def test_replay_reports_sheet_without_commit(tmp_path):
    svc, exam, path, ids = make_video(tmp_path)
    # a video where a sheet is in view only 1 frame at a time repeatedly (never stable) -> flagged
    cap = cv2.VideoCapture(path)
    frames = [cap.read()[1] for _ in range(30)]
    sheet, empty = frames[6], frames[0]
    seq = []
    for _ in range(14):
        seq += [sheet, empty]
    w = replay_video.open_writer(str(tmp_path / "u.mp4"), 8, (1920, 1080))
    for f in seq:
        w.write(f)
    w.release()
    s = replay_video.main([str(tmp_path / "u.mp4"), "--data", str(tmp_path / "data"), "--dry-run"])
    assert s["commits"] == [] and len(s["missed"]) == 1 and s["missed"][0]["to"] - s["missed"][0]["from"] > 1.5
    assert s["missed"][0]["frames"] >= 14
    s0 = replay_video.main([path, "--data", str(tmp_path / "data"), "--dry-run"])
    assert s0["missed"] == []
