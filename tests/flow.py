"""Helpers shared by the API / end-to-end tests: synthetic sheets with known student ID + answers."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import synth  # noqa: E402


def random_answers(layout, seed):
    """{question: choice} for every question of the exam (all single, clear marks)."""
    rng = np.random.default_rng(seed)
    nq, nc = sum(layout.questions_per_page), layout.template.choices
    return {q: int(rng.integers(nc)) for q in range(1, nq + 1)}


def sheet_jpeg(layout, pdf, page, student_id, answers, seed, serial=None, subject="07"):
    """JPEG bytes of one synthetic camera frame of ``page`` carrying exactly these marks
    (``subject``: the marked subject code; None = left blank; ignored for forms without subject code)."""
    pg = layout.pages[page]
    page_answers = {q: a for q, a in answers.items() if pg.first_question <= q < pg.first_question + pg.n_questions}
    frame, truth = synth.make_frame(layout, pdf, synth.CONDS["clean"], np.random.default_rng(seed), page=page,
                                    fixed={"student_id": student_id, "answers": page_answers, "subject": subject}, serial=serial)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes()


def upload(client, jpegs):
    files = [("files", (f"s{i}.jpg", b, "image/jpeg")) for i, b in enumerate(jpegs)]
    r = client.post("/api/upload", files=files)
    assert r.status_code == 200, r.text
    return r.json()["results"]


def accepted_key(answers):
    """API answer-key payload: each question accepts exactly its single choice."""
    return {str(q): [[a]] for q, a in answers.items()}


def make_exam(client, svc, subject="07", **form):
    """Pre-create an exam through the API (answer keys before scanning).  Returns (exam json, layout, pdf path);
    the PDF is rendered from the layout's template exactly like a user's download."""
    r = client.post("/api/exams", json={**form, "subject": subject})
    assert r.status_code == 200, r.text
    e = r.json()
    L = svc.exam_layout(e["id"])
    return e, L, svc.form_pdf(L.template)["path"]
