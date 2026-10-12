"""FastAPI app: camera WebSocket, upload, and the small JSON API the scan page needs."""
from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from omr import markers as mk
from omr.layout import Template
from omr.recognize import recognize
from omr.review import annotate_warped, cached_preview
from omr.service import OmrService
from omr.stream import StreamSession

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
UPLOAD_MAX_SIDE = 2600


class FormIn(BaseModel):
    """What a sheet looks like.  Everything here is printed into the sheet's header, so the form needs no
    database record: a PDF is rendered from these fields alone.  Unknown fields are an error (the old
    ``questions_per_page`` / ``choice_labels`` / ``form_id`` no longer exist: the split follows from ``pages``)."""
    model_config = ConfigDict(extra="forbid")
    title: str = ""
    questions: int = Field(45, ge=1, le=1023)
    used: Optional[int] = Field(None, ge=1)              # questions actually used (the rest are not printed / read)
    choices: int = Field(5, ge=2, le=10)
    labels: str = Field("1", pattern="^(1|A|a|가)$")      # choice label style: 1 2 3 / A B C / a b c / 가 나 다
    id_digits: int = Field(8, ge=1, le=12)
    subject_digits: int = Field(2, ge=0, le=3)
    pages: Optional[int] = Field(None, ge=1, le=16)
    bubble: float = Field(4.0, ge=4.0, le=5.75)           # smallest bubble diameter, mm (4.0, 4.25, ... 5.75)

    def to_template(self) -> Template:
        return Template(questions=self.questions, choices=self.choices, id_digits=self.id_digits, pages=self.pages,
                        title=self.title, choice_labels=mk.choice_labels(mk.LABEL_STYLES.index(self.labels), self.choices),
                        min_bubble_mm=self.bubble, used=self.used, subject_digits=self.subject_digits)


def form_query(title: str = "", questions: int = 45, used: Optional[int] = None, choices: int = 5, labels: str = "1",
               id_digits: int = 8, subject_digits: int = 2, pages: Optional[int] = None, bubble: float = 4.0) -> FormIn:
    try:
        return FormIn(title=title, questions=questions, used=used, choices=choices, labels=labels, id_digits=id_digits,
                      subject_digits=subject_digits, pages=pages, bubble=bubble)
    except ValueError as ex:
        raise HTTPException(400, str(ex))


class ExamIn(FormIn):
    """Pre-create the exam of one subject of a form (to enter its answer key before scanning)."""
    exam_title: str = ""
    subject: Optional[str] = None


class ExamPatch(BaseModel):
    title: str = Field(min_length=1)


class PrintRunIn(FormIn):
    copies: int = Field(1, ge=1, le=500)
    start: Optional[int] = Field(None, ge=1, le=4095)


class RouteIn(BaseModel):
    subject: Optional[str] = None        # move the scan to this subject's exam
    student_id: Optional[str] = None


class AdoptIn(BaseModel):
    scan_id: int


class ReassignIn(BaseModel):
    scan_id: int
    student_id: str = Field(min_length=1)


class AssignIn(BaseModel):
    student_id: Optional[str] = None     # null / empty = back to the unassigned queue


class OverrideIn(BaseModel):
    question: int
    marked: Optional[List[int]] = None   # null = remove the override, [] = blank
    note: Optional[str] = None


class KeyIn(BaseModel):
    # {question: [accepted, ...]}; each accepted is a choice index or a list of indices (marked exactly)
    key: dict = Field(default_factory=dict)
    points: dict = Field(default_factory=dict)


class KeyFromScanIn(BaseModel):
    scan_ids: List[int] = Field(min_length=1)
    apply: bool = True


def clean(o):
    """Drop server file paths (``*_path`` keys) from anything sent to the browser; images go through
    /api/scans/{id}/image/{kind} which looks the file up by scan id."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if not (isinstance(k, str) and k.endswith("_path"))}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


MIN_FRAME_SIDE = 64


class FrameDump:
    """Debug aid: keep the last ``keep`` received frames (raw bytes as sent by the browser) in ``folder``,
    named with the decoded size, e.g. ``000012_1280x720.jpg`` (``..._undecodable.bin`` if not an image)."""

    def __init__(self, folder: str, keep: int):
        self.folder, self.keep, self.n = folder, max(1, int(keep)), 0
        os.makedirs(folder, exist_ok=True)

    def save(self, data: bytes, img) -> None:
        self.n += 1
        tag = f"{img.shape[1]}x{img.shape[0]}" if img is not None else "undecodable"
        name = f"{self.n:06d}_{tag}.{'jpg' if img is not None else 'bin'}"
        with open(os.path.join(self.folder, name), "wb") as f:
            f.write(data)
        old = os.path.join(self.folder, "")
        for fn in os.listdir(self.folder):
            if fn[:6].isdigit() and int(fn[:6]) <= self.n - self.keep:
                try:
                    os.remove(os.path.join(self.folder, fn))
                except OSError:
                    pass


def decode_image(data: bytes) -> Optional[np.ndarray]:
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def summarize(result) -> dict:
    return {"form_key": result.form_key, "page": result.page, "student_id": result.student_id,
            "subject": result.subject,
            "serial": result.serial,
            "n_questions": len(result.answers),
            "n_blank": sum(1 for a in result.answers if a.state == "blank"),
            "n_multi": sum(1 for a in result.answers if a.state == "multi"),
            "n_doubtful": sum(1 for a in result.answers if a.ambiguous),
            "flags": list(result.flags), "needs_review": result.needs_review,
            "sharpness": (result.quality or {}).get("sharpness")}


def create_app(data_dir: str = "data", workers: int = 2) -> FastAPI:
    app = FastAPI(title="StreamOMR")
    svc = OmrService(data_dir)
    pool = ThreadPoolExecutor(max_workers=workers)
    app.state.service = svc
    app.state.data_dir = data_dir

    def exam_or_404(exam_id: int) -> dict:
        try:
            return svc.get_exam(exam_id)
        except KeyError:
            raise HTTPException(404, f"no exam {exam_id}")

    def public_exam(e: dict) -> dict:
        d = svc.exam_overview(e["id"])
        d["pdf_url"] = f"/api/exams/{e['id']}/pdf"
        d["preview_url"] = f"/api/exams/{e['id']}/preview.png"
        return d

    def preview_png(path: str, page: int, dpi: int) -> Response:
        try:
            png = cached_preview(path, page, os.path.join(app.state.data_dir, "forms", "preview"),
                                 dpi=max(30, min(int(dpi), 200)))
        except ValueError as ex:
            raise HTTPException(404, str(ex))
        return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    def guarded(fn, *a, **kw):
        """Run a service call, mapping its exceptions to HTTP errors."""
        try:
            return fn(*a, **kw)
        except KeyError as ex:
            raise HTTPException(404, str(ex.args[0]) if ex.args else "not found")
        except (ValueError, TypeError) as ex:
            raise HTTPException(400, str(ex))

    # ------------------------------------------------------------------ pages
    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/static/index.html")

    # ------------------------------------------------------------------ forms (printing needs no database)
    @app.post("/api/forms/info")
    def form_info(body: FormIn):
        """Check a form definition: its identity, page split and a one-line description."""
        from omr.layout import describe
        r = guarded(svc.form_pdf, body.to_template())
        L = r["layout"]
        return {"form_key": L.form_key, "description": describe(L), "total_pages": L.total_pages,
                "questions_per_page": list(L.questions_per_page)}

    @app.get("/api/forms/pdf")
    def form_pdf(form: FormIn = Depends(form_query)):
        """Plain master PDF (no serial, for photocopying).  Numbered copies: POST print-runs, then GET its pdf."""
        r = guarded(svc.form_pdf, form.to_template())
        return FileResponse(r["path"], media_type="application/pdf", filename=f"form_{r['form_key']}.pdf")

    @app.get("/api/forms/preview.png")
    def form_preview(page: int = 0, dpi: int = 80, form: FormIn = Depends(form_query)):
        r = guarded(svc.form_pdf, form.to_template())
        return preview_png(r["path"], page, dpi)

    @app.post("/api/forms/print-runs")
    def create_print_run(body: PrintRunIn):
        """Reserve serials for N numbered copies of a form (the only call that issues numbers)."""
        r = guarded(svc.create_print_run, body.to_template(), body.copies, body.start)
        r["pdf_url"] = f"/api/forms/print-runs/{r['id']}/pdf"
        return r

    @app.get("/api/forms/print-runs/{run_id}/pdf")
    def print_run_pdf(run_id: int):
        r = guarded(svc.run_pdf, run_id)
        return FileResponse(r["path"], media_type="application/pdf",
                            filename=f"form_{r['form_key']}_no{r['serials'][0]:04d}-{r['serials'][-1]:04d}.pdf")

    # ------------------------------------------------------------------ exams (= one subject of a form)
    @app.get("/api/exams")
    def list_exams():
        return [public_exam(e) for e in svc.list_exams()]

    @app.get("/api/exams/{exam_id}")
    def get_exam(exam_id: int):
        exam_or_404(exam_id)
        return public_exam({"id": exam_id})

    @app.post("/api/exams")
    def create_exam(body: ExamIn):
        e = guarded(svc.create_exam, body.exam_title or body.title or "", body.to_template(), body.subject)
        return public_exam(e)

    @app.patch("/api/exams/{exam_id}")
    def rename_exam(exam_id: int, body: ExamPatch):
        exam_or_404(exam_id)
        guarded(svc.rename_exam, exam_id, body.title)
        return public_exam({"id": exam_id})

    def exam_form_pdf(exam_id: int) -> str:
        """The exam's form, re-rendered from its layout (nothing is stored per exam)."""
        exam_or_404(exam_id)
        lay = svc.exam_layout(exam_id)
        if lay.version == 0:
            raise HTTPException(404, "legacy sheet: no PDF can be generated")
        return svc.form_pdf(lay.template)["path"]

    @app.get("/api/exams/{exam_id}/pdf")
    def exam_pdf(exam_id: int):
        return FileResponse(exam_form_pdf(exam_id), media_type="application/pdf", filename=f"exam_{exam_id}.pdf")

    @app.get("/api/exams/{exam_id}/preview.png")
    def exam_preview(exam_id: int, page: int = 0, dpi: int = 80):
        return preview_png(exam_form_pdf(exam_id), page, dpi)

    @app.get("/api/exams/{exam_id}/scans")
    def recent_scans(exam_id: int, limit: int = 30):
        exam_or_404(exam_id)
        return svc.recent_scans(exam_id, limit)

    # ------------------------------------------------------------------ dashboard / csv
    @app.get("/api/exams/{exam_id}/dashboard")
    def dashboard(exam_id: int):
        exam_or_404(exam_id)
        d = svc.dashboard(exam_id)
        d["exam"]["pdf_url"] = f"/api/exams/{exam_id}/pdf"
        return d

    @app.get("/api/exams/{exam_id}/export.csv")
    def export_csv(exam_id: int):
        exam_or_404(exam_id)
        body = "\ufeff" + svc.export_csv(exam_id)      # BOM so Excel reads UTF-8 (Korean) correctly
        return Response(body.encode("utf-8"), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="exam_{exam_id}.csv"'})

    # ------------------------------------------------------------------ review queues
    @app.get("/api/exams/{exam_id}/conflicts")
    def conflicts(exam_id: int):
        exam_or_404(exam_id)
        return {"exam": svc.exam_overview(exam_id), "conflicts": clean(svc.list_conflicts(exam_id))}

    @app.get("/api/exams/{exam_id}/unassigned")
    def unassigned(exam_id: int):
        exam_or_404(exam_id)
        return {"exam": svc.exam_overview(exam_id), "scans": clean(svc.unassigned_detail(exam_id))}

    @app.get("/api/exams/{exam_id}/students/{student_id}")
    def student(exam_id: int, student_id: str):
        exam_or_404(exam_id)
        return clean(guarded(svc.student_detail, exam_id, student_id))

    # ------------------------------------------------------------------ scan images (looked up by id, never by path)
    @app.get("/api/scans/{scan_id}/image/{kind}")
    def scan_image(scan_id: int, kind: str, w: Optional[int] = None, highlight: Optional[str] = None):
        path = guarded(svc.scan_image_path, scan_id, kind)
        if w is None and not highlight:
            return FileResponse(path, headers={"Cache-Control": "max-age=3600"})
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise HTTPException(404, "image unreadable")
        if highlight and kind == "warped":
            try:
                qs = [int(x) for x in highlight.split(",") if x.strip()][:400]
            except ValueError:
                raise HTTPException(400, "highlight must be comma separated question numbers")
            layout, page = svc.scan_layout(scan_id)
            img = annotate_warped(img, layout, page, qs)
        if w is not None:
            w = max(40, min(int(w), 2600))
            if img.shape[1] > w:
                img = cv2.resize(img, (w, max(1, round(img.shape[0] * w / img.shape[1]))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    # ------------------------------------------------------------------ manual actions
    @app.post("/api/submissions/{submission_id}/adopt")
    def adopt(submission_id: int, body: AdoptIn):
        return clean(guarded(svc.resolve_adopt, submission_id, body.scan_id))

    @app.post("/api/submissions/{submission_id}/reassign")
    def reassign(submission_id: int, body: ReassignIn):
        return guarded(svc.resolve_reassign, submission_id, body.scan_id, body.student_id).to_dict()

    @app.post("/api/submissions/{submission_id}/override")
    def override(submission_id: int, body: OverrideIn):
        guarded(svc.override_answer, submission_id, body.question, body.marked, body.note)
        return {"ok": True}

    @app.post("/api/scans/{scan_id}/route")
    def route(scan_id: int, body: RouteIn):
        """"This scan belongs to subject XX (and student YYYY)": moves it to that subject's exam."""
        return guarded(svc.route_scan, scan_id, body.subject, body.student_id).to_dict()

    @app.get("/api/scans/recent")
    def recent_all(limit: int = 30):
        return svc.recent_scans(None, limit)

    @app.post("/api/scans/{scan_id}/assign")
    def assign(scan_id: int, body: AssignIn):
        return guarded(svc.assign_scan, scan_id, body.student_id).to_dict()

    # ------------------------------------------------------------------ answer key
    def key_view(exam_id: int) -> dict:
        ov = svc.exam_overview(exam_id)
        e = svc.get_exam(exam_id)
        ov["key"] = {str(q): v for q, v in e["answer_key"].items()}
        ov["points"] = {str(q): v for q, v in e["points"].items()}
        return ov

    @app.get("/api/exams/{exam_id}/answer-key")
    def get_key(exam_id: int):
        exam_or_404(exam_id)
        return key_view(exam_id)

    @app.put("/api/exams/{exam_id}/answer-key")
    def put_key(exam_id: int, body: KeyIn):
        exam_or_404(exam_id)
        guarded(svc.set_answer_key, exam_id, body.key, body.points)
        return key_view(exam_id)

    @app.post("/api/exams/{exam_id}/answer-key/from-scan")
    def key_from_scan(exam_id: int, body: KeyFromScanIn):
        exam_or_404(exam_id)
        r = guarded(svc.answer_key_from_scan, body.scan_ids, body.apply, exam_id)
        return {"result": {"key": {str(q): v for q, v in r["key"].items()}, "skipped": r["skipped"],
                           "ambiguous": r["ambiguous"]}, "exam": key_view(exam_id)}

    # ------------------------------------------------------------------ upload
    def _process_upload(name: str, data: bytes) -> dict:
        img = decode_image(data)
        if img is None:
            return {"filename": name, "status": "unreadable_image", "outcome": None}
        small = img
        if max(img.shape[:2]) > UPLOAD_MAX_SIDE:
            s = UPLOAD_MAX_SIDE / max(img.shape[:2])
            small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        res = recognize(small, svc.resolve_layout)
        out = {"filename": name, "status": res.status, "summary": summarize(res), "outcome": None}
        if res.status == "ok":
            out["outcome"] = svc.ingest(res, images={"original": img}, source="upload").to_dict()
        return out

    @app.post("/api/upload")
    async def upload(files: List[UploadFile] = File(...)):
        loop = asyncio.get_running_loop()
        out = []
        for f in files:
            data = await f.read()
            out.append(await loop.run_in_executor(pool, _process_upload, f.filename or "", data))
        return {"results": out}

    # ------------------------------------------------------------------ live scanning
    def _frame_step(session: StreamSession, data: bytes, dump: Optional[FrameDump] = None) -> dict:
        t0 = time.perf_counter()
        img = decode_image(data)
        if dump is not None:
            dump.save(data, img)
        if img is None:
            return {"state": session.state, "hint": "Bad frame", "overlay": {}, "error": "undecodable frame"}
        if min(img.shape[:2]) < MIN_FRAME_SIDE:
            # e.g. a browser that sent a 0x0 / 1x1 canvas before the camera metadata was ready
            w, h = int(img.shape[1]), int(img.shape[0])
            return {"state": session.state, "hint": f"Camera frame too small ({w}x{h})", "overlay": {},
                    "error": f"frame too small: {w}x{h}", "w": w, "h": h}
        res = recognize(img, svc.resolve_layout)
        ev = session.feed(res, time.monotonic(), payload=img)
        ev["status"] = res.status
        ev["w"], ev["h"] = int(img.shape[1]), int(img.shape[0])
        commit = ev.pop("commit", None)
        if commit is not None:
            r = commit["result"]
            outcome = svc.ingest(r, images={"original": commit["payload"]}, source="stream")
            ev["commit"] = {"outcome": outcome.to_dict(), "summary": summarize(r), "frames": commit["frames"]}
        missed = session.missed()
        if missed:
            ev["missed"] = missed
            ev["missed_serials"] = sorted({m["serial"] for m in missed if m["serial"]})
        ev["ms"] = {"server": round((time.perf_counter() - t0) * 1000, 1), "recognize": round(res.timing_ms.get("total", 0), 1)}
        return ev

    @app.websocket("/ws/scan")
    async def ws_scan(ws: WebSocket, exam_id: Optional[int] = None, subject: Optional[str] = None,
                      debug_frames: Optional[int] = None):
        """No exam is needed: any sheet is read and routed by its own header.  Optional filters: ``exam_id``
        (its form and subject) and/or ``subject`` (sheets with another readable subject are ignored)."""
        forms, subjects = None, None
        if exam_id is not None:
            try:
                e = svc.get_exam(exam_id)
            except KeyError:
                await ws.close(code=4404)
                return
            forms = {e["form_key"]}
            subjects = {e["subject"]} if e["subject"] else None
        if subject:
            subjects = {subject.strip()}
        await ws.accept()
        n_dump = debug_frames if debug_frames else int(os.environ.get("STREAMOMR_DEBUG_FRAMES", "0") or 0)
        dump = FrameDump(os.path.join(app.state.data_dir, "debug_frames"), n_dump) if n_dump > 0 else None
        session = StreamSession(allowed_forms=forms, allowed_subjects=subjects, resolve_layout=svc.resolve_layout)
        loop = asyncio.get_running_loop()
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data is None:
                    if msg.get("text") == "reset":
                        session.reset()
                        session.reset_tracking()
                    continue
                ev = await loop.run_in_executor(pool, _frame_step, session, data, dump)
                await ws.send_json(ev)
        except WebSocketDisconnect:
            pass

    @app.middleware("http")
    async def _no_stale_static(request, call_next):
        # Browsers (Safari especially) otherwise keep running an old scan.js after an update.
        resp = await call_next(request)
        if request.url.path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def app_from_env() -> FastAPI:
    return create_app(os.environ.get("STREAMOMR_DATA", "data"))
