"""Domain operations for exams, scans, the duplicate/conflict policy and grading.

The sheet describes itself (its header says questions / choices / ID digits / subject digits / pages), so scanners
need no database to read it.  An *exam* here is a (form_key, subject) pair: one printed form used for one subject
code.  Exams are created automatically by the first scan of a pair (title "과목 07", editable) or by hand.  A scan
whose subject code could not be read lands in the form's holding exam (subject NULL) and waits for a human to say
which subject it belongs to (``route_scan``); a scan is never moved by mutation: the move is a copy in the target
exam plus an append-only ``scan_moves`` row.

Policy summary (see ``OmrService.ingest``):
  * every accepted scan is stored first (append-only), then the policy runs;
  * a scan's student is derived (latest manual assignment, else the ID read from the sheet);
  * a (exam, student, page) *submission* adopts one scan; identical content seen again only
    raises ``confirm_count``; different content opens a ``conflict`` that a human resolves.

Answer keys: ``key[q]`` is a list of accepted answers; each accepted answer is either a single
choice index (``0``) or a list of indices that must be marked *exactly* (``[0, 2]``).  So
``[0, 2]`` means "A or C is fine" while ``[[0, 2]]`` means "both A and C must be marked".
Blank is always wrong; a multi-mark is right only if that exact set is accepted.

Legacy sheets (printed with the 16-bit form id strip, result.form_id set) are ingested only when their layout was
registered (``register_legacy_layout``): form_key "F%04X", subject "".  Printing needs no database: PDFs are
rendered from a Template (``form_pdf``); only numbered print runs (serials) are recorded, per form_key.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import os
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import db
from .identity import serial_of, sheet_key as result_sheet_key
from .layout import Layout, Template, build_layout, describe, header_from_form_key
from .recognize import hard_flags, layout_for_header

OK = "ok"
KINDS = ("new", "duplicate", "conflict", "unassigned", "rejected", "unchanged", "moved")


@dataclass
class IngestOutcome:
    kind: str                       # new | duplicate | conflict | unassigned | rejected | unchanged | moved
    scan_id: Optional[int] = None
    exam_id: Optional[int] = None
    submission_id: Optional[int] = None
    student_id: Optional[str] = None
    page: Optional[int] = None
    confirm_count: int = 0
    minor: bool = False             # conflict only: differences are ambiguous-vs-confident reads
    reason: str = ""
    subject: Optional[str] = None   # the exam's subject: digits, "" = form without subject code, None = not readable
    form_key: Optional[str] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def norm_student(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def norm_subject(v) -> Optional[str]:
    """Digits of a subject code, or None for missing / empty input."""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    if not s.isdigit():
        raise ValueError("과목코드는 숫자여야 합니다")
    return s


def _guessed(result: dict, what: str) -> bool:
    """The result's student ID (what='id') / subject code carries a best guess, not a confident read."""
    return bool(result.get(what + "_guessed")) or (what + "_guessed") in (result.get("flags") or [])


def _answers_of(result: dict) -> Dict[int, dict]:
    """Effective per-question reading.  An uncertain blank row that carries a best guess counts as that single
    choice (graded, exported) but stays ``ambiguous`` and ``guessed`` - it is never confirmed, only visible."""
    out = {}
    for a in result.get("answers", []):
        state, marked, g = a["state"], sorted(int(x) for x in a["marked"]), a.get("guess")
        guessed = g is not None and state == "blank" and bool(a.get("ambiguous"))    # a multi row is never collapsed to one choice
        if guessed:
            state, marked = "single", [int(g)]
        out[int(a["number"])] = {"state": state, "marked": marked, "ambiguous": bool(a.get("ambiguous")),
                                 "confidence": float(a.get("confidence") or 0.0), "guessed": guessed}
    return out


def content_hash(result: dict) -> str:
    qs = [[n, a["state"], a["marked"]] for n, a in sorted(_answers_of(result).items())]
    return hashlib.sha256(json.dumps([result.get("form_key"), result.get("page"), qs]).encode()).hexdigest()


def result_hash(result: dict, student_id: Optional[str]) -> str:
    qs = [[n, a["state"], a["marked"]] for n, a in sorted(_answers_of(result).items())]
    return hashlib.sha256(json.dumps([result.get("form_key"), result.get("page"), student_id, qs]).encode()).hexdigest()


def _quality(row) -> tuple:
    """Sort key, smaller = better: fewer doubtful answers, higher mean confidence, newer."""
    ans = _answers_of(json.loads(row["result_json"])).values()
    bad = sum(1 for a in ans if a["ambiguous"] or a["state"] != "single")
    mean = sum(a["confidence"] for a in ans) / len(ans) if ans else 0.0
    return (bad, -mean, -row["id"])


def same_variant(a, b) -> bool:
    """The one identity test between two scans of the SAME submission (exam, student, page): same reading
    of the same printed copy.  Within a submission the student part of the sheet key is equal by
    construction, so this compares the serial; a different serial = a different physical sheet."""
    return a["content_hash"] == b["content_hash"] and a["serial"] == b["serial"]


def id_signature(result: dict) -> str:
    """Raw per-digit ID reading (state+value), used only when no full ID was read."""
    if norm_student(result.get("student_id")) is not None and not _guessed(result, "id"):
        return ""
    digs = [[d.get("index"), d.get("state"), d.get("value")] for d in result.get("id_digits") or []]
    return hashlib.sha256(json.dumps(digs).encode()).hexdigest()


def _enc(img, jpeg: bool = False) -> Optional[tuple]:
    """(bytes, ext) for an ndarray or raw bytes.  Full camera frames go out as JPEG (q90)."""
    if img is None:
        return None
    ext = ".jpg" if jpeg else ".png"
    if isinstance(img, (bytes, bytearray)):
        return bytes(img), ext
    import cv2
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 90] if jpeg else [])
    if not ok:
        raise ValueError("cannot encode image")
    return buf.tobytes(), ext



class OmrService:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, db.DB_NAME)
        os.makedirs(os.path.join(data_dir, "scans"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "forms"), exist_ok=True)
        db.init_db(self.db_path)
        self._legacy: Dict[int, Layout] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ printing (stateless)
    @staticmethod
    def as_template(template=None, **params) -> Template:
        """A Template from a Template, a dict of its fields and/or keyword fields (unknown names are an error)."""
        if isinstance(template, dict):
            params = {**template, **params}
            template = None
        t = dataclasses.replace(template) if template is not None else Template()
        return dataclasses.replace(t, **params) if params else t

    def form_pdf(self, template=None, **params) -> dict:
        """Render (idempotently) the plain master PDF of a form: no database state is read or written.
        Returns {"path", "form_key", "layout"}; the file name is derived from the template, so equal templates share it."""
        from .render_pdf import render_pdf
        t = self.as_template(template, **params)
        layout = build_layout(t)
        tag = hashlib.sha1(json.dumps(dataclasses.asdict(layout.template), sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]
        path = os.path.join(self.data_dir, "forms", f"{layout.form_key}_{tag}.pdf")
        if not os.path.exists(path):
            tmp = path + ".tmp"
            render_pdf(layout, tmp)
            os.replace(tmp, path)
        return {"path": path, "form_key": layout.form_key, "layout": layout}

    # ------------------------------------------------------------------ exams
    def create_exam(self, title: str, template: Optional[Template] = None, subject=None, **params) -> dict:
        """Pre-create the exam of (form of ``template``, ``subject``) to enter its answer key before scanning.
        ``subject``: digits (as many as the form has subject digits); forms without subject code take none."""
        t = self.as_template(template, **params)
        layout = build_layout(t)
        subject = norm_subject(subject)
        nd = layout.template.subject_digits
        if nd == 0:
            if subject is not None:
                raise ValueError("이 양식에는 과목코드가 없습니다")
            subject = ""
        elif subject is None or len(subject) != nd:
            raise ValueError(f"과목코드는 {nd}자리 숫자여야 합니다")
        with db.session(self.db_path, write=True) as c:
            if self._exam_row(c, layout.form_key, subject) is not None:
                raise ValueError("이미 있는 과목입니다")
            row = self._insert_exam(c, layout.form_key, subject, layout, "manual", title or None)
            self._act(c, row["id"], "create_exam", detail={"form_key": layout.form_key, "subject": subject})
        return self.get_exam(row["id"])

    @staticmethod
    def _exam_row(c, form_key, subject):
        return c.execute("SELECT * FROM exams WHERE form_key=? AND IFNULL(subject,'?')=IFNULL(?,'?')",
                         (form_key, subject)).fetchone()

    @staticmethod
    def default_title(subject: Optional[str]) -> str:
        return "과목코드 미확인" if subject is None else f"과목 {subject}" if subject else "시험"

    def _insert_exam(self, c, form_key, subject, layout: Layout, origin: str, title: Optional[str] = None):
        cur = c.execute("INSERT INTO exams(title, form_key, subject, layout_json, origin, created_at) VALUES(?,?,?,?,?,?)",
                        (title or self.default_title(subject), form_key, subject, layout.to_json(), origin, db.now()))
        return c.execute("SELECT * FROM exams WHERE id=?", (cur.lastrowid,)).fetchone()

    def _ensure_exam(self, c, form_key, subject, layout: Layout):
        """The exam of (form_key, subject), created on first use (the first scan of a subject makes its exam)."""
        row = self._exam_row(c, form_key, subject)
        if row is None:
            row = self._insert_exam(c, form_key, subject, layout, "auto")
            self._act(c, row["id"], "auto_create_exam", detail={"form_key": form_key, "subject": subject})
        return row

    def rename_exam(self, exam_id: int, title: str) -> dict:
        title = (title or "").strip()
        if not title:
            raise ValueError("제목이 비어 있습니다")
        with db.session(self.db_path, write=True) as c:
            if c.execute("SELECT 1 FROM exams WHERE id=?", (exam_id,)).fetchone() is None:
                raise KeyError(f"no exam {exam_id}")
            c.execute("UPDATE exams SET title=? WHERE id=?", (title, exam_id))
            self._act(c, exam_id, "rename_exam", detail={"title": title})
        return self.get_exam(exam_id)

    def get_exam(self, exam_id: int) -> dict:
        with db.session(self.db_path) as c:
            r = c.execute("SELECT id,title,form_key,subject,origin,created_at,answer_key_json,points_json FROM exams WHERE id=?",
                          (exam_id,)).fetchone()
        if r is None:
            raise KeyError(f"no exam {exam_id}")
        d = dict(r)
        d["answer_key"] = _key_from_json(r["answer_key_json"])
        d["points"] = _points_from_json(r["points_json"])
        return d

    def list_exams(self) -> List[dict]:
        with db.session(self.db_path) as c:
            return [dict(r) for r in c.execute(
                "SELECT id,title,form_key,subject,origin,created_at FROM exams ORDER BY form_key, subject IS NULL, subject")]

    def find_exam(self, form_key: str, subject: Optional[str]) -> Optional[dict]:
        with db.session(self.db_path) as c:
            r = self._exam_row(c, form_key, subject)
        return None if r is None else self.get_exam(r["id"])

    # ------------------------------------------------------------------ legacy sheets
    def register_legacy_layout(self, layout: Layout) -> str:
        """Make sheets printed with the old 16-bit form id strip ingestible (form_key "F%04X", subject "").
        Their layout cannot be derived from the sheet, so it has to be given.  Returns the form_key."""
        if layout.form_id is None:
            raise ValueError("not a legacy layout (it describes itself)")
        key = layout.form_key or f"F{layout.form_id:04X}"
        with db.session(self.db_path, write=True) as c:
            if c.execute("SELECT 1 FROM legacy_forms WHERE form_key=?", (key,)).fetchone() is None:
                c.execute("INSERT INTO legacy_forms(form_key,form_id,layout_json,created_at) VALUES(?,?,?,?)",
                          (key, layout.form_id, layout.to_json(), db.now()))
        self._legacy[layout.form_id] = layout
        return key

    def resolve_layout(self, form_id: int) -> Optional[Layout]:
        """Callable handed to ``recognize`` for LEGACY sheets only: form_id -> registered Layout (None if unknown)."""
        if form_id is None:
            return None
        form_id = int(form_id)
        lay = self._legacy.get(form_id)
        if lay is None:
            with db.session(self.db_path) as c:
                r = c.execute("SELECT layout_json FROM legacy_forms WHERE form_id=?", (form_id,)).fetchone()
            if r is None:
                return None
            lay = self._legacy[form_id] = Layout.from_json(r["layout_json"])
        return lay

    def exam_layout(self, exam_id: int) -> Layout:
        with db.session(self.db_path) as c:
            return self._layout_of_exam(c, exam_id)

    def _result_layout(self, d: dict):
        """(Layout, "") of an ok result, or (None, reason): derived from its form_key, or registered (legacy)."""
        key = d.get("form_key")
        if d.get("form_id") is not None:                      # legacy strip
            lay = self.resolve_layout(d["form_id"])
            if lay is None:
                return None, f"unknown legacy form {key} (register its layout)"
            return lay, ""
        try:
            return layout_for_header(header_from_form_key(key)), ""
        except (ValueError, TypeError):
            return None, f"bad form_key {key!r}"

    # ------------------------------------------------------------------ numbered print runs
    def create_print_run(self, template=None, copies: int = 1, start: Optional[int] = None, **params) -> dict:
        """Reserve ``copies`` serials for a print run of a form and return them.  ``start=None`` continues after the
        highest serial ever issued for this form_key; an explicit start must not overlap an earlier run."""
        from . import markers as mk
        copies = int(copies)
        if copies < 1:
            raise ValueError("copies must be >= 1")
        t = self.as_template(template, **params)
        layout = build_layout(t)
        key = layout.form_key
        with db.session(self.db_path, write=True) as c:
            runs = c.execute("SELECT start_serial, copies FROM print_runs WHERE form_key=?", (key,)).fetchall()
            nxt = max([r[0] + r[1] for r in runs], default=1)
            start = nxt if start is None else int(start)
            if start < 1 or start + copies - 1 > mk.SERIAL_MAX:
                raise ValueError(f"serials {start}..{start + copies - 1} out of range 1..{mk.SERIAL_MAX}")
            for r in runs:
                if start < r[0] + r[1] and r[0] < start + copies:
                    raise ValueError(f"serials {start}..{start + copies - 1} overlap an earlier print run")
            cur = c.execute("INSERT INTO print_runs(form_key,start_serial,copies,template_json,created_at) VALUES(?,?,?,?,?)",
                            (key, start, copies, json.dumps(dataclasses.asdict(layout.template), ensure_ascii=False), db.now()))
        return {"id": cur.lastrowid, "form_key": key, "start": start, "copies": copies,
                "serials": list(range(start, start + copies))}

    def run_pdf(self, run_id: int) -> dict:
        """Render (idempotently) the PDF of a recorded print run: one numbered set per serial."""
        from .render_pdf import render_pdf
        with db.session(self.db_path) as c:
            r = c.execute("SELECT * FROM print_runs WHERE id=?", (run_id,)).fetchone()
        if r is None:
            raise KeyError(f"no print run {run_id}")
        lay = build_layout(Template(**{"tracks": False, **json.loads(r["template_json"])}))   # runs recorded before tracks existed: no tracks
        serials = list(range(r["start_serial"], r["start_serial"] + r["copies"]))
        path = os.path.join(self.data_dir, "forms", f"{r['form_key']}_run{run_id}_no{serials[0]:04d}-{serials[-1]:04d}.pdf")
        if not os.path.exists(path):
            tmp = path + ".tmp"
            render_pdf(lay, tmp, serials=serials)
            os.replace(tmp, path)
        return {"path": path, "serials": serials, "pages": len(serials) * lay.total_pages, "form_key": r["form_key"]}

    # ------------------------------------------------------------------ ingest
    def _prepare(self, result, images):
        """Validate a result for storage -> IngestOutcome("rejected") or (d, layout, form_key, page, subject, raw_id, paths)."""
        d = result if isinstance(result, dict) else result.to_dict()
        images = dict(images or {})
        for k, attr in (("warped", "warped"), ("name_crop", "name_crop")):
            if images.get(k) is None and not isinstance(result, dict):
                images[k] = getattr(result, attr, None)

        if d.get("status") != OK:
            return IngestOutcome("rejected", reason=f"status={d.get('status')}")
        layout, why = self._result_layout(d)
        if layout is None:
            return IngestOutcome("rejected", form_key=d.get("form_key"), reason=why)
        form_key = layout.form_key
        page = d.get("page")
        if not isinstance(page, int) or not 0 <= page < layout.total_pages:
            return IngestOutcome("rejected", form_key=form_key, reason=f"bad page {page}")
        if d.get("total_pages") not in (None, layout.total_pages):
            return IngestOutcome("rejected", form_key=form_key, reason="total_pages mismatch")
        pg = layout.pages[page]
        if sorted(_answers_of(d)) != list(range(pg.first_question, pg.first_question + pg.n_questions)):
            return IngestOutcome("rejected", form_key=form_key, reason="answer numbering does not match layout")
        nd = layout.template.subject_digits
        subject = d.get("subject")
        if nd == 0:
            if subject not in ("", None):
                return IngestOutcome("rejected", form_key=form_key, reason="subject read on a form without subject code")
            subject = ""
        elif _guessed(d, "subject") or not (isinstance(subject, str) and subject.isdigit() and len(subject) == nd):
            subject = None       # unreadable / blank / inconsistent / only a best guess: holding exam (the guess stays in result_json)

        raw_id = None if _guessed(d, "id") else norm_student(d.get("student_id"))   # a guessed ID is never a confirmed student
        paths = self._save_images(form_key, images)
        return d, layout, form_key, page, subject, raw_id, paths

    def ingest(self, result, images: Optional[dict] = None, source: str = "stream") -> IngestOutcome:
        """Store the scan in the exam of its (form_key, subject) - created on first use - then apply the
        duplicate/conflict policy.  A scan whose subject code was not readable goes to the form's holding exam
        (subject NULL) and is never placed: a human routes it with ``route_scan``.

        ``result``: RecognitionResult or its ``to_dict()``.  ``images``: optional dict with
        ``original`` / ``warped`` / ``name_crop`` (ndarray or PNG/JPEG bytes); missing ones are
        taken from ``result.warped`` / ``result.name_crop`` when present.
        """
        if source not in ("stream", "upload"):
            raise ValueError("source must be 'stream' or 'upload'")
        prep = self._prepare(result, images)
        if isinstance(prep, IngestOutcome):
            return prep
        d, layout, form_key, page, subject, raw_id, paths = prep
        with db.session(self.db_path, write=True) as c:
            exam = self._ensure_exam(c, form_key, subject, layout)
            cur = c.execute(
                "INSERT INTO scans(exam_id,page,subject_raw,student_id_raw,result_json,result_hash,content_hash,original_path,"
                "warped_path,name_crop_path,source,created_at,needs_review,id_sig,serial,sheet_key) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (exam["id"], page, subject, raw_id, json.dumps(d), result_hash(d, raw_id), content_hash(d),
                 paths.get("original"), paths.get("warped"), paths.get("name_crop"), source, db.now(),
                 int(bool(d.get("needs_review"))), id_signature(d), serial_of(d), result_sheet_key(d)))
            scan_id = cur.lastrowid
            followed = self._follow_human_routing(c, exam, scan_id)
            if followed is not None:
                return followed
            if subject is None:
                fl = d.get("flags") or []
                reason = ("subject_blank" if "subject_blank" in fl else "subject_guessed" if "subject_guessed" in fl
                          else "subject_unreadable")
                self._act(c, exam["id"], "unassigned", scan_id=scan_id, detail={"reason": reason})
                return IngestOutcome("unassigned", scan_id, exam["id"], page=page, reason=reason, subject=None,
                                     form_key=form_key)
            if raw_id is None:
                self._act(c, exam["id"], "unassigned", scan_id=scan_id)
                return IngestOutcome("unassigned", scan_id, exam["id"], page=page, subject=subject, form_key=form_key)
            return self._place(c, exam["id"], scan_id, raw_id)

    def supersede_scan(self, old_scan_id: int, result, images: Optional[dict] = None) -> IngestOutcome:
        """A REVISED reading of a sheet already ingested as ``old_scan_id`` (omr.stream "revision" event): the new
        result is stored as a new scan that takes the old one's place - the old row is kept for audit and leaves every
        queue through a ``scan_moves`` record (note "revision"), exactly like a human move.  The policy then runs for the
        new scan, so it neither duplicates nor conflicts with the scan it replaces: a revised ID that is now confident
        assigns a previously unassigned scan, a confident subject routes it out of the holding exam.
        Human decisions win: a student assigned by hand stays (travels to the new scan), a hand-routed subject stays,
        and answer overrides stay on the submission (they are keyed by submission, not scan)."""
        with db.session(self.db_path, write=True) as c:
            old = c.execute("SELECT * FROM scans WHERE id=?", (old_scan_id,)).fetchone()
            if old is None:
                raise KeyError(f"no scan {old_scan_id}")
            human_route = False
            for _ in range(32):        # follow the chain to the live copy
                m = c.execute("SELECT * FROM scan_moves WHERE from_scan_id=?", (old["id"],)).fetchone()
                if m is None:
                    break
                human_route = human_route or m["note"] != "revision"
                old = c.execute("SELECT * FROM scans WHERE id=?", (m["to_scan_id"],)).fetchone()
        prep = self._prepare(result, images)
        if isinstance(prep, IngestOutcome):
            return prep
        d, layout, form_key, page, subject, raw_id, paths = prep
        with db.session(self.db_path, write=True) as c:
            old_exam = c.execute("SELECT * FROM exams WHERE id=?", (old["exam_id"],)).fetchone()
            if form_key != old_exam["form_key"] or page != old["page"]:
                return IngestOutcome("rejected", form_key=form_key, reason="revision is another form / page")
            exam = old_exam if human_route and old_exam["subject"] is not None else self._ensure_exam(c, form_key, subject, layout)
            own = c.execute("SELECT student_id FROM scan_owner WHERE scan_id=?", (old["id"],)).fetchone()
            old_owner = own["student_id"]
            human_owner = c.execute("SELECT student_id FROM assignments WHERE scan_id=? ORDER BY id DESC LIMIT 1",
                                    (old["id"],)).fetchone()
            cur = c.execute(
                "INSERT INTO scans(exam_id,page,subject_raw,moved_from,student_id_raw,result_json,result_hash,content_hash,original_path,"
                "warped_path,name_crop_path,source,created_at,needs_review,id_sig,serial,sheet_key) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (exam["id"], page, old["subject_raw"] if human_route else subject, old["id"], raw_id, json.dumps(d), result_hash(d, raw_id),
                 content_hash(d), paths.get("original") or old["original_path"], paths.get("warped") or old["warped_path"],
                 paths.get("name_crop") or old["name_crop_path"], old["source"], db.now(), int(bool(d.get("needs_review"))),
                 id_signature(d), serial_of(d), result_sheet_key(d)))
            new_id = cur.lastrowid
            c.execute("INSERT INTO scan_moves(from_scan_id,to_scan_id,from_exam_id,to_exam_id,note,created_at) VALUES(?,?,?,?,?,?)",
                      (old["id"], new_id, old["exam_id"], exam["id"], "revision", db.now()))
            self._act(c, old["exam_id"], "revision", scan_id=old["id"], student_id=old_owner,
                      detail={"to_scan": new_id, "to_exam": exam["id"], "flags": d.get("flags")})
            if old_owner is not None:
                osub = c.execute("SELECT id FROM submissions WHERE exam_id=? AND student_id=? AND page=?",
                                 (old["exam_id"], old_owner, page)).fetchone()
                if osub:
                    self._after_removal(c, osub["id"])
            owner = raw_id
            if human_owner is not None:                    # a person's student decision about the sheet outlives the revision
                owner = human_owner["student_id"]
                c.execute("INSERT INTO assignments(scan_id,student_id,note,created_at) VALUES(?,?,?,?)",
                          (new_id, owner, "carried over (revision)", db.now()))
            if exam["subject"] is None:
                fl = d.get("flags") or []
                reason = ("subject_blank" if "subject_blank" in fl else "subject_guessed" if "subject_guessed" in fl
                          else "subject_unreadable")
                self._act(c, exam["id"], "unassigned", scan_id=new_id, detail={"reason": reason})
                return IngestOutcome("unassigned", new_id, exam["id"], page=page, reason=reason, subject=None, form_key=form_key)
            if owner is None:
                self._act(c, exam["id"], "unassigned", scan_id=new_id)
                return IngestOutcome("unassigned", new_id, exam["id"], page=page, subject=exam["subject"], form_key=form_key)
            return self._place(c, exam["id"], new_id, owner)

    def _decision_of(self, c, scan) -> tuple:
        """(exam_id, owner) a human decision on ``scan`` led to: a moved scan is followed to its (final) copy."""
        for _ in range(16):
            m = c.execute("SELECT to_scan_id FROM scan_moves WHERE from_scan_id=?", (scan["id"],)).fetchone()
            if m is None:
                break
            scan = c.execute("SELECT * FROM scans WHERE id=?", (m[0],)).fetchone()
        owner = c.execute("SELECT student_id FROM scan_owner WHERE scan_id=?", (scan["id"],)).fetchone()[0]
        return scan["exam_id"], owner

    def _follow_human_routing(self, c, exam, scan_id: int) -> Optional[IngestOutcome]:
        """If a human already routed an identical-looking sheet, route this one the same way.

        Candidates: scans with a manual decision (a student assignment, or a move to another subject), same
        exam/page, same ``sheet_key`` (printed serial + subject and ID reading, see omr/identity.py) and same
        content.  Followed only when all candidates agree on one (exam, owner); otherwise the normal policy runs.
        Known limitation: two different students with the same ID reading and identical answers are
        indistinguishable here (a name-crop comparison could be added at this seam).
        """
        exam_id = exam["id"]
        this = c.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
        cands = c.execute(
            "SELECT s.* FROM scans s WHERE s.id!=? AND s.exam_id=? AND s.page=? AND s.sheet_key=? AND s.content_hash=? "
            "AND (EXISTS (SELECT 1 FROM assignments a WHERE a.scan_id=s.id) "
            "OR EXISTS (SELECT 1 FROM scan_moves m WHERE m.from_scan_id=s.id))",
            (scan_id, exam_id, this["page"], this["sheet_key"], this["content_hash"])).fetchall()
        decisions = {self._decision_of(c, r) for r in cands}
        if len(decisions) != 1:
            return None
        target_exam, owner = decisions.pop()
        if target_exam != exam_id:
            return self._move(c, scan_id, target_exam, owner, "auto-follow", follow_ids=[r["id"] for r in cands])
        if exam["subject"] is None or owner == this["student_id_raw"]:
            return None
        c.execute("INSERT INTO assignments(scan_id,student_id,note,created_at) VALUES(?,?,?,?)",
                  (scan_id, owner, "auto-follow", db.now()))
        self._act(c, exam_id, "follow_routing", scan_id=scan_id, student_id=owner,
                  detail={"matched_scans": [r["id"] for r in cands], "raw": this["student_id_raw"]})
        if owner is None:
            return IngestOutcome("unassigned", scan_id, exam_id, page=this["page"], subject=exam["subject"],
                                 form_key=exam["form_key"])
        return self._place(c, exam_id, scan_id, owner)

    def _save_images(self, folder_name: str, images: dict) -> Dict[str, str]:
        out = {}
        token = uuid.uuid4().hex
        folder = os.path.join(self.data_dir, "scans", str(folder_name))
        for k in ("original", "warped", "name_crop"):
            e = _enc(images.get(k), jpeg=(k == "original"))
            if e is None:
                continue
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{token}_{k}{e[1]}")
            with open(path, "wb") as f:
                f.write(e[0])
            out[k] = path
        return out

    # ------------------------------------------------------------------ policy core
    def _act(self, c, exam_id, kind, scan_id=None, submission_id=None, student_id=None, detail=None):
        c.execute("INSERT INTO actions(exam_id,kind,scan_id,submission_id,student_id,detail_json,created_at) "
                  "VALUES(?,?,?,?,?,?,?)",
                  (exam_id, kind, scan_id, submission_id, student_id,
                   None if detail is None else json.dumps(detail), db.now()))

    def _members(self, c, exam_id, student_id, page) -> list:
        return c.execute("SELECT s.* FROM scans s JOIN scan_owner o ON o.scan_id=s.id "
                         "WHERE o.exam_id=? AND o.student_id=? AND o.page=? ORDER BY s.id",
                         (exam_id, student_id, page)).fetchall()

    def _place(self, c, exam_id: int, scan_id: int, student_id: str) -> IngestOutcome:
        """Run the policy for a scan whose (effective) student is ``student_id``."""
        this = c.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
        page = this["page"]
        members = self._members(c, exam_id, student_id, page)
        others = [m for m in members if m["id"] != scan_id]
        sub = c.execute("SELECT * FROM submissions WHERE exam_id=? AND student_id=? AND page=?",
                        (exam_id, student_id, page)).fetchone()

        ex = c.execute("SELECT subject, form_key FROM exams WHERE id=?", (exam_id,)).fetchone()

        def out(kind, sub_id, **kw):
            s = c.execute("SELECT confirm_count FROM submissions WHERE id=?", (sub_id,)).fetchone()
            self._act(c, exam_id, kind, scan_id=scan_id, submission_id=sub_id, student_id=student_id,
                      detail=kw or None)
            return IngestOutcome(kind, scan_id, exam_id, sub_id, student_id, page, s["confirm_count"],
                                 subject=ex["subject"], form_key=ex["form_key"], **kw)

        if sub is None:
            cur = c.execute("INSERT INTO submissions(exam_id,student_id,page,adopted_scan_id,status,confirm_count,updated_at)"
                            " VALUES(?,?,?,?,?,?,?)", (exam_id, student_id, page, scan_id, "ok", 1, db.now()))
            return out("new", cur.lastrowid)
        if sub["status"] == "empty" or not others:
            c.execute("UPDATE submissions SET adopted_scan_id=?, adopted_by='auto', status='ok', confirm_count=?, "
                      "diff_json=NULL, minor=0, updated_at=? WHERE id=?",
                      (scan_id, len([m for m in members if same_variant(m, this)]), db.now(), sub["id"]))
            return out("new", sub["id"])

        adopted = c.execute("SELECT * FROM scans WHERE id=?", (sub["adopted_scan_id"],)).fetchone()
        if sub["status"] == "ok":
            if same_variant(adopted, this):
                self._set_confirm(c, sub["id"], members, adopted)
                return out("duplicate", sub["id"])
            minor = self._open_conflict(c, sub, members, keep_human=True)
            return out("conflict", sub["id"], minor=minor)
        # already in conflict
        if any(same_variant(m, this) for m in others):
            self._set_confirm(c, sub["id"], members, adopted)
            return out("duplicate", sub["id"])
        minor = self._open_conflict(c, sub, members, keep_human=True)
        return out("conflict", sub["id"], minor=minor)

    def _set_confirm(self, c, sub_id, members, adopted):
        c.execute("UPDATE submissions SET confirm_count=?, updated_at=? WHERE id=?",
                  (sum(1 for m in members if same_variant(m, adopted)), db.now(), sub_id))

    def _open_conflict(self, c, sub, members, keep_human: bool) -> bool:
        """(Re)compute provisional adoption + diff for a submission whose members disagree."""
        by_id = {m["id"]: m for m in members}
        if keep_human and sub["adopted_by"] == "human" and sub["adopted_scan_id"] in by_id:
            adopted, by = by_id[sub["adopted_scan_id"]], "human"
        else:
            adopted, by = min(members, key=_quality), "auto"
        diff, minor = self._diff(adopted, members)
        c.execute("UPDATE submissions SET adopted_scan_id=?, adopted_by=?, status='conflict', confirm_count=?, "
                  "diff_json=?, minor=?, updated_at=? WHERE id=?",
                  (adopted["id"], by, sum(1 for m in members if same_variant(m, adopted)),
                   json.dumps(diff), int(minor), db.now(), sub["id"]))
        return minor

    @staticmethod
    def _diff(adopted, members):
        a_ans = _answers_of(json.loads(adopted["result_json"]))
        groups: List[list] = []
        for m in members:
            if not same_variant(m, adopted):
                for g in groups:
                    if same_variant(g[0], m):
                        g.append(m)
                        break
                else:
                    groups.append([m])
        diff, minor = [], True
        for ms in groups:
            rep = min(ms, key=_quality)
            o_ans = _answers_of(json.loads(rep["result_json"]))
            qs = []
            for q in sorted(set(a_ans) | set(o_ans)):
                a, o = a_ans.get(q), o_ans.get(q)
                if a is not None and o is not None and (a["state"], a["marked"]) == (o["state"], o["marked"]):
                    continue
                qs.append({"question": q, "adopted": a, "other": o})
                if a is None or o is None or a["ambiguous"] == o["ambiguous"]:
                    minor = False
            other_sheet = rep["serial"] != adopted["serial"]
            if other_sheet:
                minor = False
            diff.append({"scan_id": rep["id"], "scan_ids": [m["id"] for m in ms], "questions": qs,
                         "different_sheet": other_sheet, "serial": rep["serial"]})
        return diff, bool(diff) and minor

    def _after_removal(self, c, sub_id: int):
        """Re-settle a submission after one of its scans was assigned elsewhere."""
        sub = c.execute("SELECT * FROM submissions WHERE id=?", (sub_id,)).fetchone()
        members = self._members(c, sub["exam_id"], sub["student_id"], sub["page"])
        if not members:
            c.execute("UPDATE submissions SET adopted_scan_id=NULL, adopted_by='auto', status='empty', confirm_count=0,"
                      " diff_json=NULL, minor=0, updated_at=? WHERE id=?", (db.now(), sub_id))
            return
        ids = {m["id"] for m in members}
        if sub["status"] == "ok" and sub["adopted_scan_id"] in ids:
            ad = next(m for m in members if m["id"] == sub["adopted_scan_id"])
            self._set_confirm(c, sub_id, members, ad)
            return
        if any(not same_variant(m, members[0]) for m in members):
            self._open_conflict(c, sub, members, keep_human=True)
        else:
            best = min(members, key=_quality)
            c.execute("UPDATE submissions SET adopted_scan_id=?, adopted_by='auto', status='ok', confirm_count=?, "
                      "diff_json=NULL, minor=0, updated_at=? WHERE id=?", (best["id"], len(members), db.now(), sub_id))

    # ------------------------------------------------------------------ manual actions
    def assign_scan(self, scan_id: int, student_id: Optional[str], note: Optional[str] = None) -> IngestOutcome:
        """Assign (or reassign) a scan to a student id and run the policy as if it had been read so.

        ``student_id=None`` sends the scan back to the unassigned queue.  Recorded as a new
        ``assignments`` row; the scan itself is never modified.
        """
        with db.session(self.db_path, write=True) as c:
            return self._assign(c, scan_id, norm_student(student_id), note)

    def _assign(self, c, scan_id: int, student_id: Optional[str], note: Optional[str]) -> IngestOutcome:
        scan = c.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
        if scan is None:
            raise KeyError(f"no scan {scan_id}")
        exam = c.execute("SELECT * FROM exams WHERE id=?", (scan["exam_id"],)).fetchone()
        if exam["subject"] is None:
            raise ValueError("과목코드를 먼저 지정하세요 (route_scan)")
        own = c.execute("SELECT student_id, moved FROM scan_owner WHERE scan_id=?", (scan_id,)).fetchone()
        if own["moved"]:
            raise ValueError("이미 다른 과목으로 옮긴 스캔입니다")
        old = own["student_id"]
        exam_id, page = scan["exam_id"], scan["page"]
        ex = dict(subject=exam["subject"], form_key=exam["form_key"])
        if old == student_id:
            sub = None
            if student_id is not None:
                sub = c.execute("SELECT * FROM submissions WHERE exam_id=? AND student_id=? AND page=?",
                                (exam_id, student_id, page)).fetchone()
            return IngestOutcome("unchanged" if student_id is not None else "unassigned", scan_id, exam_id,
                                 sub["id"] if sub else None, student_id, page,
                                 sub["confirm_count"] if sub else 0, bool(sub["minor"]) if sub else False, **ex)
        c.execute("INSERT INTO assignments(scan_id,student_id,note,created_at) VALUES(?,?,?,?)",
                  (scan_id, student_id, note, db.now()))
        self._act(c, exam_id, "assign", scan_id=scan_id, student_id=student_id,
                  detail={"from": old, "note": note})
        if old is not None:
            osub = c.execute("SELECT id FROM submissions WHERE exam_id=? AND student_id=? AND page=?",
                             (exam_id, old, page)).fetchone()
            if osub:
                self._after_removal(c, osub["id"])
        if student_id is None:
            return IngestOutcome("unassigned", scan_id, exam_id, page=page, **ex)
        return self._place(c, exam_id, scan_id, student_id)

    def route_scan(self, scan_id: int, subject=None, student_id=None, note: Optional[str] = None) -> IngestOutcome:
        """Human decision "this scan belongs to subject ``subject`` (and student ``student_id``)".

        A different subject MOVES the scan: a copy of it is added to the target exam (created if needed) with a
        ``scan_moves`` record, and the source scan leaves every queue.  ``student_id`` (optional) sets the student
        of the moved scan; without it the scan keeps the student it had.  Without ``subject`` this only assigns the
        student (not allowed while the subject is still unknown)."""
        subject, student_id = norm_subject(subject), norm_student(student_id)
        with db.session(self.db_path, write=True) as c:
            scan = c.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
            if scan is None:
                raise KeyError(f"no scan {scan_id}")
            exam = c.execute("SELECT * FROM exams WHERE id=?", (scan["exam_id"],)).fetchone()
            if subject is None:
                if exam["subject"] is None:
                    raise ValueError("과목코드를 입력하세요")
                if student_id is None:
                    raise ValueError("과목코드나 수험번호를 입력하세요")
                return self._assign(c, scan_id, student_id, note)
            layout = Layout.from_json(exam["layout_json"])
            nd = layout.template.subject_digits
            if nd == 0:
                raise ValueError("이 양식에는 과목코드가 없습니다")
            if len(subject) != nd:
                raise ValueError(f"과목코드는 {nd}자리 숫자여야 합니다")
            if subject == exam["subject"]:
                if student_id is None:
                    raise ValueError("이미 이 과목의 스캔입니다")
                return self._assign(c, scan_id, student_id, note)
            target = self._ensure_exam(c, exam["form_key"], subject, layout)
            return self._move(c, scan_id, target["id"], student_id, note)

    def _move(self, c, scan_id: int, target_id: int, student_id: Optional[str], note: Optional[str],
              follow_ids=None) -> IngestOutcome:
        """Copy ``scan_id`` into exam ``target_id`` and record the move (the source is never modified).
        ``student_id``: owner of the copy (None = keep the source's owner)."""
        src = c.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
        target = c.execute("SELECT * FROM exams WHERE id=?", (target_id,)).fetchone()
        src_exam = c.execute("SELECT * FROM exams WHERE id=?", (src["exam_id"],)).fetchone()
        own = c.execute("SELECT student_id, moved FROM scan_owner WHERE scan_id=?", (scan_id,)).fetchone()
        if own["moved"]:
            raise ValueError("이미 다른 과목으로 옮긴 스캔입니다")
        if target["form_key"] != src_exam["form_key"] or target["subject"] is None or target["id"] == src_exam["id"]:
            raise ValueError("cannot move a scan to this exam")
        old_owner = own["student_id"]
        cur = c.execute(
            "INSERT INTO scans(exam_id,page,subject_raw,moved_from,student_id_raw,result_json,result_hash,content_hash,id_sig,"
            "serial,sheet_key,original_path,warped_path,name_crop_path,source,created_at,needs_review) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (target_id, src["page"], src["subject_raw"], scan_id, src["student_id_raw"], src["result_json"],
             src["result_hash"], src["content_hash"], src["id_sig"], src["serial"], src["sheet_key"],
             src["original_path"], src["warped_path"], src["name_crop_path"], src["source"], db.now(), src["needs_review"]))
        new_id = cur.lastrowid
        c.execute("INSERT INTO scan_moves(from_scan_id,to_scan_id,from_exam_id,to_exam_id,note,created_at) VALUES(?,?,?,?,?,?)",
                  (scan_id, new_id, src_exam["id"], target_id, note, db.now()))
        self._act(c, src_exam["id"], "move", scan_id=scan_id, student_id=old_owner,
                  detail={"to_exam": target_id, "to_subject": target["subject"], "to_scan": new_id, "note": note,
                          "matched_scans": follow_ids})
        if old_owner is not None:                          # the source no longer counts for its old student
            osub = c.execute("SELECT id FROM submissions WHERE exam_id=? AND student_id=? AND page=?",
                             (src_exam["id"], old_owner, src["page"])).fetchone()
            if osub:
                self._after_removal(c, osub["id"])
        owner = student_id if student_id is not None else old_owner
        if owner != src["student_id_raw"]:                 # a human decision about the student travels with the scan
            c.execute("INSERT INTO assignments(scan_id,student_id,note,created_at) VALUES(?,?,?,?)",
                      (new_id, owner, note or "moved", db.now()))
        if owner is None:
            self._act(c, target_id, "unassigned", scan_id=new_id)
            return IngestOutcome("unassigned", new_id, target_id, page=src["page"], subject=target["subject"],
                                 form_key=target["form_key"], reason="moved")
        return self._place(c, target_id, new_id, owner)

    def resolve_adopt(self, submission_id: int, scan_id: int, note: Optional[str] = None) -> dict:
        """Human decision: this scan is the right one for the submission (closes the conflict)."""
        with db.session(self.db_path, write=True) as c:
            sub = c.execute("SELECT * FROM submissions WHERE id=?", (submission_id,)).fetchone()
            if sub is None:
                raise KeyError(f"no submission {submission_id}")
            members = self._members(c, sub["exam_id"], sub["student_id"], sub["page"])
            m = next((x for x in members if x["id"] == scan_id), None)
            if m is None:
                raise ValueError("scan does not belong to this submission")
            c.execute("UPDATE submissions SET adopted_scan_id=?, adopted_by='human', status='ok', confirm_count=?, "
                      "diff_json=NULL, minor=0, updated_at=? WHERE id=?",
                      (scan_id, sum(1 for x in members if same_variant(x, m)), db.now(),
                       submission_id))
            self._act(c, sub["exam_id"], "resolve_adopt", scan_id=scan_id, submission_id=submission_id,
                      student_id=sub["student_id"],
                      detail={"previous_adopted": sub["adopted_scan_id"], "previous_diff": sub["diff_json"], "note": note})
        return self.get_submission(submission_id)

    def resolve_reassign(self, submission_id: int, scan_id: int, student_id: str, note: Optional[str] = None) -> IngestOutcome:
        """Human decision: this scan is a *different* student's sheet; move it to ``student_id``."""
        student_id = norm_student(student_id)
        if student_id is None:
            raise ValueError("student_id required")
        with db.session(self.db_path) as c:
            sub = c.execute("SELECT * FROM submissions WHERE id=?", (submission_id,)).fetchone()
            if sub is None:
                raise KeyError(f"no submission {submission_id}")
            if not any(m["id"] == scan_id for m in self._members(c, sub["exam_id"], sub["student_id"], sub["page"])):
                raise ValueError("scan does not belong to this submission")
        return self.assign_scan(scan_id, student_id, note or "resolve_reassign")

    def override_answer(self, submission_id: int, question: int, marked: Optional[List[int]], note: Optional[str] = None):
        """Human correction of one answer (``marked=[]`` = blank, ``None`` = remove override)."""
        with db.session(self.db_path, write=True) as c:
            sub = c.execute("SELECT * FROM submissions WHERE id=?", (submission_id,)).fetchone()
            if sub is None:
                raise KeyError(f"no submission {submission_id}")
            lay = self._layout_of_exam(c, sub["exam_id"])
            pg = lay.pages[sub["page"]]
            if not pg.first_question <= question < pg.first_question + pg.n_questions:
                raise ValueError("question not on this page")
            if marked is not None:
                marked = sorted({int(x) for x in marked})
                if any(not 0 <= x < lay.template.choices for x in marked):
                    raise ValueError("choice out of range")
            c.execute("INSERT INTO answer_overrides(submission_id,question,marked_json,note,created_at) VALUES(?,?,?,?,?)",
                      (submission_id, question, None if marked is None else json.dumps(marked), note, db.now()))
            self._act(c, sub["exam_id"], "override", submission_id=submission_id, student_id=sub["student_id"],
                      detail={"question": question, "marked": marked, "note": note})

    def _layout_of_exam(self, c, exam_id) -> Layout:
        return Layout.from_json(c.execute("SELECT layout_json FROM exams WHERE id=?", (exam_id,)).fetchone()[0])

    # ------------------------------------------------------------------ queries
    @staticmethod
    def _scan_info(r, adopted_id=None) -> dict:
        ans = _answers_of(json.loads(r["result_json"])).values()
        return {"scan_id": r["id"], "page": r["page"], "student_id_raw": r["student_id_raw"], "source": r["source"],
                "subject_raw": r["subject_raw"], "moved_from": r["moved_from"],
                "created_at": r["created_at"], "needs_review": bool(r["needs_review"]),
                "n_doubtful": sum(1 for a in ans if a["ambiguous"] or a["state"] != "single"),
                "mean_confidence": (sum(a["confidence"] for a in ans) / len(ans)) if ans else 0.0,
                "original_path": r["original_path"], "warped_path": r["warped_path"],
                "name_crop_path": r["name_crop_path"], "adopted": r["id"] == adopted_id,
                "has_original": bool(r["original_path"]), "has_warped": bool(r["warped_path"]),
                "has_name_crop": bool(r["name_crop_path"]),
                "content_hash": r["content_hash"], "serial": r["serial"]}

    def get_submission(self, submission_id: int) -> dict:
        """Submission + all its scans (with name-crop paths) + conflict diff, for review UIs."""
        with db.session(self.db_path) as c:
            s = c.execute("SELECT * FROM submissions WHERE id=?", (submission_id,)).fetchone()
            if s is None:
                raise KeyError(f"no submission {submission_id}")
            members = self._members(c, s["exam_id"], s["student_id"], s["page"])
        return {"id": s["id"], "exam_id": s["exam_id"], "student_id": s["student_id"], "page": s["page"],
                "status": s["status"], "adopted_scan_id": s["adopted_scan_id"], "adopted_by": s["adopted_by"],
                "confirm_count": s["confirm_count"], "minor": bool(s["minor"]),
                "different_sheets": s["status"] == "conflict" and len({m["serial"] for m in members}) > 1,
                "label": ("같은 수험번호, 다른 시트" if s["status"] == "conflict" and len({m["serial"] for m in members}) > 1
                          else None),
                "diff": json.loads(s["diff_json"]) if s["diff_json"] else None,
                "scans": [self._scan_info(m, s["adopted_scan_id"]) for m in members]}

    def list_conflicts(self, exam_id: int) -> List[dict]:
        with db.session(self.db_path) as c:
            ids = [r[0] for r in c.execute("SELECT id FROM submissions WHERE exam_id=? AND status='conflict' ORDER BY id",
                                           (exam_id,))]
        return [self.get_submission(i) for i in ids]

    @staticmethod
    def _unassigned_where(c, exam_id: int) -> str:
        """SQL condition (on scan_owner o) of the scans a human still has to route: in a holding exam (subject
        unknown) every scan that was not moved yet, elsewhere those without a student."""
        subj = c.execute("SELECT subject FROM exams WHERE id=?", (exam_id,)).fetchone()
        if subj is None:
            raise KeyError(f"no exam {exam_id}")
        return "o.moved=0" if subj[0] is None else "o.moved=0 AND o.student_id IS NULL"

    def list_unassigned(self, exam_id: int) -> List[dict]:
        with db.session(self.db_path) as c:
            rows = c.execute(f"SELECT s.* FROM scans s JOIN scan_owner o ON o.scan_id=s.id "
                             f"WHERE o.exam_id=? AND {self._unassigned_where(c, exam_id)} ORDER BY s.id", (exam_id,)).fetchall()
        return [self._scan_info(r) for r in rows]

    _RECENT_SQL = ("SELECT s.id, s.exam_id, s.page, s.student_id_raw, s.subject_raw, s.serial, s.source, s.created_at, "
                   "s.needs_review, o.moved, e.subject AS exam_subject, e.title AS exam_title, "
                   "CASE WHEN o.moved THEN 'moved' ELSE "
                   "(SELECT a.kind FROM actions a WHERE a.scan_id=s.id ORDER BY a.id LIMIT 1) END AS kind "
                   "FROM scans s JOIN scan_owner o ON o.scan_id=s.id JOIN exams e ON e.id=s.exam_id ")

    @staticmethod
    def _recent_row(r) -> dict:
        return {"scan_id": r["id"], "exam_id": r["exam_id"], "exam_title": r["exam_title"], "page": r["page"],
                "student_id": r["student_id_raw"], "subject": r["exam_subject"], "serial": r["serial"],
                "source": r["source"], "created_at": r["created_at"], "needs_review": bool(r["needs_review"]),
                "kind": r["kind"]}

    def recent_scans(self, exam_id: Optional[int] = None, limit: int = 30) -> List[dict]:
        """Newest scans (of one exam, or of all) with the policy outcome recorded for each (``kind``)."""
        limit = max(1, min(int(limit), 500))
        with db.session(self.db_path) as c:
            if exam_id is None:
                rows = c.execute(self._RECENT_SQL + "WHERE o.moved=0 ORDER BY s.id DESC LIMIT ?", (limit,)).fetchall()
            else:
                rows = c.execute(self._RECENT_SQL + "WHERE s.exam_id=? ORDER BY s.id DESC LIMIT ?", (exam_id, limit)).fetchall()
        return [self._recent_row(r) for r in rows]

    def scan_count(self, exam_id: Optional[int] = None) -> int:
        with db.session(self.db_path) as c:
            if exam_id is None:      # physical scans: the copies made by moves are not counted twice
                return c.execute("SELECT COUNT(*) FROM scans WHERE moved_from IS NULL").fetchone()[0]
            return c.execute("SELECT COUNT(*) FROM scans WHERE exam_id=?", (exam_id,)).fetchone()[0]

    # ------------------------------------------------------------------ student view
    def _effective(self, c, sub) -> Dict[int, dict]:
        out: Dict[int, dict] = {}
        if sub["adopted_scan_id"] is not None:
            r = c.execute("SELECT result_json FROM scans WHERE id=?", (sub["adopted_scan_id"],)).fetchone()
            for q, a in _answers_of(json.loads(r[0])).items():
                out[q] = dict(a, source="scan", scan_id=sub["adopted_scan_id"])
        rows = c.execute("SELECT question, marked_json FROM answer_overrides WHERE submission_id=? ORDER BY id",
                         (sub["id"],)).fetchall()
        latest = {r["question"]: r["marked_json"] for r in rows}
        for q, mj in latest.items():
            if mj is None:
                continue
            m = json.loads(mj)
            out[q] = {"state": "blank" if not m else "single" if len(m) == 1 else "multi", "marked": m,
                      "ambiguous": False, "guessed": False, "confidence": 1.0, "source": "override", "scan_id": None}
        return out

    def student_summary(self, exam_id: int, student_id: str) -> dict:
        """Pages present/missing, overall status and effective answers (global question numbers).

        status precedence: conflict > incomplete > needs_review > complete.  ``needs_review`` means
        an effective (non-overridden) answer is ambiguous or multi-marked, or the adopted scan
        carries recognizer flags.  A plain blank answer is not a review reason.
        """
        student_id = norm_student(student_id)
        with db.session(self.db_path) as c:
            lay = self._layout_of_exam(c, exam_id)
            subs = c.execute("SELECT * FROM submissions WHERE exam_id=? AND student_id=? AND status!='empty' ORDER BY page",
                             (exam_id, student_id)).fetchall()
            answers: Dict[int, dict] = {}
            review_flags, conflicts = [], []
            for s in subs:
                answers.update(self._effective(c, s))
                if s["status"] == "conflict":
                    conflicts.append(s["page"])
                if s["adopted_scan_id"] is not None:
                    d = json.loads(c.execute("SELECT result_json FROM scans WHERE id=?", (s["adopted_scan_id"],)).fetchone()[0])
                    # id_blank / id_unreadable only describe the sheet's ID area; once the scan belongs to a
                    # student (a human assigned it) they no longer call for review.
                    flags = [f for f in hard_flags(d.get("flags")) if f not in ("id_blank", "id_unreadable", "id_guessed")]
                    if flags:
                        review_flags.append({"page": s["page"], "flags": flags})
        present = [s["page"] for s in subs]
        missing = [p for p in range(lay.total_pages) if p not in present]
        doubtful = sorted(q for q, a in answers.items() if a["ambiguous"] or a["state"] == "multi")
        guessed = sorted(q for q, a in answers.items() if a.get("guessed") and a["ambiguous"])
        multi = sorted(q for q, a in answers.items() if a["state"] == "multi")
        if conflicts:
            status = "conflict"
        elif missing:
            status = "incomplete"
        elif doubtful or review_flags:
            status = "needs_review"
        else:
            status = "complete"
        return {"exam_id": exam_id, "student_id": student_id, "status": status, "total_pages": lay.total_pages,
                "pages_present": present, "pages_missing": missing, "conflict_pages": conflicts,
                "doubtful_questions": doubtful, "guessed_questions": guessed, "multi_questions": multi, "scan_flags": review_flags, "answers": answers,
                "submission_ids": {s["page"]: s["id"] for s in subs}}

    def list_students(self, exam_id: int) -> List[str]:
        with db.session(self.db_path) as c:
            return [r[0] for r in c.execute("SELECT DISTINCT student_id FROM submissions WHERE exam_id=? "
                                            "AND status!='empty' ORDER BY student_id", (exam_id,))]

    # ------------------------------------------------------------------ answer key + grading
    def set_answer_key(self, exam_id: int, key, points: Optional[dict] = None) -> dict:
        """Set the key.  ``key``: {question: accepted} or a list (question 1..n; None = not graded).

        ``accepted`` is a choice index/label, or a list of accepted answers where each is an index/label
        (single choice) or a list of them (a set to be marked exactly).  Missing questions are not graded.
        ``points``: {question: number}; default 1.  Replaces any previous key and points.
        """
        with db.session(self.db_path, write=True) as c:
            lay = self._layout_of_exam(c, exam_id)
            k = _norm_key(lay, key)
            pts = {}
            for q, v in (points or {}).items():
                q = int(q)
                if q not in k:
                    raise ValueError(f"points given for ungraded question {q}")
                if float(v) < 0:
                    raise ValueError("points must be >= 0")
                pts[q] = float(v)
            c.execute("UPDATE exams SET answer_key_json=?, points_json=? WHERE id=?",
                      (json.dumps({str(q): v for q, v in k.items()}), json.dumps({str(q): v for q, v in pts.items()}),
                       exam_id))
            self._act(c, exam_id, "set_answer_key", detail={"n_questions": len(k)})
        return self.get_exam(exam_id)["answer_key"]

    def answer_key_from_scan(self, scan_ids, apply: bool = True, exam_id: Optional[int] = None) -> dict:
        """Build the key from teacher-marked sheet(s) (one scan per page for multi-page forms).

        Blank questions are left ungraded (listed in ``skipped``); ambiguous ones are used but listed
        in ``ambiguous`` for the teacher to check; a multi-mark becomes "exactly this set".
        Existing per-question points are kept.
        """
        if isinstance(scan_ids, int):
            scan_ids = [scan_ids]
        key, skipped, amb = {}, [], []
        with db.session(self.db_path) as c:
            for sid in scan_ids:
                r = c.execute("SELECT * FROM scans WHERE id=?", (sid,)).fetchone()
                if r is None:
                    raise KeyError(f"no scan {sid}")
                if exam_id not in (None, r["exam_id"]):
                    raise ValueError("scans belong to different exams" if len(scan_ids) > 1
                                     else "scan belongs to a different exam")
                exam_id = r["exam_id"]
                for q, a in _answers_of(json.loads(r["result_json"])).items():
                    if not a["marked"]:
                        skipped.append(q)
                        continue
                    key[q] = [a["marked"][0]] if len(a["marked"]) == 1 else [a["marked"]]
                    if a["ambiguous"]:
                        amb.append(q)
        if apply:
            old = self.get_exam(exam_id)["points"]
            self.set_answer_key(exam_id, key, {q: p for q, p in old.items() if q in key})
        return {"exam_id": exam_id, "key": key, "skipped": sorted(skipped), "ambiguous": sorted(amb)}

    def grade(self, exam_id: int, student_id: str) -> dict:
        """Score one student.  Unanswered/missing-page questions count as wrong (included in max).

        ``provisional`` is true if any graded answer is ambiguous or the student is in conflict / incomplete
        (``reasons`` says which).  Raises ValueError if the exam has no answer key.
        """
        exam = self.get_exam(exam_id)
        key = exam["answer_key"]
        if not key:
            raise ValueError("exam has no answer key")
        st = self.student_summary(exam_id, student_id)
        return _grade(key, exam["points"], st)

    # ------------------------------------------------------------------ export
    def export_csv(self, exam_id: int) -> str:
        exam = self.get_exam(exam_id)
        lay = self.exam_layout(exam_id)
        labels = lay.template.choice_labels
        nq = sum(lay.questions_per_page)
        out = io.StringIO()
        w = csv.writer(out, lineterminator="\n")
        w.writerow(["student_id", "status", "score", "max_score", "provisional", "pages_present", "flags"]
                   + [f"q{q}" for q in range(1, nq + 1)])
        for sid in self.list_students(exam_id):
            st = self.student_summary(exam_id, sid)
            flags = []
            if st["pages_missing"]:
                flags.append("missing_pages:" + ",".join(str(p + 1) for p in st["pages_missing"]))
            if st["conflict_pages"]:
                flags.append("conflict_pages:" + ",".join(str(p + 1) for p in st["conflict_pages"]))
            if st["doubtful_questions"]:
                flags.append("doubtful:" + ",".join(map(str, st["doubtful_questions"])))
            if st["multi_questions"]:
                flags.append("multi:" + ",".join(map(str, st["multi_questions"])))      # 복수마킹; the cell shows the whole set
            if st["guessed_questions"]:
                flags.append("guessed:" + ",".join(map(str, st["guessed_questions"])))
            if st["scan_flags"]:
                flags.append("scan_flags")
            if any(a["source"] == "override" for a in st["answers"].values()):
                flags.append("overrides")
            score = mx = ""
            prov = ""
            if exam["answer_key"]:
                g = _grade(exam["answer_key"], exam["points"], st)
                score, mx, prov = _num(g["score"]), _num(g["max_score"]), str(g["provisional"]).lower()
            cells = []
            for q in range(1, nq + 1):
                a = st["answers"].get(q)
                cells.append("" if a is None else "|".join(labels[i] for i in a["marked"]))
            w.writerow([sid, st["status"], score, mx, prov, ",".join(str(p + 1) for p in st["pages_present"]),
                        ";".join(flags)] + cells)
        return out.getvalue()


    # ------------------------------------------------------------------ review / dashboard queries
    def exam_overview(self, exam_id: int) -> dict:
        """Exam facts + counters shown on the exam list and dashboard header."""
        e = self.get_exam(exam_id)
        lay = self.exam_layout(exam_id)
        t = lay.template
        with db.session(self.db_path) as c:
            n_conf = c.execute("SELECT COUNT(*) FROM submissions WHERE exam_id=? AND status='conflict'", (exam_id,)).fetchone()[0]
            n_un = c.execute(f"SELECT COUNT(*) FROM scan_owner o WHERE o.exam_id=? AND {self._unassigned_where(c, exam_id)}",
                             (exam_id,)).fetchone()[0]
            n_scans = c.execute("SELECT COUNT(*) FROM scan_owner o WHERE o.exam_id=? AND o.moved=0", (exam_id,)).fetchone()[0]
            n_students = c.execute("SELECT COUNT(DISTINCT student_id) FROM submissions WHERE exam_id=? AND status!='empty'",
                                   (exam_id,)).fetchone()[0]
        subject = e["subject"]
        return {"id": e["id"], "title": e["title"], "form_key": e["form_key"], "subject": subject,
                "subject_label": "과목코드 미확인" if subject is None else f"과목 {subject}" if subject else "",
                "origin": e["origin"], "form_desc": describe(lay), "created_at": e["created_at"],
                "n_questions": sum(lay.questions_per_page), "total_pages": lay.total_pages,
                "questions_per_page": list(lay.questions_per_page), "choices": t.choices,
                "choice_labels": list(t.choice_labels), "id_digits": t.id_digits, "subject_digits": t.subject_digits,
                "n_students": n_students, "n_conflicts": n_conf, "n_unassigned": n_un, "n_scans": n_scans,
                "has_key": bool(e["answer_key"]), "n_keyed": len(e["answer_key"])}

    def serial_report(self, exam_id: int) -> dict:
        """Issued serials (print runs of the exam's FORM) vs serials seen on scans of that form (any subject: one
        print run is handed out to several subjects).  Information only: unused blank copies are normal.

        ``unscanned``: issued, never seen on any scan.  ``partial``: serial seen, but not on all pages
        (``pages_missing`` 1-based; ``students`` = owners of its scans).  ``unissued``: seen but not issued
        by this system (printed elsewhere / another database)."""
        lay = self.exam_layout(exam_id)
        with db.session(self.db_path) as c:
            key = c.execute("SELECT form_key FROM exams WHERE id=?", (exam_id,)).fetchone()[0]
            runs = [dict(r) for r in c.execute("SELECT start_serial, copies, created_at FROM print_runs "
                                               "WHERE form_key=? ORDER BY id", (key,))]
            rows = c.execute("SELECT s.serial, s.page, o.student_id FROM scans s JOIN scan_owner o ON o.scan_id=s.id "
                             "JOIN exams e ON e.id=s.exam_id WHERE e.form_key=? AND s.serial!=0 AND o.moved=0", (key,)).fetchall()
        issued = {x for r in runs for x in range(r["start_serial"], r["start_serial"] + r["copies"])}
        seen: Dict[int, dict] = {}
        for r in rows:
            e = seen.setdefault(r["serial"], {"pages": set(), "students": set()})
            e["pages"].add(r["page"])
            if r["student_id"]:
                e["students"].add(r["student_id"])
        partial = [{"serial": sn, "pages_missing": [p + 1 for p in range(lay.total_pages) if p not in e["pages"]],
                    "students": sorted(e["students"])}
                   for sn, e in sorted(seen.items()) if len(e["pages"]) < lay.total_pages]
        return {"runs": runs, "n_issued": len(issued), "n_scanned": len(set(seen) & issued),
                "unscanned": _ranges(sorted(issued - set(seen))), "n_unscanned": len(issued - set(seen)),
                "partial": partial, "unissued": sorted(set(seen) - issued)}

    def dashboard(self, exam_id: int) -> dict:
        """Overview + one row per student (status, score if a key exists, pages present/missing)."""
        ov = self.exam_overview(exam_id)
        exam = self.get_exam(exam_id)
        rows = []
        for sid in self.list_students(exam_id):
            st = self.student_summary(exam_id, sid)
            row = {"student_id": sid, "status": st["status"], "pages_present": [p + 1 for p in st["pages_present"]],
                   "pages_missing": [p + 1 for p in st["pages_missing"]], "n_doubtful": len(st["doubtful_questions"]),
                   "n_overrides": sum(1 for a in st["answers"].values() if a["source"] == "override"),
                   "score": None, "max_score": None, "provisional": None}
            if exam["answer_key"]:
                g = _grade(exam["answer_key"], exam["points"], st)
                row.update(score=g["score"], max_score=g["max_score"], provisional=g["provisional"])
            rows.append(row)
        return {"exam": ov, "students": rows, "serials": self.serial_report(exam_id)}

    def student_detail(self, exam_id: int, student_id: str) -> dict:
        """Everything the student page needs: per-question view vs key, pages, owned scans."""
        student_id = norm_student(student_id)
        if student_id not in self.list_students(exam_id):
            raise KeyError(f"no student {student_id} in exam {exam_id}")
        exam = self.get_exam(exam_id)
        lay = self.exam_layout(exam_id)
        st = self.student_summary(exam_id, student_id)
        key, pts = exam["answer_key"], exam["points"]
        g = _grade(key, pts, st) if key else None
        page_of = {}
        for pg in lay.pages:
            for q in range(pg.first_question, pg.first_question + pg.n_questions):
                page_of[q] = pg.index
        questions = []
        for q in range(1, sum(lay.questions_per_page) + 1):
            a = st["answers"].get(q)
            p = page_of[q]
            item = {"number": q, "page": p, "submission_id": st["submission_ids"].get(p),
                    "state": a["state"] if a else None, "marked": a["marked"] if a else None,
                    "ambiguous": bool(a and a["ambiguous"]), "source": a["source"] if a else None,
                    "scan_id": a["scan_id"] if a else None, "graded": q in key,
                    "accepted": key.get(q), "points": pts.get(q, 1.0) if q in key else None,
                    "correct": None, "earned": None}
            if g and q in g["per_question"]:
                item["correct"] = g["per_question"][q]["correct"]
                item["earned"] = g["per_question"][q]["points"]
            questions.append(item)
        with db.session(self.db_path) as c:
            subs = {r["page"]: r for r in c.execute("SELECT * FROM submissions WHERE exam_id=? AND student_id=?",
                                                    (exam_id, student_id))}
            rows = c.execute(
                "SELECT s.*, (SELECT a.kind FROM actions a WHERE a.scan_id=s.id ORDER BY a.id LIMIT 1) AS kind "
                "FROM scans s JOIN scan_owner o ON o.scan_id=s.id WHERE o.exam_id=? AND o.student_id=? ORDER BY s.id",
                (exam_id, student_id)).fetchall()
        pages = []
        for pg in lay.pages:
            s = subs.get(pg.index)
            pages.append({"page": pg.index, "first_question": pg.first_question, "n_questions": pg.n_questions,
                          "present": pg.index in st["pages_present"],
                          "status": None if s is None else s["status"],
                          "submission_id": None if s is None else s["id"],
                          "adopted_scan_id": None if s is None else s["adopted_scan_id"],
                          "confirm_count": None if s is None else s["confirm_count"]})
        adopted = {p["adopted_scan_id"] for p in pages}
        scans = []
        for r in rows:
            info = self._scan_info(r, r["id"] if r["id"] in adopted else None)
            info["kind"] = r["kind"]
            scans.append(info)
        return {"exam": self.exam_overview(exam_id), "student_id": student_id, "status": st["status"],
                "pages": pages, "pages_missing": [p + 1 for p in st["pages_missing"]],
                "doubtful_questions": st["doubtful_questions"], "scan_flags": st["scan_flags"],
                "questions": questions, "scans": scans,
                "grade": None if g is None else {k: g[k] for k in ("score", "max_score", "provisional", "reasons")}}

    def unassigned_detail(self, exam_id: int) -> List[dict]:
        """Unassigned scans with their per-digit ID reading and suggested students (those missing that page)."""
        ids = self.list_students(exam_id)
        e = self.get_exam(exam_id)
        subjects = sorted(x["subject"] for x in self.list_exams() if x["form_key"] == e["form_key"] and x["subject"])
        out = []
        for info in self.list_unassigned(exam_id):
            with db.session(self.db_path) as c:
                r = c.execute("SELECT result_json FROM scans WHERE id=?", (info["scan_id"],)).fetchone()
                have = {x[0] for x in c.execute("SELECT student_id FROM submissions WHERE exam_id=? AND page=? "
                                                "AND status!='empty'", (exam_id, info["page"]))}
            d = json.loads(r[0])
            def digits(key):
                return [{"index": x.get("index"), "state": x.get("state"), "value": x.get("value"),
                         "ambiguous": bool(x.get("ambiguous"))} for x in d.get(key) or []]
            info["id_digits"] = digits("id_digits")
            info["suggested_id"] = d.get("student_id") if _guessed(d, "id") else None    # best guess, to be confirmed by a human
            info["suggested_subject"] = d.get("subject") if _guessed(d, "subject") else None
            info["subject_digits"] = digits("subject_digits")
            info["subjects"] = subjects
            info["suggestions"] = [s for s in ids if s not in have]
            info["all_students"] = ids
            out.append(info)
        return out

    def scan_image_path(self, scan_id: int, kind: str) -> str:
        """Path of a stored image for a scan, looked up by (scan id, kind) only - never from user input."""
        col = {"original": "original_path", "warped": "warped_path", "name_crop": "name_crop_path"}.get(kind)
        if col is None:
            raise ValueError(f"unknown image kind {kind!r}")
        with db.session(self.db_path) as c:
            r = c.execute(f"SELECT {col} AS p FROM scans WHERE id=?", (scan_id,)).fetchone()
        if r is None:
            raise KeyError(f"no scan {scan_id}")
        root = os.path.realpath(os.path.join(self.data_dir, "scans")) + os.sep
        if not r["p"] or not os.path.realpath(r["p"]).startswith(root) or not os.path.isfile(r["p"]):
            raise KeyError(f"scan {scan_id} has no {kind} image")
        return r["p"]

    def scan_layout(self, scan_id: int):
        """(Layout, page) of a scan."""
        with db.session(self.db_path) as c:
            r = c.execute("SELECT s.page, e.layout_json FROM scans s JOIN exams e ON e.id=s.exam_id WHERE s.id=?",
                          (scan_id,)).fetchone()
        if r is None:
            raise KeyError(f"no scan {scan_id}")
        return Layout.from_json(r["layout_json"]), r["page"]

    def exam_scan_ids(self, exam_id: int) -> set:
        with db.session(self.db_path) as c:
            return {r[0] for r in c.execute("SELECT id FROM scans WHERE exam_id=?", (exam_id,))}

    def get_scan_exam(self, scan_id: int) -> int:
        with db.session(self.db_path) as c:
            r = c.execute("SELECT exam_id FROM scans WHERE id=?", (scan_id,)).fetchone()
        if r is None:
            raise KeyError(f"no scan {scan_id}")
        return r[0]

    def get_submission_exam(self, submission_id: int) -> int:
        with db.session(self.db_path) as c:
            r = c.execute("SELECT exam_id FROM submissions WHERE id=?", (submission_id,)).fetchone()
        if r is None:
            raise KeyError(f"no submission {submission_id}")
        return r[0]


# ---------------------------------------------------------------------- helpers
def _ranges(nums: List[int]) -> List[List[int]]:
    """[1,2,3,7] -> [[1,3],[7,7]]"""
    out: List[List[int]] = []
    for n in nums:
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return out


def _num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def _key_from_json(s) -> Dict[int, List[List[int]]]:
    return {int(q): v for q, v in json.loads(s).items()} if s else {}


def _points_from_json(s) -> Dict[int, float]:
    return {int(q): float(v) for q, v in json.loads(s).items()} if s else {}


def _norm_key(lay: Layout, key) -> Dict[int, List[List[int]]]:
    nq, nc, labels = sum(lay.questions_per_page), lay.template.choices, lay.template.choice_labels

    def idx(x) -> int:
        if isinstance(x, str):
            if x not in labels:
                raise ValueError(f"unknown choice label {x!r}")
            return labels.index(x)
        x = int(x)
        if not 0 <= x < nc:
            raise ValueError(f"choice {x} out of range")
        return x

    items = key.items() if isinstance(key, dict) else enumerate(key, 1)
    out = {}
    for q, v in items:
        q = int(q)
        if not 1 <= q <= nq:
            raise ValueError(f"question {q} out of range")
        if v is None:
            continue
        accepted = [v] if isinstance(v, (int, str)) else list(v)
        alts = []
        for a in accepted:
            alt = [idx(a)] if isinstance(a, (int, str)) else sorted({idx(x) for x in a})
            if not alt:
                raise ValueError(f"question {q}: empty accepted answer")
            if alt not in alts:
                alts.append(alt)
        if not alts:
            raise ValueError(f"question {q}: no accepted answer")
        out[q] = alts
    return out


def _grade(key, points, st) -> dict:
    per, score, mx, amb = {}, 0.0, 0.0, False
    for q, alts in sorted(key.items()):
        p = points.get(q, 1.0)
        a = st["answers"].get(q)
        ok = bool(a) and sorted(a["marked"]) in alts      # set-based: a multi row is right only for the same set
        per[q] = {"correct": ok, "points": p if ok else 0.0, "ambiguous": bool(a and a["ambiguous"])}
        score += per[q]["points"]
        mx += p
        amb = amb or per[q]["ambiguous"]
    reasons = (["ambiguous"] if amb else []) + (["conflict"] if st["conflict_pages"] else []) \
        + (["incomplete"] if st["pages_missing"] else [])
    return {"student_id": st["student_id"], "score": score, "max_score": mx, "provisional": bool(reasons),
            "reasons": reasons, "per_question": per, "status": st["status"]}
