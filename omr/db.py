"""SQLite schema + low-level access (stdlib sqlite3, WAL, foreign keys).

Design notes
* The sheet describes itself (omr.markers.Header), so no exam has to exist before scanning.  An *exam* is a
  (form_key, subject) pair - one printed form used for one subject code - created automatically by the first
  scan of that pair (or by hand to enter the answer key early).  ``subject`` NULL = "the form has a subject code
  but this scan's could not be read": a holding exam whose scans a human moves to the right subject.
  ``subject`` '' = the form has no subject code.
* ``scans``, ``assignments``, ``answer_overrides``, ``actions``, ``print_runs`` and ``scan_moves`` are
  append-only; triggers abort any UPDATE/DELETE on them.  Manual decisions are new rows, never mutations.
  Moving a scan to another subject = a copy of the scan in the target exam (``scans.moved_from``) plus a
  ``scan_moves`` row; the source then disappears from every queue (see view ``scan_owner``).
* Which student a scan belongs to is *derived*: the latest ``assignments`` row for the scan,
  else the ID read from the sheet (``scans.student_id_raw``).  See view ``scan_owner``.
* ``submissions`` is the only mutable policy state (current adoption / status / counters).
* Every service call opens its own short-lived connection; writers use BEGIN IMMEDIATE so
  concurrent ingests serialise instead of racing.
"""
from __future__ import annotations

import contextlib
import os
import sqlite3
from datetime import datetime, timezone

SCHEMA_VERSION = 3
DB_NAME = "streamomr.sqlite3"          # (the pre-redesign database was omr.sqlite3: left untouched)

SCHEMA = """
CREATE TABLE IF NOT EXISTS exams (
    id              INTEGER PRIMARY KEY,
    title           TEXT NOT NULL,
    form_key        TEXT NOT NULL,        -- omr.markers.Header.form_key ("F%04X" for legacy sheets)
    subject         TEXT,                 -- digits; '' = form without subject code; NULL = not readable (holding exam)
    layout_json     TEXT NOT NULL,
    origin          TEXT NOT NULL DEFAULT 'auto' CHECK (origin IN ('auto','manual')),
    created_at      TEXT NOT NULL,
    answer_key_json TEXT,
    points_json     TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS exams_form_subject ON exams(form_key, IFNULL(subject, '?'));

CREATE TABLE IF NOT EXISTS legacy_forms (  -- layouts of sheets printed before the self-describing header
    form_key   TEXT PRIMARY KEY,
    form_id    INTEGER NOT NULL,
    layout_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scans (
    id              INTEGER PRIMARY KEY,
    exam_id         INTEGER NOT NULL REFERENCES exams(id),
    page            INTEGER NOT NULL,
    subject_raw     TEXT,                 -- subject code as read from the sheet; NULL = unreadable, '' = form has none
    moved_from      INTEGER REFERENCES scans(id),  -- set on the copy a human moved into this exam
    student_id_raw  TEXT,                 -- as read from the sheet; NULL = unreadable/ambiguous
    result_json     TEXT NOT NULL,        -- RecognitionResult.to_dict()
    result_hash     TEXT NOT NULL,        -- form, page, student id, per-question state+marked
    content_hash    TEXT NOT NULL,        -- same without the student id (used by the policy)
    id_sig          TEXT NOT NULL DEFAULT '',  -- per-digit raw ID reading when student_id_raw is NULL
    serial          INTEGER NOT NULL DEFAULT 0,  -- printed per-copy number; 0 = none / unreadable
    sheet_key       TEXT NOT NULL DEFAULT '',  -- omr.identity.sheet_key: hash(serial|subject text|student id text)
    original_path   TEXT,
    warped_path     TEXT,
    name_crop_path  TEXT,                 -- kept for human review / future name OCR
    source          TEXT NOT NULL CHECK (source IN ('stream','upload')),
    created_at      TEXT NOT NULL,
    needs_review    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS scans_exam ON scans(exam_id, page);

CREATE TABLE IF NOT EXISTS print_runs (    -- issued serial ranges per printed form: [start_serial, start_serial+copies)
    id            INTEGER PRIMARY KEY,
    form_key      TEXT NOT NULL,
    start_serial  INTEGER NOT NULL,
    copies        INTEGER NOT NULL,
    template_json TEXT NOT NULL,         -- what to print again (omr.layout.Template as a dict)
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS print_runs_form ON print_runs(form_key);

CREATE TABLE IF NOT EXISTS scan_moves (    -- a human moved scan from_scan_id to another subject (copy to_scan_id)
    id           INTEGER PRIMARY KEY,
    from_scan_id INTEGER NOT NULL UNIQUE REFERENCES scans(id),
    to_scan_id   INTEGER NOT NULL REFERENCES scans(id),
    from_exam_id INTEGER NOT NULL REFERENCES exams(id),
    to_exam_id   INTEGER NOT NULL REFERENCES exams(id),
    note         TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS submissions (
    id              INTEGER PRIMARY KEY,
    exam_id         INTEGER NOT NULL REFERENCES exams(id),
    student_id      TEXT NOT NULL,
    page            INTEGER NOT NULL,
    adopted_scan_id INTEGER REFERENCES scans(id),
    adopted_by      TEXT NOT NULL DEFAULT 'auto' CHECK (adopted_by IN ('auto','human')),
    status          TEXT NOT NULL CHECK (status IN ('ok','conflict','empty')),
    confirm_count   INTEGER NOT NULL DEFAULT 0,
    diff_json       TEXT,                 -- conflict: per-challenger, per-question differences
    minor           INTEGER NOT NULL DEFAULT 0,
    roster_ref      TEXT,                 -- reserved: link to a student roster entry
    updated_at      TEXT NOT NULL,
    UNIQUE (exam_id, student_id, page)
);

CREATE TABLE IF NOT EXISTS assignments (   -- manual (re)assignment of a scan to a student id
    id         INTEGER PRIMARY KEY,
    scan_id    INTEGER NOT NULL REFERENCES scans(id),
    student_id TEXT,                       -- NULL = send back to the unassigned queue
    note       TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS assignments_scan ON assignments(scan_id, id);

CREATE TABLE IF NOT EXISTS answer_overrides (
    id            INTEGER PRIMARY KEY,
    submission_id INTEGER NOT NULL REFERENCES submissions(id),
    question      INTEGER NOT NULL,
    marked_json   TEXT,                    -- JSON list of choice indices; NULL = override cleared
    note          TEXT,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS overrides_sub ON answer_overrides(submission_id, question, id);

CREATE TABLE IF NOT EXISTS actions (       -- audit trail of every policy decision / manual action
    id            INTEGER PRIMARY KEY,
    exam_id       INTEGER NOT NULL REFERENCES exams(id),
    kind          TEXT NOT NULL,
    scan_id       INTEGER,
    submission_id INTEGER,
    student_id    TEXT,
    detail_json   TEXT,
    created_at    TEXT NOT NULL
);

-- a moved-away scan has no owner any more (it no longer belongs to any submission or queue); moved = 1 marks it
CREATE VIEW IF NOT EXISTS scan_owner AS
SELECT s.id AS scan_id, s.exam_id, s.page,
       CASE WHEN m.id IS NOT NULL THEN NULL
            WHEN a.id IS NULL THEN s.student_id_raw ELSE a.student_id END AS student_id,
       m.id IS NOT NULL AS moved
FROM scans s
LEFT JOIN assignments a ON a.id = (SELECT MAX(id) FROM assignments WHERE scan_id = s.id)
LEFT JOIN scan_moves m ON m.from_scan_id = s.id;
"""

for _t in ("scans", "assignments", "answer_overrides", "actions", "print_runs", "scan_moves"):
    SCHEMA += f"""
CREATE TRIGGER IF NOT EXISTS {_t}_no_update BEFORE UPDATE ON {_t}
BEGIN SELECT RAISE(ABORT, '{_t} is append-only'); END;
CREATE TRIGGER IF NOT EXISTS {_t}_no_delete BEFORE DELETE ON {_t}
BEGIN SELECT RAISE(ABORT, '{_t} is append-only'); END;
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)  # autocommit; we manage BEGIN
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = connect(path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")  # persistent in the db file
        conn.executescript(SCHEMA)
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    finally:
        conn.close()


@contextlib.contextmanager
def session(path: str, write: bool = False):
    """Open a connection; with write=True run inside one BEGIN IMMEDIATE transaction."""
    conn = connect(path)
    try:
        if write:
            conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        else:
            if conn.in_transaction:
                conn.execute("COMMIT")
    finally:
        conn.close()
