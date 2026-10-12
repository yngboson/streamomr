"""Physical-sheet identity, shared by the stream state machine and the database policy.

    sheet_key = sha1(f"{serial:04d}|{subject_text}|{student_id_text}")

``serial`` is the printed per-copy number (0 = none, e.g. a photocopied master -> "0000", so the key
degrades to "identity = subject + student ID" without any special case).  ``student_id_text`` is the read ID, or
the raw per-digit reading with '?' for unread digits when the ID could not be read (deterministic).
``subject_text`` is the marked subject code, read the same way ('?' per unread digit; "" for a form without a
subject code).  The form itself (``form_key``) and the page are NOT part of the key: callers pair it with them.
Works on a RecognitionResult or on its ``to_dict()``.
"""
from __future__ import annotations

import hashlib
from typing import Optional


def _get(r, name, default=None):
    return r.get(name, default) if isinstance(r, dict) else getattr(r, name, default)


def serial_of(r) -> int:
    try:
        return int(_get(r, "serial") or 0)
    except (TypeError, ValueError):
        return 0


def _digits_text(value, digits) -> str:
    if value is not None and str(value).strip():
        return str(value).strip()
    out = []
    for d in digits or []:
        v = d.get("value") if isinstance(d, dict) else getattr(d, "value", None)
        out.append("?" if v is None else str(v))
    return "".join(out)


def student_id_text(r) -> str:
    return _digits_text(_get(r, "student_id"), _get(r, "id_digits"))


def subject_text(r) -> str:
    return _digits_text(_get(r, "subject"), _get(r, "subject_digits"))


def make_sheet_key(serial: int, id_text: str, subject: str = "") -> str:
    return hashlib.sha1(f"{int(serial or 0):04d}|{subject}|{id_text}".encode()).hexdigest()[:16]


def sheet_key(r) -> str:
    return make_sheet_key(serial_of(r), student_id_text(r), subject_text(r))


def format_serial(serial: Optional[int]) -> str:
    return f"No. {int(serial):04d}" if serial else ""
