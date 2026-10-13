"""Revision of a committed sheet: upgrade-only merge of a later (fused) reading into the committed one.

After a commit the sheet usually stays in view for a while and its frames keep fusing (omr/fusion.Track).  `merge`
compares the fresh fused reading `now` with the committed one `base`, row by row (answer rows, the ID, the subject):

  * base row doubtful (ambiguous / guessed / unreadable) and now confident  -> adopted ("changed")
  * base row confident and now confident but DIFFERENT                      -> NOT adopted; reported as a contradiction
  * anything else (still doubtful, or a confident row turning doubtful)     -> base kept

so a confident row is never flipped quietly; a contradiction adds the soft-hard flag "contradiction" (needs review).
Pure.  Works on RecognitionResult objects; `now` supplies images / geometry, the rows come from `base` or `now`.
"""
from __future__ import annotations

import copy
from typing import List, Tuple

CONTRADICTION_FLAG = "contradiction"
_ID_FLAGS = ("id_blank", "id_unreadable", "id_guessed")
_SUBJECT_FLAGS = ("subject_blank", "subject_unreadable", "subject_guessed")


def answer_confident(a) -> bool:
    return not (a.ambiguous or a.guessed)


def _val(a):
    return (a.state, tuple(a.marked))


def id_confident(r) -> bool:
    return r.student_id is not None and "id_guessed" not in r.flags


def subject_confident(r) -> bool:
    return r.subject is not None and "subject_guessed" not in r.flags


def doubtful(r) -> bool:
    """Anything left that a later frame could still resolve?"""
    if r.id_digits and not id_confident(r):
        return True
    if r.subject_digits and not subject_confident(r):
        return True
    return any(not answer_confident(a) for a in r.answers)


def merge(base, now) -> Tuple[object, List[str], List[str]]:
    """-> (merged result, changed rows, contradicted rows); rows are "q<number>", "id", "subject"."""
    changed: List[str] = []
    contra: List[str] = []
    answers = []
    by_num = {a.number: a for a in now.answers}
    for b in base.answers:
        n = by_num.get(b.number)
        keep = b
        if n is not None:
            if answer_confident(b):
                if answer_confident(n) and _val(n) != _val(b):
                    contra.append(f"q{b.number}")
            elif answer_confident(n):
                keep = n
                changed.append(f"q{b.number}")
        answers.append(keep)
    out = copy.copy(now)
    out.answers = answers
    flags = [f for f in now.flags if f not in _ID_FLAGS + _SUBJECT_FLAGS + ("guessed_answers", CONTRADICTION_FLAG)]
    for name, conf, digits_attr, val_attr, fl in (("id", id_confident, "id_digits", "student_id", _ID_FLAGS),
                                                  ("subject", subject_confident, "subject_digits", "subject", _SUBJECT_FLAGS)):
        if not getattr(base, digits_attr):
            src = now
        elif conf(base):
            src = base
            if conf(now) and getattr(now, val_attr) != getattr(base, val_attr):
                contra.append(name)
        elif conf(now):
            src = now
            changed.append(name)
        else:
            src = base
        setattr(out, digits_attr, getattr(src, digits_attr))
        setattr(out, val_attr, getattr(src, val_attr))
        flags += [f for f in src.flags if f in fl]
    if any(a.guessed for a in answers):
        flags.append("guessed_answers")
    if contra or CONTRADICTION_FLAG in base.flags:
        flags.append(CONTRADICTION_FLAG)
    out.flags = flags
    return out, changed, contra
