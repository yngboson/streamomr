"""Multi-frame fusion: one physical sheet seen in many frames -> one set of decisions.

A *track* is a run of frames believed to be the same physical sheet.  Per-bubble raw measurements (fill and
strength, exposed by `recognize` as `result.raw`) are fused with a per-bubble median over the track's recent
frames, and the ONE decision function (`omr.recognize.decide_page`) runs on the fused arrays - the rules are not
duplicated here.  Different frames typically leave different digits unreadable (blur, a light mark, a shadow);
the median resolves what no single frame could.

A frame joins the track only if it is compatible with it, otherwise the track ends and a new one starts with
that frame.  Two sheets are never blended:
  * another form (form_key) / page, or another (readable) serial;
  * a jump of the sheet quad between consecutive frames (moved or replaced);
  * a confident contradiction: a digit (student ID or subject code) / answer the fused track reads confidently
    that this frame reads confidently as something else (a different readable subject = another sheet), or >= CONTRADICT_BUBBLES bubbles whose fill swings by >= CONTRADICT_FILL
    against the fused value (a different hand-drawn mark pattern, even when no single decision is confident).
Frames whose bubble alignment could not be verified (`raw.verified` False) never contribute.
Pure: no I/O, no clock.
"""
from __future__ import annotations

import copy
import warnings
from collections import Counter, deque
from typing import Optional, Tuple

import numpy as np

from .recognize import PageRaw, decide_page

WINDOW = 12              # most recent frames fused (the median adapts, memory is bounded)
CONTRADICT_FILL = 0.6    # per-bubble fill swing that counts as a contradiction ...
CONTRADICT_BUBBLES = 2   # ... when at least this many bubbles swing
JUMP_FRACTION = 0.2      # sheet-quad centre jump (fraction of sheet size) between consecutive frames
SCALE_JUMP = 0.35        # relative change of the sheet size between consecutive frames
ID_FLAGS = ("id_blank", "id_unreadable", "alignment_unverified", "subject_blank", "subject_unreadable",
            "id_guessed", "subject_guessed", "guessed_answers")


def sheet_quad(r):
    """(cx, cy, size) of the projected page rectangle in image px (independent of which markers were visible)."""
    if r.homography is None:
        return None
    try:
        q = r.image_points([[0, 0], [210, 0], [210, 297], [0, 297]])
    except Exception:
        return None
    return (float(q[:, 0].mean()), float(q[:, 1].mean()),
            float(max(np.ptp(q[:, 0]), np.ptp(q[:, 1]))))


def _sig(answers, student_id, subject):
    return (tuple((a.number, a.state, tuple(a.marked), a.guess) for a in answers), student_id, subject)


def trustworthy(r) -> bool:
    """A frame can feed a track: raw measurements present, verified, and consistent with the frame's decisions
    (a result whose decisions were edited after the fact is not fused)."""
    raw = r.raw
    if raw is None or not raw.verified:
        return False
    d = decide_page(raw)
    return _sig(d.answers, d.student_id, d.subject) == _sig(r.answers, r.student_id, r.subject)


class Track:
    def __init__(self, no: int, r, t: float):
        self.no = no
        self.form_key, self.page = r.form_key, r.page
        self.frames: deque = deque(maxlen=WINDOW)
        self.serials: Counter = Counter()
        self.t0 = self.t_last = t
        self.n = 0
        self.quad = None
        self.committed = False
        self.held_since: Optional[float] = None    # set while a stable-but-unreadable-ID result is being held
        self.best = None                            # (sharpness, result, payload, t) of the sharpest frame
        self._dec = None
        self._raw: Optional[PageRaw] = None
        self.add(r, t, None)

    # ------------------------------------------------------------------ fusion
    def fused_raw(self) -> PageRaw:
        if self._raw is None:
            fr = list(self.frames)
            med = lambda f: np.median(np.stack([getattr(x, f) for x in fr]), axis=0)
            r0 = fr[-1]

            def nanmed(f):    # disc darkness (relative rule): NaN = unknowable bubble; a frame without it does not vote; None when no frame has it
                have = [getattr(x, f) for x in fr if getattr(x, f) is not None]
                if not have:
                    return None
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    return np.nanmedian(np.stack(have), axis=0)

            subj = {}
            if r0.FS is not None:    # same form on every frame of a track (form_key), so FS is present on all or none
                subj = dict(FS=med("FS"), SS=med("SS"), MS=nanmed("MS"))
            self._raw = PageRaw(r0.numbers, r0.labels, r0.digit_index, med("FA"), med("SA"), med("FI"), med("SI"),
                                float(np.median([x.ink for x in fr])), float(np.median([x.min_strength for x in fr])), True,
                                ink_source="default" if any(x.ink_source == "default" for x in fr) else "auto",
                                ink_separation=float(np.median([x.ink_separation for x in fr])),
                                MA=nanmed("MA"), MI=nanmed("MI"), **subj)
        return self._raw

    def decisions(self):
        if self._dec is None:
            self._dec = decide_page(self.fused_raw())
        return self._dec

    def serial(self) -> Optional[int]:
        return self.serials.most_common(1)[0][0] if self.serials else None

    # ------------------------------------------------------------------ membership
    def incompatible(self, r) -> Optional[str]:
        if (r.form_key, r.page) != (self.form_key, self.page):
            return "other_sheet"
        if r.serial is not None and self.serial() is not None and r.serial != self.serial():
            return "other_serial"
        q = sheet_quad(r)
        if q is not None and self.quad is not None:
            (x0, y0, s0), (x1, y1, s1) = self.quad, q
            if ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 > JUMP_FRACTION * max(s0, s1) or abs(s1 - s0) > SCALE_JUMP * max(s0, s1):
                return "moved"
        if self.n >= 2 and self._contradicts(r):
            return "contradiction"
        return None

    def _contradicts(self, r) -> bool:
        dec = self.decisions()
        for a, b in zip(dec.answers, r.answers):
            if not a.ambiguous and not b.ambiguous and (a.state, a.marked) != (b.state, b.marked):   # (a guessed row is ambiguous: never a contradiction)
                return True
        for a, b in list(zip(dec.id_digits, r.id_digits)) + list(zip(dec.subject_digits, r.subject_digits)):
            if a.value is not None and b.value is not None and not (a.guessed or b.guessed) and a.value != b.value:
                return True
        fr, raw = self.fused_raw(), r.raw
        swing = int((np.abs(fr.FA - raw.FA) >= CONTRADICT_FILL).sum() + (np.abs(fr.FI - raw.FI) >= CONTRADICT_FILL).sum())
        if fr.FS is not None and raw.FS is not None:
            swing += int((np.abs(fr.FS - raw.FS) >= CONTRADICT_FILL).sum())
        return swing >= CONTRADICT_BUBBLES

    def add(self, r, t: float, payload) -> None:
        self.frames.append(r.raw)
        self.n += 1
        self.t_last = t
        if r.serial is not None:
            self.serials[r.serial] += 1
        self.quad = sheet_quad(r) or self.quad
        self._raw = self._dec = None
        sharp = float((r.quality or {}).get("sharpness", 0.0))
        if self.best is None or sharp > self.best[0]:
            self.best = (sharp, r, payload, t)

    # ------------------------------------------------------------------ result
    def apply(self, r):
        """Copy of frame result `r` carrying the track's fused decisions."""
        dec = self.decisions()
        out = copy.copy(r)
        out.answers, out.id_digits, out.student_id, out.calibration = dec.answers, dec.id_digits, dec.student_id, dec.calibration
        out.subject, out.subject_digits = dec.subject, dec.subject_digits
        out.flags = [f for f in r.flags if f not in ID_FLAGS] + dec.flags
        s = self.serial()
        if s is not None:
            out.serial = s
            out.flags = [f for f in out.flags if f != "serial_unreadable"]
        out.fused_frames = len(self.frames)
        return out

    def id_unreadable(self) -> bool:
        """No confident ID yet: unreadable, or only a best guess (more frames may still settle it)."""
        f = self.decisions().flags
        return "id_unreadable" in f or "id_guessed" in f

    def subject_unreadable(self) -> bool:
        f = self.decisions().flags
        return "subject_unreadable" in f or "subject_guessed" in f
