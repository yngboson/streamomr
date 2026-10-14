"""Frame-by-frame "barcode scanner" state machine.  Pure: no I/O, no clock, no threads.

    session = StreamSession()        # self-describing sheets: no database or layout resolver is needed
    event = session.feed(recognition_result, t_seconds, payload=frame)

event = {"state", "hint", "overlay", "frame_index", "commit"?}

States
  idle       no sheet in view
  detecting  a sheet is in view but nothing is committed for it yet
  committed  this very frame completed a commit (event["commit"] is set)
  cooldown   the committed sheet is still in view; nothing more is committed for it

Rules
  * A commit needs K consecutive `ok` frames with the same decision key
    (form_key, page, subject code, student id, per-question state + marked).  One frame never commits.
  * After a commit the same sheet cannot commit again until it has been gone for M consecutive frames
    (no_sheet / partial), after which the session is idle again.
  * Cooldown, sheet held continuously (no interruption since the commit): strict.  Only another form /
    page, another readable student id or subject code, or answers differing in more than SWAP_FRACTION of the questions
    count as a different sheet; smaller changes (a borderline bubble flipping) never commit a second version.
  * Cooldown after an INTERRUPTION since the commit: a run of >= INTERRUPT_FRAMES (2) consecutive non-ok
    frames of any status, or a jump of the marker quad between consecutive ok frames (centre moved by more
    than JUMP_FRACTION of the sheet size, or its size changed by more than SCALE_JUMP) means the sheet was
    moved or replaced.  The next K-stable key then commits if it differs from the committed key in any way
    (a missed sheet is worse than an extra commit; the DB dedupes identical ones).  An identical key is the
    same sheet put back: no commit, and the interruption is considered settled.
  * Sheet identity: `sheet_key` = hash(serial|subject text|student-id text) (omr/identity.py).  A different key is another
    sheet (commit once K-stable, in cooldown too); the same key and page is the same sheet.  Serial 0 (a
    photocopied master) is just "0000", so identity degrades to the student ID.  Unreadable-ID digits can
    flicker and occasionally cost an extra commit; the DB dedupes / queues it.
  * Optional filters (`allowed_forms`: form_keys, `allowed_subjects`: subject codes) make a scanner ignore other
    forms / subjects.  A frame whose subject is not readable is NOT filtered out (it may be the wanted one).
  * Missed-sheet check: a (form_key, page, sheet_key) seen in >= MISSED_MIN_FRAMES ok frames that was never
    committed in this session, and is no longer the sheet in view, is reported by `missed()`.
  * Presence: a frame with any marker visible (status != no_sheet, i.e. also `partial`, `bad_*`, `inconsistent`)
    means a sheet is in view.  Consecutive presence forms an episode that ends after M consecutive no_sheet
    frames.  An episode of >= MISSED_MIN_FRAMES frames that never produced a commit is reported by
    `unread_episodes()`: a sheet that was shown but never read (covered marker, blur, ...), also when it never
    produced a single `ok` frame.  Events carry `present` and `unread` (the current episode is long enough
    and still without a commit) for the live hint.
  * Fusion (omr/fusion.py): frames of one physical sheet form a track; their per-bubble measurements are fused
    (median) and decided by the same decision function, so a digit unreadable in every single frame can still
    resolve.  All rules above (K-stable key, cooldown, sheet identity) operate on the FUSED result.  A track
    never commits twice.  A stable result whose ID or subject code is still unreadable is held up to HOLD_SECONDS while the sheet
    stays in view; it is committed unreadable only when the hold expires or the track ends (>= 2 non-contributing
    frames, a contradicting / different sheet, `finish()`).  The commit carries the fused decisions with the
    sharpest frame's images and `result.fused_frames`.
  * Revision (omr/revision.py): frames of the SAME track keep fusing in cooldown.  While the committed result still has a
    doubtful / guessed / unreadable row (at most REVISE_FRAMES frames / REVISE_SECONDS), a fused reading that makes such a
    row confident emits event["revision"] (a commit-like payload + `changed` rows + `severity`/`prev_severity`).  UPGRADE-ONLY:
    a row that was confident at commit is never changed; fused evidence that contradicts it is reported once as
    `contradiction` rows (flag "contradiction", needs review) with the row left as committed.  A different sheet ends
    the track as before.  Each revision becomes the new committed result.
  * The commit carries the sharpest of the K agreeing frames (result + the caller's payload).
    Fusing the K frames' scores into one result would go in `_pick`.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .fusion import Track, trustworthy
from .identity import sheet_key as _sheet_key, serial_of, severity
from .revision import merge as _merge, doubtful as _doubtful

IDLE, DETECTING, COMMITTED, COOLDOWN = "idle", "detecting", "committed", "cooldown"

DEFAULT_K = 3
DEFAULT_M = 5
DEFAULT_FAST_K = 2           # frames needed instead of k when the fused result has nothing left to doubt (see `_confident`)
DEFAULT_HINT_AFTER = 2.0     # seconds a sheet may be in view without stabilising before a hint appears
INTERRUPT_FRAMES = 2         # consecutive non-ok frames that count as "the sheet was disturbed"
JUMP_FRACTION = 0.2          # marker-quad centre jump, as a fraction of the sheet size
SCALE_JUMP = 0.35            # relative change of the sheet size between consecutive frames
SWAP_FRACTION = 0.3          # share of differing questions that marks a different physical sheet
LOW_RES_PX_PER_MM = 2.2      # source px per mm below which "move closer" is the likely fix
LOW_SHARPNESS = 40.0         # Laplacian variance below which "hold still" is the likely fix
DEFAULT_BUBBLE_MM = 4.0

HOLD_SECONDS = 1.5           # an unreadable ID is given this long (sheet still in view) to resolve through fusion
TRACK_GAP = INTERRUPT_FRAMES  # consecutive non-contributing frames that end a track

MISSED_MIN_FRAMES = 2        # ok frames a sheet must be seen in to count as "seen" for the missed-sheet check

REVISE_FRAMES = 40           # ok frames after a commit during which the same sheet's evidence may still revise it ...
REVISE_SECONDS = 6.0         # ... or this many seconds, whichever ends first (or earlier: nothing doubtful is left)

ABSENT = ("no_sheet", "partial")

# overlay bubble codes
B_EMPTY, B_MARKED, B_MULTI, B_AMBIGUOUS = 0, 1, 2, 3


def _confident(r) -> bool:
    """Nothing left to doubt: ID and subject read (not guessed), no review flag, every answer single and unambiguous."""
    return (r.student_id is not None and r.subject is not None and not r.needs_review
            and not r.id_guessed and not r.subject_guessed)


def decision_key(r) -> tuple:
    return (r.form_key, r.page, r.student_id,
            tuple((a.number, a.state, tuple(a.marked)) for a in r.answers), _sheet_key(r), r.subject)


def identity(r) -> tuple:
    return (r.form_key, r.page, r.student_id, r.subject)


def sheet_id(r) -> tuple:
    """(form_key, page, sheet_key): the physical-sheet identity used by the missed-sheet check."""
    return (r.form_key, r.page, _sheet_key(r))


def _answer_distance(a: tuple, b: tuple) -> float:
    da = {n: (s, m) for n, s, m in a}
    db = {n: (s, m) for n, s, m in b}
    keys = set(da) | set(db)
    if not keys:
        return 0.0
    return sum(1 for k in keys if da.get(k) != db.get(k)) / len(keys)


def _quad(r):
    """(cx, cy, size) of the marker quad in image px, or None."""
    pts = [p for q in r.marker_corners.values() for p in q]
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(max(xs) - min(xs), max(ys) - min(ys)))


def is_other_sheet(committed_key: tuple, key: tuple) -> bool:
    """Is `key` a different physical sheet than the one already committed (both ok-frame keys)?"""
    f0, p0, s0, a0, k0, u0 = committed_key
    f1, p1, s1, a1, k1, u1 = key
    if f0 != f1 or p0 != p1 or k0 != k1:
        return True
    if s0 is not None and s1 is not None and s0 != s1:
        return True
    if u0 is not None and u1 is not None and u0 != u1:
        return True
    return _answer_distance(a0, a1) > SWAP_FRACTION


class StreamSession:
    def __init__(self, k: int = DEFAULT_K, m: int = DEFAULT_M, hint_after: float = DEFAULT_HINT_AFTER,
                 allowed_forms: Optional[Iterable[str]] = None, allowed_subjects: Optional[Iterable[str]] = None,
                 resolve_layout: Optional[Callable[[int], Any]] = None, fuse: bool = True, hold_s: float = HOLD_SECONDS,
                 fast_k: Optional[int] = DEFAULT_FAST_K):
        self.fuse, self.hold_s = fuse, hold_s
        self.fast_k = fast_k            # frames needed instead of k when the result is fully confident (None = always k)
        if fast_k is not None and fast_k < 2:
            raise ValueError("fast_k must be >= 2: a single frame must never commit")
        if k < 2:
            raise ValueError("k must be >= 2: a single frame must never commit")
        self.k, self.m, self.hint_after = k, m, hint_after
        self.allowed_forms = None if allowed_forms is None else set(allowed_forms)
        self.allowed_subjects = None if allowed_subjects is None else set(allowed_subjects)
        self.resolve_layout = resolve_layout       # legacy sheets only: form_id -> Layout (for the overlay bubble size)
        self.frame_index = -1
        self.reset_tracking()
        self.reset()

    def reset_tracking(self) -> None:
        """Forget which sheets were seen / committed (start of a new scanning session)."""
        self._seen: Dict[Any, dict] = {}         # track number (or sheet_id for unfused frames) -> info
        self._committed_ids: set = set()
        self._current: Any = None                # _seen key of the sheet in view right now
        self._track: Optional[Track] = None
        self._track_no = 0
        self._tgap = 0                           # consecutive frames that did not contribute to the track
        self._ep: Optional[dict] = None           # presence episode in progress
        self._episodes: List[dict] = []           # finished episodes
        self._gap = 0                             # consecutive no_sheet frames

    def missed(self, final: bool = False) -> List[dict]:
        """Sheets (tracks) seen in >= MISSED_MIN_FRAMES ok frames but never committed (the sheet in view is
        excluded unless ``final``)."""
        out = []
        for key, info in self._seen.items():
            if info["committed"] or info["sid"] in self._committed_ids or info["frames"] < MISSED_MIN_FRAMES:
                continue
            if key == self._current and not final:
                continue
            m = {"form_key": info["sid"][0], "page": info["sid"][1], "serial": info["serial"],
                 "student_id": info["student_id"], "frames": info["frames"]}
            if info.get("possibly_same_as"):
                m["possibly_same_as"] = info["possibly_same_as"]
            out.append(m)
        return sorted(out, key=lambda m: (m["serial"] or 0, m["page"] if m["page"] is not None else -1))

    def _track_presence(self, status: str, t: float, hint_status: str) -> None:
        if status == "no_sheet":
            self._gap += 1
            if self._ep is not None and self._gap >= self.m:
                self._episodes.append(self._ep)
                self._ep = None
            return
        self._gap = 0
        if self._ep is None:
            self._ep = {"t0": t, "t1": t, "frames": 0, "ok": 0, "commits": 0, "statuses": {}, "page": None}
        e = self._ep
        e["t1"] = t
        e["frames"] += 1
        e["ok"] += status == "ok"
        e["statuses"][status] = e["statuses"].get(status, 0) + 1

    def unread_episodes(self, final: bool = False) -> List[dict]:
        """Sheet-present stretches (any marker visible, incl. `partial`) that never produced a commit.
        The one in progress is included only with ``final`` (it may still be read)."""
        eps = list(self._episodes) + ([self._ep] if final and self._ep is not None else [])
        return [dict(e) for e in eps if e["commits"] == 0 and e["frames"] >= MISSED_MIN_FRAMES]

    def reset(self) -> None:
        self.state = IDLE
        self._absent = 0
        self._run_key: Optional[tuple] = None
        self._run_n = 0
        self._best: Optional[tuple] = None          # (sharpness, result, payload, t)
        self._committed_key: Optional[tuple] = None
        self._committed_res = None                   # the committed (fused) result, for identity-by-evidence
        self._since: Optional[float] = None          # when the current uncommitted sheet came into view
        self._nonok = 0                              # consecutive non-ok frames
        self._interrupted = False                    # sheet disturbed since the last commit
        self._prev_quad: Optional[tuple] = None      # (cx, cy, size) of the previous ok frame
        self._rev: Optional[dict] = None             # revision watch of the committed track (see _revise)

    # ------------------------------------------------------------------ main entry
    def feed(self, r, t: float, payload: Any = None) -> dict:
        self.frame_index += 1
        commit = None
        hint = ""
        self._revision = None
        status = r.status
        r0 = r
        pre = None                    # commit produced by a track that ended with this frame
        if status == "ok":
            self._nonok = 0
        else:
            self._nonok += 1
            self._prev_quad = None
            if self._nonok >= INTERRUPT_FRAMES and self._committed_key is not None:
                self._interrupted = True
        # ---- track bookkeeping (fusion)
        if status == "ok" and self.fuse and trustworthy(r):
            self._tgap = 0
            why = self._track.incompatible(r) if self._track is not None else None
            if why is not None:
                pre = self._end_track()
            if self._track is None:
                self._track_no += 1
                self._track = Track(self._track_no, r, t)
                self._track.best = (float((r.quality or {}).get("sharpness", 0.0)), r, payload, t)
            else:
                self._track.add(r, t, payload)
            r = self._track.apply(r)
        else:
            self._tgap += 1
            if self._tgap >= TRACK_GAP and self._track is not None:
                pre = self._end_track()
        self._track_presence(status, t, status)

        if pre is not None:
            commit = pre
        if status in ABSENT:
            self._run_key, self._run_n, self._best = None, 0, None
            self._absent += 1
            if self._absent >= self.m:
                self._current = None
                self.reset()
            elif self.state == COMMITTED:
                self.state = COOLDOWN
            hint = self._absent_hint(r)
        else:
            self._absent = 0
            if self._since is None and self.state in (IDLE, DETECTING):
                self._since = t
            if self.state == IDLE:
                self.state = DETECTING
            elif self.state == COMMITTED:
                self.state = COOLDOWN
            if status == "ok":
                hint, c2 = self._feed_ok(r, t, payload)
                commit = commit or c2
            else:
                self._run_key, self._run_n, self._best = None, 0, None
                hint = self._bad_hint(r, t)

        if commit is not None:
            self.state = COMMITTED
            hint = "Scanned"
            if self._ep is not None:
                self._ep["commits"] += 1
        elif self.state == COOLDOWN and not hint:
            hint = "Scanned - show the next sheet"
        elif self.state == IDLE and not hint:
            hint = "Show a sheet"
        ev = {"state": self.state, "hint": hint, "overlay": self.overlay(r), "frame_index": self.frame_index,
              "present": status != "no_sheet",
              "possibly_same": any(i.get("possibly_same_as") and not i["committed"] for i in self._seen.values()),
              "unread": bool(self._ep is not None and self._ep["commits"] == 0 and self._ep["frames"] >= MISSED_MIN_FRAMES)}
        if commit is not None:
            ev["commit"] = commit
        elif self._revision is not None:
            ev["revision"] = self._revision
        return ev

    def finish(self, t: float = 0.0) -> Optional[dict]:
        """End of stream: a held (stable, ID still unreadable) track is committed as it is."""
        c = self._end_track()
        if c is not None:
            self.state = COMMITTED
        return c

    # ------------------------------------------------------------------ tracks
    def _end_track(self) -> Optional[dict]:
        """Close the current track; commit it if it was held (stable but not yet committed)."""
        tr, self._track = self._track, None
        self._tgap = 0
        if tr is None or tr.committed or tr.held_since is None:
            return None
        sharp, r0, payload, t = tr.best
        final = tr.apply(r0)
        key = decision_key(final)
        if self._committed_key is not None and self.state == COOLDOWN:      # same re-commit rules as for a live frame
            if self._interrupted:
                if key == self._committed_key:
                    return None
                if self._same_by_evidence(final):
                    self._note_possible(("t", tr.no), final)
                    return None
            elif not is_other_sheet(self._committed_key, key):
                return None
        return self._do_commit(final, payload, t, key, tr.n, tr)

    def _do_commit(self, final, payload, t, key, frames, track=None) -> dict:
        commit = {"result": final, "payload": payload, "key": key, "identity": identity(final),
                  "t": t, "frames": frames, "frame_index": self.frame_index, "fused_frames": final.fused_frames}
        self._committed_key = key
        self._committed_res = final
        self._rev = None
        if track is not None and self.fuse and _doubtful(final):
            self._rev = {"track": track, "base": final, "t0": t, "n": 0, "contra": set(), "count": 0}
        sid = sheet_id(final)
        self._committed_ids.add(sid)
        if track is not None:
            track.committed = True
            for k, info in self._seen.items():
                if k == ("t", track.no):
                    info["committed"] = True
        self._run_key, self._run_n, self._best = None, 0, None
        self._since = None
        self._interrupted = False
        return commit

    # ------------------------------------------------------------------ ok frames
    def _feed_ok(self, r, t, payload):
        if self.allowed_forms is not None and r.form_key not in self.allowed_forms:
            self._run_key, self._run_n, self._best = None, 0, None
            return "Sheet is another form", None
        if (self.allowed_subjects is not None and r.subject is not None and not r.subject_guessed
                and r.subject not in self.allowed_subjects):   # (a guessed subject never filters a sheet out)
            self._run_key, self._run_n, self._best = None, 0, None
            return "Sheet is another subject", None
        key = decision_key(r)
        sid = sheet_id(r)
        tr = self._track if r.fused_frames else None
        skey = ("t", tr.no) if tr is not None else sid
        self._current = skey
        seen = self._seen.setdefault(skey, {"frames": 0, "committed": False})
        seen.update(sid=sid, serial=serial_of(r), student_id=r.student_id, subject=r.subject)
        seen["frames"] += 1
        q = _quad(r)
        if q is not None and self._prev_quad is not None and self._committed_key is not None:
            (x0, y0, s0), (x1, y1, s1) = self._prev_quad, q
            if ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 > JUMP_FRACTION * max(s0, s1) or abs(s1 - s0) > SCALE_JUMP * max(s0, s1):
                self._interrupted = True
        self._prev_quad = q
        if key == self._run_key:
            self._run_n += 1
        else:
            self._run_key, self._run_n, self._best = key, 1, None
        sharp = float((r.quality or {}).get("sharpness", 0.0))
        if self._best is None or sharp > self._best[0]:
            self._best = (sharp, r, payload, t)
        if tr is not None and tr.committed:
            self._revise(tr, r, t)                  # one commit per track; later evidence may only revise it
            return "", None
        committed_here = self.state == COOLDOWN and self._committed_key is not None
        if committed_here:
            if self._interrupted:
                same = key != self._committed_key and self._same_by_evidence(r)
                if same:
                    self._note_possible(skey, r)
                if key == self._committed_key or same:   # same sheet put back / re-seen
                    if self._run_n >= self.k:
                        self._interrupted = False
                    return "", None
            elif not is_other_sheet(self._committed_key, key):
                return "", None                     # same sheet still in view (maybe flickering)
        if self._run_n < (self.fast_k if self.fast_k is not None and _confident(r) else self.k):
            return self._unstable_hint(r, t) if self.state != COOLDOWN else "", None
        if tr is not None and (tr.id_unreadable() or tr.subject_unreadable()):   # stable, but may still resolve: hold while in view
            if tr.held_since is None:
                tr.held_since = t
            if t - tr.held_since < self.hold_s:
                return "Reading the student ID...", None
        best = self._pick()
        commit = self._do_commit(self._finalize(best[1]), best[2], best[3], key, self._run_n, tr)
        return "", commit

    def _revise(self, tr, r, t) -> None:
        """Same track, already committed: upgrade the committed result from the newly fused frame `r` (see module doc)."""
        rv = self._rev
        if rv is None or rv["track"] is not tr:
            return
        rv["n"] += 1
        if rv["n"] > REVISE_FRAMES or t - rv["t0"] > REVISE_SECONDS:
            self._rev = None
            return
        base = rv["base"]
        _, changed, contra = _merge(base, r)
        new_contra = [c for c in contra if c not in rv["contra"]]
        if not changed and not new_contra:
            return
        merged, changed, contra = _merge(base, tr.apply(tr.best[1]))      # images / geometry of the sharpest frame so far
        rv["contra"].update(contra)
        rv["count"] += 1
        key = decision_key(merged)
        ev = {"result": merged, "payload": tr.best[2], "key": key, "identity": identity(merged), "t": t, "frames": tr.n,
              "frame_index": self.frame_index, "fused_frames": merged.fused_frames, "changed": changed,
              "contradiction": sorted(rv["contra"]), "new_contradiction": new_contra, "revision": rv["count"],
              "severity": severity(merged), "prev_severity": severity(base), "prev_key": self._committed_key}
        rv["base"] = merged
        self._committed_key, self._committed_res = key, merged
        self._committed_ids.add(sheet_id(merged))
        if not _doubtful(merged):
            self._rev = None
        self._revision = ev

    def _same_by_evidence(self, r) -> bool:
        """After an interruption: a re-sighting whose ID is only partly readable but whose readable digits
        (at least half) all agree with the committed sheet's ID, with IDENTICAL decisions on every answer and
        equal serials (when either side has one), is treated as that sheet put back.  Never silent: the caller
        records it as "possibly the same sheet" in the missed report.  No readable digit = no evidence: it
        commits (a missed sheet is worse than an extra commit)."""
        c = self._committed_res
        if (c is None or c.student_id is None or c.id_guessed or (r.student_id is not None and not r.id_guessed)
                or (c.form_key, c.page) != (r.form_key, r.page)):
            return False
        if r.subject is not None and not r.subject_guessed and r.subject != c.subject:
            return False                            # a readable, different subject code is another sheet
        if (r.serial is not None or c.serial is not None) and r.serial != c.serial:
            return False
        read = [(d.index, d.value) for d in r.id_digits if d.value is not None and not d.guessed]   # a guess is no evidence
        if len(read) == len(r.id_digits):
            return False                            # (inconsistent: an ID that is fully read would not be None)
        if len(read) * 2 < len(c.id_digits) or any(c.student_id[i] != str(v) for i, v in read if i < len(c.student_id)):
            return False
        return [(a.number, a.state, a.marked) for a in c.answers] == [(a.number, a.state, a.marked) for a in r.answers]

    def _note_possible(self, skey, r) -> None:
        """A commit was suppressed by `_same_by_evidence`: keep it visible (missed report + event)."""
        info = self._seen.get(skey)
        if info is not None:
            info["possibly_same_as"] = self._committed_res.student_id
            info["student_id"] = r.student_id

    def _finalize(self, r):
        """The frame result chosen for the commit, carrying the track's latest fused decisions."""
        if r.fused_frames and self._track is not None:
            return self._track.apply(r)
        return r

    def _pick(self):
        """Which frame to ingest.  Seam: fuse the agreeing frames' scores here instead of picking one."""
        return self._best

    # ------------------------------------------------------------------ hints
    def _stuck(self, t) -> bool:
        return self._since is not None and (t - self._since) >= self.hint_after

    def _unstable_hint(self, r, t) -> str:
        if not self._stuck(t):
            return "Reading..."
        q = r.quality or {}
        if r.student_id is None or r.id_guessed:
            return "Student ID not readable - hold still, check the ID marks"
        if r.subject is None or r.subject_guessed:
            return "Subject code not readable - hold still, check the subject marks"
        if q.get("source_px_per_mm", 99) < LOW_RES_PX_PER_MM:
            return "Move closer"
        if q.get("sharpness", 1e9) < LOW_SHARPNESS:
            return "Hold still (blurry)"
        return "Hold still - reading keeps changing (glare or shadow?)"

    def _bad_hint(self, r, t) -> str:
        s = r.status
        if s == "unknown_form":
            return "Unknown sheet (not created in this system)"
        if not self._stuck(t):
            return "Reading..."
        if s == "bad_geometry":
            return "Flatten the sheet and keep it square to the camera"
        if s == "bad_meta":
            return "Cannot read the header - move closer or reduce glare"
        if s == "inconsistent":
            return "Several sheets visible - show one sheet"
        return "Hold still"

    def _absent_hint(self, r) -> str:
        if r.status == "partial":
            c = ",".join(r.missing_corners)
            return f"Marker covered or out of frame ({c})" if c else "Marker covered or out of frame"
        return ""

    # ------------------------------------------------------------------ overlay
    def overlay(self, r) -> dict:
        ov: Dict[str, Any] = {"markers": [[[round(x), round(y)] for x, y in q] for q in r.marker_corners.values()]}
        if r.homography is not None:
            try:
                ov["sheet"] = [[round(float(x)), round(float(y))] for x, y in
                               r.image_points([[0, 0], [210, 0], [210, 297], [0, 297]])]
            except Exception:
                pass
        if r.status == "ok" and r.homography is not None and r.bubble_centers_mm:
            ov["bubbles"], ov["bubble_r"] = self._bubbles(r)
        return ov

    def _bubbles(self, r):
        pts: List[Tuple[float, float]] = []
        codes: List[int] = []
        for a, row in zip(r.answers, r.bubble_centers_mm["answers"]):
            for k, p in enumerate(row):
                pts.append(p)
                if k in a.marked:
                    codes.append(B_MULTI if a.state == "multi" else B_AMBIGUOUS if a.ambiguous else B_MARKED)
                else:
                    codes.append(B_AMBIGUOUS if a.ambiguous and (a.guess == k or (a.scores and a.scores[k] > 0.1)) else B_EMPTY)
        for digits, rows in ((r.id_digits, r.bubble_centers_mm["id"]),
                             (r.subject_digits, r.bubble_centers_mm.get("subject") or [])):
            for d, row in zip(digits, rows):
                for k, p in enumerate(row):
                    pts.append(p)
                    if d.value == k:
                        codes.append(B_AMBIGUOUS if d.guessed else B_MARKED)
                    elif d.ambiguous and d.scores and d.scores[k] > 0.1:
                        codes.append(B_AMBIGUOUS)
                    else:
                        codes.append(B_EMPTY)
        img = r.image_points(pts)
        dia = DEFAULT_BUBBLE_MM
        if r.page is not None:
            try:
                if r.form_id is not None:                                # legacy sheet: layout from the resolver
                    lay = self.resolve_layout(r.form_id)
                else:                                                    # self-describing: from the form_key
                    from .layout import header_from_form_key
                    from .recognize import layout_for_header
                    lay = layout_for_header(header_from_form_key(r.form_key))
                dia = lay.pages[r.page].bubble_diameter
            except Exception:
                pass
        rad = max(1.5, 0.5 * dia * float((r.quality or {}).get("source_px_per_mm", 3.0)))
        return [[round(float(x)), round(float(y)), c] for (x, y), c in zip(img, codes)], round(rad, 1)
