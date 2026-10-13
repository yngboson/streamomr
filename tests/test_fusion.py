"""Multi-frame fusion (omr/fusion.py + StreamSession).  Frames are real recognition results of synthetic sheets whose
per-bubble raw measurements are degraded in controlled ways; every degraded frame's decisions are recomputed with the
same decision function, so each frame is self-consistent (as a genuinely bad frame would be)."""
import copy
import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import synth  # noqa: E402

from omr.recognize import decide_page, recognize  # noqa: E402
from omr.stream import StreamSession  # noqa: E402

DT = 0.125


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    assets = synth.prepare(str(tmp_path_factory.mktemp("sheets")), {"q30": synth.STANDARD_TEMPLATES["q30_abcd"]})
    L, pdf = assets["q30"]

    def rec(mode, seed):
        frame, _ = synth.make_frame(L, pdf, synth.CONDS["clean"], np.random.default_rng(seed), mode=mode)
        return recognize(frame)

    cands = [rec("sheet", s) for s in range(1, 25)]
    good = [r for r in cands if r.status == "ok" and r.student_id and r.subject and not r.id_guessed and not r.subject_guessed
            and not any(a.ambiguous for a in r.answers)]
    A = good[0]
    B = next(r for r in good if r.student_id != A.student_id and r.subject != A.subject)
    return {"A": A, "B": B, "E": rec("none", 3)}


def remake(r, FA=None, FI=None, SI=None, SA=None, FS=None, SS=None):
    """Copy of frame result `r` with edited raw measurements and decisions recomputed from them."""
    pick = lambda new, old: None if old is None else (old if new is None else new).copy()
    raw = dataclasses.replace(r.raw, SA=pick(SA, r.raw.SA), FA=pick(FA, r.raw.FA), FI=pick(FI, r.raw.FI), SI=pick(SI, r.raw.SI),
                              FS=pick(FS, r.raw.FS), SS=pick(SS, r.raw.SS),
                              MA=r.raw.MA if FA is None else None, MI=r.raw.MI if FI is None else None,
                              MS=r.raw.MS if FS is None else None)   # edited fills stand alone: no relative-rule measure (omr.recognize REL_MARGIN)
    out = copy.copy(r)
    out.raw = raw
    d = decide_page(raw)
    out.answers, out.id_digits, out.student_id, out.calibration = d.answers, d.id_digits, d.student_id, d.calibration
    out.subject, out.subject_digits = d.subject, d.subject_digits
    out.flags = [f for f in r.flags if not f.startswith(("id_", "subject_"))] + d.flags
    return out


def light_subject(r, digits, level=0.45):
    """The marks of the given subject-code digits come out too faint to read confidently in this frame."""
    FS = r.raw.FS.copy()
    for d in digits:
        p = int(np.argmax(FS[d]))
        FS[d] = np.where(FS[d] > 0.5, level, FS[d])
        FS[d, (p + 1) % 10] = level
    return remake(r, FS=FS)


def unread(r) -> bool:
    """No confident ID: unreadable, or only a flagged best guess (the guess is a value, never a read)."""
    return r.student_id is None or r.id_guessed


def light_digits(r, digits, level=0.45):
    """The marks of the given ID digits come out too faint to read confidently in this frame."""
    FI = r.raw.FI.copy()
    for d in digits:
        p = int(np.argmax(FI[d]))
        FI[d] = np.where(FI[d] > 0.5, level, FI[d])
        FI[d, (p + 1) % 10] = level          # two equal greys: truly unreadable, not even a best guess
    return remake(r, FI=FI)


def run(session, seq, t0=0.0):
    return [session.feed(r, t0 + i * DT, payload=i) for i, r in enumerate(seq)]


def commits(evs):
    return [e["commit"] for e in evs if "commit" in e]


def test_each_frame_unreadable_but_fusion_resolves_the_id(world):
    A = world["A"]
    nd = len(A.id_digits)
    seq = [light_digits(A, [i % nd, (i + 3) % nd, (i + 5) % nd]) for i in range(16)]
    assert all(unread(r) for r in seq)                      # no single frame has the full ID
    cs = commits(run(StreamSession(), seq + [world["E"]] * 3))
    assert len(cs) == 1
    r = cs[0]["result"]
    assert r.student_id == A.student_id and r.fused_frames >= 3 and "id_unreadable" not in r.flags
    assert [(a.number, a.state, a.marked) for a in r.answers] == [(a.number, a.state, a.marked) for a in A.answers]


def test_unresolvable_id_is_held_then_committed_once_unreadable(world):
    A, E = world["A"], world["E"]
    seq = [light_digits(A, [0]) for _ in range(30)]                   # digit 0 never readable
    evs = run(StreamSession(), seq)
    cs = commits(evs)
    assert len(cs) == 1 and cs[0]["result"].student_id is None
    assert evs.index(next(e for e in evs if "commit" in e)) * DT >= 1.5       # held while the sheet stays in view
    # the sheet leaves before the hold expires: committed at track end (not lost, not duplicated)
    evs = run(StreamSession(), seq[:6] + [E] * 4)
    assert len(commits(evs)) == 1 and unread(commits(evs)[0]["result"])
    # and a held track resolved by later frames commits ONCE, with the ID
    seq2 = seq[:4] + [A] * 8 + [E] * 3
    cs = commits(run(StreamSession(), seq2))
    assert len(cs) == 1 and cs[0]["result"].student_id == A.student_id


def test_finish_flushes_a_held_track(world):
    A = world["A"]
    s = StreamSession()
    run(s, [light_digits(A, [0]) for _ in range(6)])
    c = s.finish()
    assert c is not None and unread(c["result"]) and s.finish() is None


def test_a_track_never_commits_twice(world):
    A = world["A"]
    seq = [light_digits(A, [0]) for _ in range(14)] + [A] * 20       # hold expires, THEN the ID would resolve
    cs = commits(run(StreamSession(), seq))
    assert len(cs) == 1


def test_flicker_is_one_commit(world):
    A = world["A"]
    nd = len(A.id_digits)
    rng = np.random.default_rng(3)
    seq = [light_digits(A, list(rng.choice(nd, 3, replace=False))) if i % 2 else A for i in range(30)]
    assert len(commits(run(StreamSession(), seq))) == 1


def test_fast_swap_between_two_sheets_never_blends(world):
    A, B = world["A"], world["B"]
    # no gap at all between the sheets
    cs = commits(run(StreamSession(), [A] * 5 + [B] * 5))
    assert [c["result"].student_id for c in cs] == [A.student_id, B.student_id]
    # alternating every frame: nothing is ever stable -> never a commit with a mixed ID
    cs = commits(run(StreamSession(), [A, B] * 10))
    assert all(c["result"].student_id in (A.student_id, B.student_id, None) for c in cs)
    # sheet 2: SAME answers, ID marks too faint to read in every frame (no confident digit to contradict A)
    FI = np.zeros_like(A.raw.FI)
    for i in range(len(FI)):
        FI[i, (int(A.student_id[i]) + 3) % 10] = 0.45
    B2 = remake(A, FI=FI)
    assert unread(B2)
    evs = run(StreamSession(), [A] * 5 + [B2] * 8 + [world["E"]] * 3)
    cs = commits(evs)
    assert cs[0]["result"].student_id == A.student_id
    assert len(cs) <= 2 and all(c["result"].student_id in (A.student_id, None) or c["result"].id_guessed for c in cs)
    assert not any(c["result"].student_id == A.student_id for c in cs[1:])      # B2 never inherits A's ID


def test_confident_contradiction_breaks_the_track(world):
    A = world["A"]
    FA = A.raw.FA.copy()
    q = 0
    FA[q] = 0.0
    old = A.answers[q].marked[0]
    new = (old + 1) % FA.shape[1]
    FA[q, new] = FA_old = A.raw.FA[q, old]                              # a different answer, read confidently
    SA = A.raw.SA.copy()
    SA[q] = 0.0
    SA[q, new] = A.raw.SA[q, old]
    A2 = remake(A, FA=FA, SA=SA)
    assert A2.answers[0].marked != A.answers[0].marked and not A2.answers[0].ambiguous
    s = StreamSession()
    cs = commits(run(s, [A] * 4 + [A2] * 4))
    # a new track started with the contradicting frame: the old frames are not blended into it
    assert s._track.n == 4 and s._track.decisions().answers[0].marked == A2.answers[0].marked
    assert len(cs) == 1          # a one-question difference is still "the same sheet" for the cooldown rules


def test_unverified_frames_do_not_feed_the_track(world):
    A = world["A"]
    bad = copy.copy(A)
    bad.raw = dataclasses.replace(A.raw, verified=False)
    s = StreamSession()
    run(s, [A, A, bad, A])
    assert s._track is not None and s._track.n == 3


def test_missed_sheet_report_uses_tracks_not_changing_keys(world):
    """The ID resolving mid-track must not leave a stale 'missed' entry of the earlier, unreadable key."""
    A = world["A"]
    s = StreamSession()
    run(s, [light_digits(A, [0, 1, 2])] * 3 + [A] * 6 + [world["E"]] * 3)
    assert s.missed(final=True) == []


def test_suppressed_recommit_is_never_silent(world):
    """Committed 0000000X; after an interruption the next student's sheet has the same answers, 7 of 8 ID digits agree and
    the last is unreadable.  Either it commits or a 'possibly the same sheet' report is visible - never silence."""
    A, E = world["A"], world["E"]
    last = len(A.id_digits) - 1
    N = light_digits(A, [last])                    # same marks everywhere, last ID digit unreadable
    assert unread(N)
    s = StreamSession()
    evs = run(s, [A] * 4 + [E] * 3 + [N] * 6 + [E] * 3)
    cs = commits(evs)
    rep = [m for m in s.missed(final=True) if m.get("possibly_same_as")]
    assert len(cs) == 2 or (len(cs) == 1 and rep and rep[0]["possibly_same_as"] == A.student_id)
    assert any(e["possibly_same"] for e in evs) or len(cs) == 2
    # a serial on only one side blocks the rule: commits
    N2 = copy.copy(N)
    N2.serial = 5
    assert len(commits(run(StreamSession(), [A] * 4 + [E] * 3 + [N2] * 6 + [E] * 3))) == 2
    # differing answers (even one) block the rule: commits
    FA = N.raw.FA.copy()
    FA[0] = 0.0
    N3 = remake(N, FA=FA)
    if [(a.number, a.state, a.marked) for a in N3.answers] != [(a.number, a.state, a.marked) for a in A.answers]:
        assert len(commits(run(StreamSession(), [A] * 4 + [E] * 3 + [N3] * 6 + [E] * 3))) == 2


# ------------------------------------------------------------------ subject code
def test_each_frame_subject_unreadable_but_fusion_resolves_it(world):
    A = world["A"]
    ns = len(A.subject_digits)
    seq = [light_subject(A, [i % ns]) for i in range(16)]
    assert all(r.subject is None for r in seq)                          # no single frame has the full subject
    cs = commits(run(StreamSession(), seq + [world["E"]] * 3))
    assert len(cs) == 1
    r = cs[0]["result"]
    assert r.subject == A.subject and r.student_id == A.student_id and r.fused_frames >= 3
    assert "subject_unreadable" not in r.flags


def test_unresolvable_subject_is_held_then_committed_once_unread(world):
    A, E = world["A"], world["E"]
    seq = [light_subject(A, [0]) for _ in range(30)]
    evs = run(StreamSession(), seq)
    cs = commits(evs)
    assert len(cs) == 1 and cs[0]["result"].subject is None and cs[0]["result"].student_id == A.student_id
    assert "subject_unreadable" in cs[0]["result"].flags
    assert evs.index(next(e for e in evs if "commit" in e)) * DT >= 1.5       # held like an unreadable ID
    cs = commits(run(StreamSession(), seq[:4] + [A] * 8 + [E] * 3))
    assert len(cs) == 1 and cs[0]["result"].subject == A.subject


def test_other_subject_is_another_sheet_never_blended(world):
    A, B = world["A"], world["B"]
    assert A.subject != B.subject
    cs = commits(run(StreamSession(), [A] * 5 + [B] * 5))
    assert [c["result"].subject for c in cs] == [A.subject, B.subject]
    # SAME student and answers, but another readable subject: a different sheet (identity includes the subject)
    FS = A.raw.FS.copy()
    for i in range(len(FS)):
        FS[i] = 0.0
        FS[i, (int(A.subject[i]) + 1) % 10] = 1.0
    A2 = remake(A, FS=FS, SS=np.where(FS > 0.5, A.raw.SS.max(), 0.0))
    assert A2.subject is not None and A2.subject != A.subject
    cs = commits(run(StreamSession(), [A] * 5 + [A2] * 5))
    assert [c["result"].subject for c in cs] == [A.subject, A2.subject]
    s = StreamSession()
    run(s, [A] * 4 + [A2] * 3)
    assert s._track.n == 3                              # the contradicting frame started a new track
    assert all(c["result"].subject != A.subject or c["result"].fused_frames <= 5 for c in cs)


def test_subject_filter(world):
    A, B = world["A"], world["B"]
    s = StreamSession(allowed_subjects={A.subject})
    cs = commits(run(s, [B] * 6 + [world["E"]] * 5 + [A] * 6))
    assert [c["result"].subject for c in cs] == [A.subject]
    assert "another subject" in run(StreamSession(allowed_subjects={A.subject}), [B] * 3)[-1]["hint"]
    # an unreadable subject is NOT filtered out: it may be the wanted one
    U = light_subject(A, [0])
    assert len(commits(run(StreamSession(allowed_subjects={"zz"}), [U] * 4 + [world["E"]] * 3))) == 1


# ---------------------------------------------------------------------------- unreliable (sentinel) bubbles in fusion
def _sentinel_track(frames):
    """Track over fake frames; each frame = (FA row of Q1) with the UNRELIABLE sentinel where given as None."""
    from types import SimpleNamespace
    from omr.fusion import Track
    from omr.recognize import PageRaw, UNRELIABLE_FILL, UNRELIABLE_STRENGTH

    def mk(fa):
        F = np.zeros((1, 5))
        S = np.full((1, 5), 1.0)
        for i, v in enumerate(fa):
            if v is None:
                F[0, i], S[0, i] = UNRELIABLE_FILL, UNRELIABLE_STRENGTH
            else:
                F[0, i] = v
        r = PageRaw([1], [list("ABCDE")], [0], F, S, np.zeros((1, 10)), np.ones((1, 10)), 0.5, 0.6)
        return SimpleNamespace(raw=r, form_key="f", page=0, serial=None, homography=None, quality={})
    rs = [mk(f) for f in frames]
    t = Track(1, rs[0], 0.0)
    for i, r in enumerate(rs[1:], 1):
        t.add(r, i * DT, None)
    return t


def test_unreliable_bubble_samples_do_not_vote_in_fusion():
    # the real scan: the marked bubble (C) is measured once and unknowable in the other frames; the rest too
    u = None
    t = _sentinel_track([[u, u, 1.0, u, 0.02], [0.0, 0.01, u, u, u], [u, 0.02, u, 0.0, u]])
    fa = t.fused_raw().FA[0]
    assert fa[2] == 1.0 and fa[0] == 0.0 and abs(fa[1] - 0.015) < 1e-9 and fa[3] == 0.0 and fa[4] == 0.02    # no 0.5 contamination
    a = t.decisions().answers[0]
    assert a.state == "single" and a.marked == [2] and not a.ambiguous


def test_bubble_unreliable_in_every_frame_stays_unreliable_and_is_never_picked():
    from omr.recognize import UNRELIABLE_FILL, UNRELIABLE_STRENGTH
    u = None
    t = _sentinel_track([[u, u, u, u, u], [u, u, u, u, u]])
    raw = t.fused_raw()
    assert (raw.FA == UNRELIABLE_FILL).all() and (raw.SA == UNRELIABLE_STRENGTH).all()
    a = t.decisions().answers[0]
    assert a.ambiguous and a.guess is None and a.marked == []       # five equal unknowable greys: no pick by index
