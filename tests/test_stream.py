"""StreamSession on sequences of real recognition results from synthetic frames.

Frames are recognised once (module cache); sequences are built by repeating / copying those results, so
the state machine is exercised on real statuses (ok / partial / no_sheet / bad_*) without re-running OpenCV.
"""
import copy
import dataclasses
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import synth  # noqa: E402

from omr.recognize import recognize  # noqa: E402
from omr.stream import COOLDOWN, DETECTING, IDLE, StreamSession, decision_key  # noqa: E402

DT = 0.125


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    assets = synth.prepare(str(tmp_path_factory.mktemp("sheets")), {"q30": synth.STANDARD_TEMPLATES["q30_abcd"]})
    L, pdf = assets["q30"]

    def rec(mode, seed, cond="clean"):
        frame, _ = synth.make_frame(L, pdf, synth.CONDS[cond], np.random.default_rng(seed), mode=mode)
        return recognize(frame)

    w = {"layout": L}
    w["A"] = next(r for r in (rec("sheet", s) for s in range(1, 12)) if r.status == "ok" and r.student_id and r.subject and not r.id_guessed and not r.subject_guessed)
    w["B"] = next(r for r in (rec("sheet", s) for s in range(2, 20)) if r.status == "ok" and r.student_id and r.subject and not r.id_guessed and not r.subject_guessed
                  and r.student_id != w["A"].student_id)
    w["empty"] = rec("none", 3)
    w["thumb"] = next(r for r in (rec("thumb2", s) for s in range(20, 40)) if r.status != "ok")
    w["cut"] = next(r for r in (rec("cut", s) for s in range(40, 70)) if r.status != "ok")
    assert w["A"].status == "ok" and w["B"].status == "ok" and w["empty"].status == "no_sheet"
    return w


def run(session, seq, t0=0.0):
    evs = []
    for i, r in enumerate(seq):
        evs.append(session.feed(r, t0 + i * DT, payload=i))
    return evs


def commits(evs):
    return [e["commit"] for e in evs if "commit" in e]


def sharper(r, s):
    r = copy.copy(r)
    r.quality = dict(r.quality, sharpness=s)
    return r


def flip_answer(r, q=0):
    r = copy.copy(r)
    r.answers = [dataclasses.replace(a) for a in r.answers]
    a = r.answers[q]
    a.marked = [(a.marked[0] + 1) % 4] if a.marked else [0]
    a.state = "single"
    return r


def other_student(r):
    r = copy.copy(r)
    r.student_id = "99999999"
    return r


def test_enter_hold_leave_next_sheet_one_commit_each(world):
    A, B, E, T = world["A"], world["B"], world["empty"], world["thumb"]
    seq = [E, E, T, A, A, A, A, A, A, A, A, A, T, E, E, E, E, E, E, B, B, B, B, B, B, E, E, E, E, E]
    s = StreamSession()
    evs = run(s, seq)
    cs = commits(evs)
    assert len(cs) == 2
    assert cs[0]["result"].student_id == A.student_id and cs[1]["result"].student_id == B.student_id
    assert evs[-1]["state"] == IDLE
    # the commit is made on the K-th agreeing frame, not earlier
    assert cs[0]["frame_index"] == 5


def test_single_good_frame_never_commits(world):
    A, E, T, C = world["A"], world["empty"], world["thumb"], world["cut"]
    for seq in ([E, A, E, E, E, E], [T, A, T, A, T, A, T], [A, A, E, A, A, E, A, A, E], [A, C, A, C, A]):
        assert commits(run(StreamSession(), seq)) == []
    assert commits(run(StreamSession(k=3), [A, A])) == []
    with pytest.raises(ValueError):
        StreamSession(k=1)


def test_held_still_commits_once(world):
    A = world["A"]
    evs = run(StreamSession(), [A] * 200)
    assert len(commits(evs)) == 1
    assert evs[5 - 3]["state"] == "committed"
    assert all(e["state"] == COOLDOWN for e in evs[3:])


def test_flicker_during_cooldown_no_extra_commit(world):
    A = world["A"]
    A2 = flip_answer(A)
    assert decision_key(A) != decision_key(A2)
    seq = [A] * 4 + [A2] * 6 + [A] * 3 + [A2] * 5   # even a stable-looking alternative version
    assert len(commits(run(StreamSession(), seq))) == 1


def test_fast_swap_without_empty_frame_commits_new_sheet(world):
    A, B = world["A"], world["B"]
    evs = run(StreamSession(), [A] * 5 + [B] * 5)
    cs = commits(evs)
    assert [c["result"].student_id for c in cs] == [A.student_id, B.student_id]
    # same id but very different answers is also a different sheet; one flipped answer is not
    A3 = copy.copy(A)
    A3.answers = [dataclasses.replace(a, marked=[(a.marked[0] + 1) % 4] if a.marked else [0], state="single") for a in A.answers]
    assert len(commits(run(StreamSession(), [A] * 5 + [A3] * 5))) == 2
    assert len(commits(run(StreamSession(), [A] * 5 + [other_student(A)] * 5))) == 2


def test_swap_needs_k_stable_frames(world):
    A, B = world["A"], world["B"]
    evs = run(StreamSession(), [A] * 4 + [B, B, A, A, B, B])
    assert len(commits(evs)) == 1


def test_put_back_after_leaving_commits_again(world):
    A, E = world["A"], world["empty"]
    assert len(commits(run(StreamSession(), [A] * 4 + [E] * 5 + [A] * 4))) == 2
    assert len(commits(run(StreamSession(), [A] * 4 + [E] * 3 + [A] * 4))) == 1   # < M empty frames: same sheet


def test_picks_sharpest_of_stable_frames(world):
    A = world["A"]
    seq = [sharper(A, 10), sharper(A, 90), sharper(A, 50)]
    c = commits(run(StreamSession(), seq))[0]
    assert c["payload"] == 1 and c["result"].quality["sharpness"] == 90


def test_occluded_marker_no_commit_and_hint(world):
    T, C = world["thumb"], world["cut"]
    s = StreamSession()
    evs = run(s, [T] * 30)
    assert commits(evs) == []
    assert evs[-1]["hint"] and evs[-1]["state"] in (DETECTING, IDLE)
    if T.status == "partial":
        assert "arker" in evs[-1]["hint"]


def test_unstable_sheet_gets_hint_after_timeout_not_commit(world):
    A = world["A"]
    seq = [flip_answer(A, i % 5) if i % 2 else A for i in range(30)]   # never 3 in a row
    evs = run(StreamSession(hint_after=2.0), seq)
    assert commits(evs) == []
    assert evs[3]["hint"] == "Reading..."
    assert evs[-1]["state"] == DETECTING and evs[-1]["hint"] not in ("", "Reading...")


def test_allowed_forms_and_subjects(world):
    A = world["A"]
    evs = run(StreamSession(allowed_forms={"Q1U1C2D1S0P1L0B0V1"}), [A] * 6)
    assert commits(evs) == [] and "another form" in evs[-1]["hint"]
    assert len(commits(run(StreamSession(allowed_forms={A.form_key}), [A] * 6))) == 1
    evs = run(StreamSession(allowed_subjects={"99" if A.subject != "99" else "98"}), [A] * 6)
    assert commits(evs) == [] and "another subject" in evs[-1]["hint"]
    assert len(commits(run(StreamSession(allowed_forms={A.form_key}, allowed_subjects={A.subject}), [A] * 6))) == 1


def test_subject_is_part_of_the_sheet_identity(world):
    A = world["A"]
    A2 = copy.copy(A)
    A2.subject = "99" if A.subject != "99" else "98"
    assert decision_key(A) != decision_key(A2)
    assert len(commits(run(StreamSession(fuse=False), [A] * 5 + [A2] * 5))) == 2


def test_unknown_legacy_form_hint():
    from omr.recognize import RecognitionResult
    r = RecognitionResult(status="unknown_form", form_id=7)
    ev = StreamSession().feed(r, 0.0)
    assert ev["state"] != "committed" and "Unknown sheet" in ev["hint"]


def test_overlay(world):
    A, E, T = world["A"], world["empty"], world["thumb"]
    s = StreamSession()        # self-describing sheet: the bubble size comes from the form_key, no resolver
    ov = s.feed(A, 0)["overlay"]
    assert len(ov["markers"]) == 4 and len(ov["sheet"]) == 4
    n_q = len(A.answers)
    assert len(ov["bubbles"]) == n_q * 4 + (len(A.id_digits) + len(A.subject_digits)) * 10 and ov["bubble_r"] > 1.5
    assert sum(1 for b in ov["bubbles"] if b[2] == 1) >= n_q // 2
    assert "bubbles" not in s.feed(T, 1)["overlay"]
    assert s.feed(E, 2)["overlay"]["markers"] == []


def bad_frames(world, n):
    return [world["thumb"], world["cut"], world["thumb"]][:n]


def similar_sheet_without_id(A):
    """Another student's sheet: unreadable ID, answers identical except for ~10% of the questions."""
    r = copy.copy(A)
    r.student_id = None
    r.answers = [dataclasses.replace(a) for a in A.answers]
    for a in r.answers[::10]:
        a.marked = [(a.marked[0] + 1) % 4] if a.marked else [0]
        a.state = "single"
    return r


def test_pile_swap_unreadable_id_similar_answers_commits(world):
    A = world["A"]
    N = similar_sheet_without_id(A)
    assert 0 < sum(x.marked != y.marked for x, y in zip(A.answers, N.answers)) <= 0.3 * len(A.answers)
    for gap in (2, 3):
        evs = run(StreamSession(), [A] * 4 + bad_frames(world, gap) + [N] * 4)
        cs = commits(evs)
        assert len(cs) == 2 and cs[1]["result"].student_id is None
    # only one of the two IDs readable
    evs = run(StreamSession(), [N] * 4 + bad_frames(world, 2) + [flip_answer(A)] * 4)
    assert len(commits(evs)) == 2


def test_held_still_flicker_with_unreadable_id_no_commit(world):
    A = world["A"]
    N = similar_sheet_without_id(A)
    assert len(commits(run(StreamSession(), [A] * 4 + [N] * 8 + [A] * 3 + [N] * 8))) == 1
    # a single disturbed frame is not an interruption
    assert len(commits(run(StreamSession(), [A] * 4 + [world["thumb"]] + [N] * 8))) == 1


def test_same_sheet_lifted_and_put_back_no_new_commit(world):
    A = world["A"]
    for gap in (2, 3):
        assert len(commits(run(StreamSession(), [A] * 4 + bad_frames(world, gap) + [A] * 6))) == 1
    # and the interruption is settled afterwards: later flicker stays quiet
    assert len(commits(run(StreamSession(), [A] * 4 + bad_frames(world, 2) + [A] * 4 + [flip_answer(A)] * 6))) == 1


def test_quad_jump_counts_as_interruption(world):
    A = world["A"]
    moved = copy.copy(A)
    moved.marker_corners = {k: [[x + 400, y + 100] for x, y in q] for k, q in A.marker_corners.items()}
    N = similar_sheet_without_id(A)
    N.marker_corners = moved.marker_corners
    assert len(commits(run(StreamSession(), [A] * 4 + [N] * 4))) == 2


# ------------------------------------------------------------------ "sheet present" accounting
def test_partial_frames_count_as_sheet_present(world):
    """A sheet in view with covered markers (partial) is present: events say so, and a never-read sheet is reported."""
    T, E, A = world["thumb"], world["empty"], world["A"]
    assert T.status == "partial" or T.status != "no_sheet"
    s = StreamSession()
    evs = run(s, [T] * 12)
    assert all(e["present"] for e in evs)
    assert evs[-1]["unread"] and not evs[0]["unread"]       # long enough without a commit
    assert not run(StreamSession(), [E] * 4)[-1]["present"]
    run(s, [E] * 6, t0=10.0)                                # gone for >= M frames: the episode ends
    eps = s.unread_episodes()
    assert len(eps) == 1 and eps[0]["ok"] == 0 and eps[0]["frames"] == 12 and eps[0]["statuses"] == {T.status: 12}


def test_read_sheet_is_not_reported_unread_and_gaps_do_not_split_episodes(world):
    A, E, T = world["A"], world["empty"], world["thumb"]
    s = StreamSession()
    run(s, [A] * 4 + [E, E] + [T] * 5 + [E] * 6)
    assert s.unread_episodes(final=True) == []              # one episode (gap < M), it did commit
    s2 = StreamSession()
    run(s2, [T] * 3 + [E] * 6 + [A] * 4 + [E] * 6)
    assert len(s2.unread_episodes(final=True)) == 1         # the first stretch never read, the second did


def guessed_id_sheet(A):
    """Another student's sheet whose ID is only a best guess (every digit valued but flagged guessed)."""
    r = similar_sheet_without_id(A)
    r.id_digits = [dataclasses.replace(d, value=(d.value + 1) % 10 if d.value is not None else 0, guessed=True, ambiguous=True)
                   for d in A.id_digits]
    r.student_id = "".join(str(d.value) for d in r.id_digits)
    r.flags = list(A.flags) + ["id_guessed"]
    return r


def test_guessed_id_is_no_evidence_a_pile_swap_still_commits_and_the_commit_stays_flagged(world):
    A = world["A"]
    N = guessed_id_sheet(A)
    assert N.id_guessed and N.needs_review
    cs = commits(run(StreamSession(), [A] * 4 + bad_frames(world, 2) + [N] * 4))
    assert len(cs) == 2 and cs[1]["result"].id_guessed         # not mistaken for A put back; reaches the DB as a flagged guess
    # a guessed digit never counts as agreeing evidence, even when it equals the committed digit
    S = StreamSession()
    run(S, [A] * 4)
    assert S._same_by_evidence(N) is False
    # the hint asks for a clearer ID instead of "hold still, reading keeps changing"
    H = StreamSession()
    H._since = 0.0
    assert "ID" in H._unstable_hint(N, 99.0)


# ---------------------------------------------------------------- revision of a committed sheet (omr/revision.py)
import test_fusion as TF  # noqa: E402


@pytest.fixture(scope="module")
def fw(tmp_path_factory):
    assets = synth.prepare(str(tmp_path_factory.mktemp("sheets_rev")), {"q30": synth.STANDARD_TEMPLATES["q30_abcd"]})
    L, pdf = assets["q30"]

    def rec(mode, seed):
        frame, _ = synth.make_frame(L, pdf, synth.CONDS["clean"], np.random.default_rng(seed), mode=mode)
        return recognize(frame)

    good = [r for r in (rec("sheet", s) for s in range(1, 25)) if r.status == "ok" and r.student_id and r.subject and not r.id_guessed
            and not r.subject_guessed and not any(a.ambiguous for a in r.answers)]
    A = good[0]
    B = next(r for r in good if r.student_id != A.student_id and r.subject != A.subject)
    return {"A": A, "B": B, "E": rec("none", 3)}


def _doubt_row(r, q):
    """Frame whose answer row q is a grey zone (doubtful) in this frame."""
    FA = r.raw.FA.copy()
    p = int(np.argmax(FA[q]))
    FA[q] = np.where(FA[q] > 0.5, 0.45, FA[q])
    FA[q, (p + 1) % FA.shape[1]] = 0.45
    return TF.remake(r, FA=FA)


def revisions(evs):
    return [e["revision"] for e in evs if "revision" in e]


def test_revision_resolves_unreadable_id_after_commit(fw):
    A, E = fw["A"], fw["E"]
    seq = [TF.light_digits(A, [0]) for _ in range(16)] + [A] * 16 + [E] * 6
    evs = run(StreamSession(), seq)
    cs, rs = commits(evs), revisions(evs)
    assert len(cs) == 1 and cs[0]["result"].student_id is None
    assert rs and rs[-1]["result"].student_id == A.student_id and "id" in rs[0]["changed"]
    assert rs[0]["prev_severity"] == "severe" and rs[-1]["severity"] in ("ok", "minor")
    assert len(cs) == 1                                               # never a second commit


def test_revision_upgrades_doubtful_answer_row(fw):
    A, E = fw["A"], fw["E"]
    q = 4
    seq = [_doubt_row(A, q) for _ in range(16)] + [A] * 16 + [E] * 6
    evs = run(StreamSession(), seq)
    cs, rs = commits(evs), revisions(evs)
    assert len(cs) == 1 and cs[0]["result"].answers[q].ambiguous
    assert rs and f"q{q + 1}" in rs[0]["changed"] or any(f"q{A.answers[q].number}" in x["changed"] for x in rs)
    last = rs[-1]["result"]
    assert not last.answers[q].ambiguous and (last.answers[q].state, last.answers[q].marked) == (A.answers[q].state, A.answers[q].marked)
    assert not rs[-1]["contradiction"]


def test_revision_never_flips_a_confident_row_silently(fw):
    A, E = fw["A"], fw["E"]
    seq = [TF.light_digits(A, [0]) for _ in range(16)]                # commit with a doubtful ID ...
    B = flip_answer(A, 0)                                             # ... then the fused evidence says Q1 is something else
    B = TF.remake(A, FA=_swap_row(A, 0))
    evs = run(StreamSession(), seq + [B] * 24 + [E] * 6)
    rs = revisions(evs)
    cs = commits(evs)
    committed = cs[0]["result"].answers[0]
    for x in rs:
        a = x["result"].answers[0]
        assert (a.state, a.marked) == (committed.state, committed.marked)     # the committed confident row is untouched
    assert any(x["contradiction"] for x in rs) or not rs
    for x in rs:
        if x["contradiction"]:
            assert "contradiction" in x["result"].flags and x["result"].needs_review and x["severity"] != "ok"


def _swap_row(r, q):
    FA = r.raw.FA.copy()
    row = FA[q].copy()
    FA[q] = np.roll(row, 1)
    return FA


def test_no_revision_for_a_different_sheet(fw):
    A, B, E = fw["A"], fw["B"], fw["E"]
    seq = [TF.light_digits(A, [0]) for _ in range(16)] + [B] * 8 + [E] * 6
    evs = run(StreamSession(), seq)
    assert len(commits(evs)) == 2 and not revisions(evs)
    assert commits(evs)[1]["result"].student_id == B.student_id
