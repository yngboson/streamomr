"""Decision rules after the illum / auto-ink integration: unreliable bubbles carry through the fills, strict digits
need a clear margin, the ink-floor choice is recorded and fused."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from omr import recognize as R  # noqa: E402
from omr.fusion import Track  # noqa: E402

CAL = R._Calib(ink=0.5, typ=1.0, t_hi=0.65, t_lo=0.13)
LIGHT = R._Calib(ink=0.3, typ=0.69, t_hi=R.T_HI_MIN, t_lo=R.T_LO_FRAC * R.T_HI_MIN)   # the loosest legal calibration


def test_unreliable_bubble_makes_its_row_ambiguous_whatever_the_others_are():
    D = np.zeros((10, 40), np.float32)
    D[3] = 0.9                                        # a clear mark ...
    unrel = np.zeros(10, bool)
    unrel[7] = True                                   # ... and a bubble nobody can judge
    for cal in (CAL, LIGHT):
        f, s = R._fill(D, 0.5, unrel)
        assert f[7] == R.UNRELIABLE_FILL and s[7] == R.UNRELIABLE_STRENGTH
        for strict in (False, True):
            state, marked, conf, amb = R._decide(f, s, cal, strict=strict)
            assert amb
        # same for an otherwise blank row: the unknown bubble is never silently "blank"
        fb, sb = R._fill(np.zeros_like(D), 0.5, unrel)
        assert R._decide(fb, sb, cal, strict=True)[3]


def test_reliable_rows_are_untouched_by_the_flag_argument():
    D = np.random.default_rng(0).uniform(0, 1, (10, 40)).astype(np.float32)
    f0, s0 = R._fill(D, 0.5)
    f1, s1 = R._fill(D, 0.5, np.zeros(10, bool))
    assert np.array_equal(f0, f1) and np.array_equal(s0, s1)


def test_strict_digit_needs_a_margin_above_t_hi_but_answers_do_not():
    f = np.zeros(10)
    st = np.zeros(10)
    f[4], st[4] = CAL.t_hi + 0.02, 1.0                # right at the threshold: a half-filled bubble can measure this
    assert R._decide(f, st, CAL)[:2] == ("single", [4]) and not R._decide(f, st, CAL)[3]
    state, marked, conf, amb = R._decide(f, st, CAL, strict=True)
    assert amb                                        # a digit is not read from it
    f[4] = CAL.t_hi + R.DIGIT_MARGIN + 0.01
    assert R._decide(f, st, CAL, strict=True) == ("single", [4], R._decide(f, st, CAL, strict=True)[2], False)


def _raw(source, sep):
    z = np.zeros((2, 5))
    return R.PageRaw([1, 2], [["A"] * 5] * 2, [0], z, z, np.zeros((1, 10)), np.zeros((1, 10)), 0.5, 0.6,
                     ink_source=source, ink_separation=sep)


def test_ink_choice_is_recorded_and_fused_conservatively():
    c = R.decide_page(_raw("auto", 0.2)).calibration
    assert c["ink_auto"] == 1.0 and c["ink_source"] == "auto" and abs(c["ink_separation"] - 0.2) < 1e-9
    assert R.decide_page(_raw("default", 0.0)).calibration["ink_auto"] == 0.0

    class Fr:                                          # the bits of a recognition result a Track reads
        def __init__(self, raw):
            self.raw, self.form_key, self.page, self.serial, self.quality, self.homography = raw, "f", 0, None, {}, None
    tr = Track(1, Fr(_raw("auto", 0.1)), 0.0)
    tr.add(Fr(_raw("auto", 0.3)), 0.1, None)
    tr.add(Fr(_raw("auto", 0.2)), 0.2, None)
    fr = tr.fused_raw()
    assert fr.ink_source == "auto" and abs(fr.ink_separation - 0.2) < 1e-9
    tr.add(Fr(_raw("default", 0.0)), 0.3, None)
    assert tr.fused_raw().ink_source == "default"     # one frame without evidence: the whole track keeps the default rule


# ---------------------------------------------------------------- relative rule (REL_MARGIN): faint marks beside blank choices
def _faint_row(n=5, k=2, excess=0.2, fill=0.3, strength=0.5):
    """A row whose absolute rules cannot call (low fill / strength) but where choice k is `excess` darker than the rest."""
    f, st, md = np.zeros(n), np.zeros(n), np.full(n, 0.04)
    f[k], st[k], md[k] = fill, strength, 0.04 + excess
    return f, st, md


def test_relative_rule_reads_a_faint_mark_the_absolute_rules_only_guess():
    f, st, md = _faint_row(excess=0.22)
    assert R._decide(f, st, CAL)[2:] == (0.2, True)                         # before: a flagged guess at best
    state, marked, conf, amb = R._decide(f, st, CAL, md=md)
    assert (state, marked, amb) == ("single", [2], False) and conf > 0.5


def test_relative_rule_close_to_the_margin_is_read_but_flagged():
    f, st, md = _faint_row(excess=R.REL_MARGIN + 0.01)
    state, marked, conf, amb = R._decide(f, st, CAL, md=md)
    assert (state, marked, amb) == ("single", [2], True)
    f, st, md = _faint_row(excess=R.REL_MARGIN - 0.02)                      # below 15 points: nothing new
    assert R._decide(f, st, CAL, md=md) == R._decide(f, st, CAL)
    assert R._decide(np.zeros(5), np.zeros(5), CAL, md=np.full(5, 0.05))[0] == "blank"


def test_relative_rule_two_faint_marks_are_a_multi_mark():
    f, st, md = _faint_row(excess=0.25)
    md[4] = 0.04 + 0.25                                                    # a second faint mark; the median of the others stays blank
    state, marked, conf, amb = R._decide(f, st, CAL, md=md)
    assert (state, marked, amb) == ("multi", [2, 4], False)


def test_relative_rule_never_changes_a_row_with_a_confident_absolute_mark():
    f, st = np.array([0.0, 0.95, 0.0, 0.0, 0.0]), np.array([0, 1.0, 0, 0, 0.0])
    md = np.array([0.04, 0.8, 0.04, 0.3, 0.04])                           # an erased ghost beside a real mark: 0.26 above the rest
    assert R._decide(f, st, CAL, md=md) == R._decide(f, st, CAL) and R._decide(f, st, CAL)[1] == [1]


def test_relative_rule_leaves_unknowable_rows_and_partial_marks_alone():
    f, st, md = _faint_row(excess=0.3)
    md[0] = np.nan                                                         # glare / shadow edge on a bubble of the row
    assert R._decide(f, st, CAL, md=md) == R._decide(f, st, CAL)
    D = np.full((5, 60), 0.03, np.float32)
    D[1, :20] = 0.6                                                        # half-filled dark bubble: dark on a third of the disc
    D[2] = 0.25                                                            # complete faint mark
    D[3, :4] = 0.7                                                         # a stray tick
    m = R._mean_dark(D)
    assert m[2] > 0.2 and m[1] == 0 and m[3] < 0.1 and m[0] < 0.1
    assert np.isnan(R._mean_dark(D, np.array([0, 0, 0, 0, 1], bool))[4])


def test_relative_rule_digits_need_the_stricter_excess_and_one_mark():
    f, st, md = _faint_row(n=10, k=6, excess=R.REL_SURE_DIGIT + 0.02)
    state, marked, conf, amb = R._decide(f, st, CAL, strict=True, md=md)
    assert (state, marked, amb) == ("single", [6], False)
    f, st, md = _faint_row(n=10, k=6, excess=R.REL_SURE_DIGIT - 0.03)      # faint: read as a mark but flagged -> digit unread
    assert R._decide(f, st, CAL, strict=True, md=md)[3]
    md[1] = md[6]                                                          # two faint marks in the column
    assert R._decide(f, st, CAL, strict=True, md=md)[0] == "multi"


def test_relative_measure_is_fused_and_used_by_decide_page():
    z = np.zeros((2, 5))
    MA = np.full((2, 5), 0.04)
    MA[0, 3] = 0.3
    raw = R.PageRaw([1, 2], [["A", "B", "C", "D", "E"]] * 2, [0], z, z, np.zeros((1, 10)), np.zeros((1, 10)), 0.5, 0.6, MA=MA)
    a = R.decide_page(raw).answers
    assert a[0].state == "single" and a[0].marked == [3] and a[1].state == "blank"
    assert R.decide_page(_raw("auto", 0.2)).answers[0].state == "blank"   # PageRaw without MA: absolute rules only


def test_relative_rule_is_off_unless_the_page_looks_like_faint_pencil_throughout():
    f, st, md = _faint_row(excess=0.25)
    assert R._decide(f, st, CAL, md=md, trust_rel=False) == R._decide(f, st, CAL)
    n, nc = 20, 5
    MA = np.full((n, nc), 0.04)
    MA[np.arange(n), np.arange(n) % nc] = 0.3                              # every question has one faint mark
    z = np.zeros((n, nc))
    mk = lambda MA_: R.PageRaw(list(range(1, n + 1)), [list("ABCDE")] * n, [0], z, z, np.zeros((1, 10)), np.zeros((1, 10)), 0.5, 0.6, MA=MA_)
    tr, page = R._rel_trust_answers(mk(MA), CAL)
    assert page and tr.all()
    assert [a.marked for a in R.decide_page(mk(MA)).answers] == [[i % nc] for i in range(n)]
    few = np.full((n, nc), 0.04)
    few[:3, 0] = 0.3                                                       # a few ghost-like rows on an otherwise blank page
    assert not R._rel_trust_answers(mk(few), CAL)[0][:3].any() and all(a.state == "blank" for a in R.decide_page(mk(few)).answers)
    # a confident dark mark with a ghost beside it is erasure evidence: the page is not trusted
    F = z.copy()
    F[:, 0] = 0.95
    S = z.copy()
    S[:, 0] = 1.0
    ghosty = MA.copy()
    ghosty[:, 0] = 0.8
    ghosty[:, 1] = 0.3
    raw = R.PageRaw(list(range(1, n + 1)), [list("ABCDE")] * n, [0], F, S, np.zeros((1, 10)), np.zeros((1, 10)), 0.5, 0.6, MA=ghosty)
    assert not R._rel_trust_answers(raw, CAL)[0].any()


def test_on_a_faint_pencil_page_a_part_dark_bubble_read_by_the_absolute_rules_is_flagged():
    f, st = np.array([0.0, 0.0, 0.8, 0.0, 0.0]), np.array([0, 0, 1.0, 0, 0.0])
    md = np.array([0.04, 0.04, 0.0, 0.04, 0.04])                          # dark on part of the disc only (md 0), fill 0.58
    assert not R._decide(f, st, CAL, md=md, trust_rel=False)[3]
    assert R._decide(f, st, CAL, md=md, trust_rel=True)[3]
    md[2] = 0.3                                                            # an evenly dark mark: untouched
    assert not R._decide(f, st, CAL, md=md, trust_rel=True)[3]


def _page(n, MA, F=None, S=None, FI=None, SI=None, MI=None):
    z = np.zeros((n, 5))
    return R.PageRaw(list(range(1, n + 1)), [list("ABCDE")] * n, list(range(0 if FI is None else len(FI))),
                     z if F is None else F, z if S is None else S,
                     np.zeros((0, 10)) if FI is None else FI, np.zeros((0, 10)) if SI is None else SI, 0.5, 0.6, MA=MA,
                     MI=MI)


def test_relative_trust_is_local_to_the_region_not_the_page():
    """A faint-ink part (the last questions) and an ordinary part on one page: the rule is on where the faint marks are (and the dark half has no ghosts)."""
    n = 40
    MA = np.full((n, 5), 0.04)
    MA[np.arange(n), np.arange(n) % 5] = 0.4                      # every question has a strong pencil mark the strict rules cannot take ...
    F = np.zeros((n, 5)); S = np.zeros((n, 5))
    F[:28, 0] = 0.95; S[:28, 0] = 1.0                            # ... but the first half is dark and confident
    MA[:28] = 0.04; MA[:28, 0] = 0.8
    MA[28:] = 0.04; MA[28 + np.arange(12), np.arange(12) % 5] = 0.4   # (second half: faint ink only)
    t, page = R._rel_trust_answers(_page(n, MA, F, S), CAL)
    assert not page
    assert t[35:].all() and not t[:18].any()


def test_relative_trust_digits_block_has_its_own_evidence():
    FI = np.zeros((6, 10)); SI = np.zeros((6, 10)); MI = np.full((6, 10), 0.03)
    MI[np.arange(6), np.arange(6)] = 0.45                         # pencil ID: uniform, no fill read by the strict absolute rules
    assert R._rel_trust_digits(FI, SI, MI, CAL)
    MI2 = np.full((6, 10), 0.03); MI2[0, 1] = 0.45               # one lone faint mark: erasure-like
    assert not R._rel_trust_digits(FI, SI, MI2, CAL)


def test_pencil_id_next_to_pen_answers_is_read():
    """Pen answers set the page ink level high; the pencil digits are judged against their own block (omr.ink.local_levels)."""
    from omr import ink as inkmod
    row_marks = np.concatenate([np.full(20, 0.77), np.full(7, 0.46)])
    lv = inkmod.local_levels(row_marks, 0.4, 0.77, [(range(20), True), (range(20, 27), False)])
    assert np.allclose(lv[:20], 0.77) and np.allclose(lv[20:], 0.46)
    D = np.full((10, 40), 0.03, np.float32); D[4] = 0.46          # a pencil digit
    f_page, s_page = R._fill(D, 0.77)
    f_loc, s_loc = R._fill(D, np.full(10, 0.46, np.float32))
    assert s_page[4] < R.MIN_STRENGTH <= s_loc[4] and f_loc[4] > f_page[4]
    cal = R._Calib(ink=0.46, typ=1.0, t_hi=0.65, t_lo=0.13)
    assert R._decide(f_loc, s_loc, cal, strict=True)[:2] == ("single", [4])
    assert R._decide(f_page, s_page, cal, strict=True)[3]         # page-wide level: the same mark is flagged (the old failure)


def test_local_ink_never_turns_a_half_filled_digit_into_a_read():
    D = np.full((10, 60), 0.03, np.float32); D[4, :27] = 0.46    # a pencil half-filled bubble (45 % of the disc)
    f, s = R._fill(D, np.full(10, 0.46, np.float32))
    cal = R._Calib(ink=0.46, typ=1.0, t_hi=0.65, t_lo=0.13)
    assert R._decide(f, s, cal, strict=True)[3] or R._decide(f, s, cal, strict=True)[0] != "single"


def test_local_levels_only_go_down_in_a_lighter_region_and_ignore_ghosts():
    from omr import ink as inkmod
    rm = np.array([0.8, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5, 0.0, 0.3, 0.5, 0.5, 0.5])   # dark pen, then a lighter pen; a blank row and a ghost (0.3 < floor)
    lv = inkmod.local_levels(rm, 0.4, 0.7, [(range(12), True)])
    assert np.allclose(lv[:2], 0.7)                                     # dark / similar to the page: the page level, unchanged
    assert np.allclose(lv[7:], 0.5)   # lighter region: its own level; blank / ghost rows follow their neighbours
    assert np.allclose(inkmod.local_levels(np.array([0.0, 0.0]), 0.4, 0.7, [(range(2), True)]), 0.7)   # no marks in reach: page level


# ------------------------------------------------------------------ best guesses: uncertain rows keep their flag AND a value
def _row(vals, strength=None):
    f = np.array(vals, float)
    return f, (np.ones_like(f) if strength is None else np.asarray(strength, float))


def _gpage(fa, fi, fs=None):
    fa, fi = np.asarray(fa, float), np.asarray(fi, float)
    return R.PageRaw([1, 2], [["A"] * 5] * 2, list(range(len(fi))), fa, np.ones_like(fa), fi, np.ones_like(fi), 0.5, 0.6)


def test_guess_needs_a_leader_and_never_picks_an_unknowable_bubble():
    f, s = _row([0.0, 0.40, 0.05, 0.0, 0.0])
    assert R._guess(f, s, CAL) == 1                                           # weak but clearly the darkest
    f, s = _row([0.0, 0.40, 0.35, 0.0, 0.0])
    assert R._guess(f, s, CAL) is None                                        # two near-equal: nothing stands out
    f, s = _row([0.02, 0.04, 0.03, 0.0, 0.01])
    assert R._guess(f, s, CAL) is None                                        # noise on a blank row
    f, s = _row([0.0, R.UNRELIABLE_FILL, 0.0, 0.0, 0.0], [1, R.UNRELIABLE_STRENGTH, 1, 1, 1])
    assert R._guess(f, s, CAL) is None                                        # an invented value is not evidence


def test_weak_rows_carry_a_guess_multi_rows_none_and_clear_blanks_do_not_but_stay_flagged_and_clear_blanks_do_not():
    fa = [[0.0, 0.40, 0.05, 0.0, 0.0],         # weak: grey mark, ambiguous
          [0.0, 0.0, 0.0, 0.0, 0.0]]          # clearly blank
    d = R.decide_page(_gpage(fa, np.zeros((1, 10))))
    q1, q2 = d.answers
    assert q1.ambiguous and q1.guessed and q1.answer == 1
    assert q2.state == "blank" and not q2.ambiguous and not q2.guessed and q2.answer is None
    # a multi-mark row (even where one mark dominates) is never collapsed: no guess, the marks are the answer
    fa = [[0.95, 0.0, 0.75, 0.4, 0.0], [0.0] * 5]
    q = R.decide_page(_gpage(fa, np.zeros((1, 10)))).answers[0]
    assert q.state == "multi" and q.marked == [0, 2] and q.guess is None and not q.guessed and q.answer is None
    # two equal marks: no guess
    fa = [[0.95, 0.0, 0.95, 0.4, 0.0], [0.0] * 5]
    q = R.decide_page(_gpage(fa, np.zeros((1, 10)))).answers[0]
    assert q.state == "multi" and q.answer is None and not q.guessed


def test_guess_never_changes_state_marked_or_the_flag():
    fa = [[0.0, 0.40, 0.05, 0.0, 0.0], [0.0, 0.0, 0.9, 0.0, 0.0]]
    d = R.decide_page(_gpage(fa, np.zeros((1, 10))))
    assert [a.ambiguous for a in d.answers] == [True, False] and d.answers[1].guessed is False
    assert d.answers[1].answer == 2


def test_uncertain_id_digit_gets_its_guess_and_the_id_is_flagged_guessed():
    fi = np.zeros((3, 10))
    fi[0, 4] = 0.95
    fi[1, 7] = 0.95
    fi[2, 2] = 0.40                             # weak: strict digit rules do not read it
    d = R.decide_page(_gpage([[0.0] * 5] * 2, fi))
    assert [x.value for x in d.id_digits] == [4, 7, 2] and [x.guessed for x in d.id_digits] == [False, False, True]
    assert all(x.ambiguous for x in d.id_digits if x.guessed)
    assert d.student_id == "472" and "id_guessed" in d.flags and "id_unreadable" not in d.flags
    # a digit column that is clearly blank has no value: no ID at all (never invented)
    fi[2] = 0.0
    d = R.decide_page(_gpage([[0.0] * 5] * 2, fi))
    assert d.student_id is None and d.id_digits[2].value is None and "id_guessed" not in d.flags


def test_guessed_flags_and_serialisation():
    from omr.recognize import RecognitionResult, OK
    fi = np.zeros((2, 10))
    fi[0, 1], fi[1, 3] = 0.95, 0.40
    d = R.decide_page(_gpage([[0.0, 0.40, 0.05, 0.0, 0.0]] * 2, fi))
    r = RecognitionResult(OK, answers=d.answers, id_digits=d.id_digits, student_id=d.student_id, flags=d.flags, subject="")
    j = r.to_dict()
    assert j["student_id"] == "13" and j["id_guessed"] and j["guessed_answers"] == 2 and j["needs_review"]
    assert j["answers"][0]["guessed"] is True and j["id_digits"][1]["guessed"] is True and j["id_digits"][0]["guessed"] is False
    assert "guessed_answers" in R.SOFT_FLAGS and "id_guessed" not in R.SOFT_FLAGS


def test_fused_evidence_wins_over_a_single_frame_guess():
    z = np.zeros((1, 10))
    def raw(v):
        fi = z.copy()
        fi[0, 5] = v
        return R.PageRaw([1], [["A"] * 5], [0], np.zeros((1, 5)), np.ones((1, 5)), fi, np.ones((1, 10)), 0.5, 0.6)
    from omr.fusion import Track
    class F:    # minimal frame stand-in for Track
        def __init__(self, v):
            self.raw, self.form_key, self.page, self.serial, self.quality, self.homography = raw(v), "k", 0, None, {}, None
    tr = Track(1, F(0.40), 0.0)
    assert tr.decisions().id_digits[0].guessed and "id_guessed" in tr.decisions().flags and tr.id_unreadable()
    for i in range(4):
        tr.add(F(0.95), 0.1 * i, None)                       # later frames read it cleanly: the median settles
    d = tr.decisions()
    assert d.id_digits[0].value == 5 and not d.id_digits[0].guessed and "id_guessed" not in d.flags and not tr.id_unreadable()
