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
