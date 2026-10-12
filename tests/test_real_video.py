"""Regression on the first real recording (iPhone, 1080x1920 portrait, a hand-held pile of printed 45-question sheets
on a printer that renders black as grey with white streaks; the thumb covers a corner marker most of the time).

Ground truth: tests/data/IMG_4425.truth.json (ordered physical sheets, read by eye).  Skipped when the video is absent.
Rules checked at native / 1280 / 960 px: every commit belongs to a truth sheet; its student ID is correct or None (never
wrong); Q1-5 equal the truth or are flagged; Q6-45 are blank; a sheet committed more than once always has the same content.
"""
import json
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
VIDEO = os.path.join(ROOT, "realasset", "IMG_4425.MOV")
LAYOUT = os.path.join(ROOT, "out", "q45.layout.json")
TRUTH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "IMG_4425.truth.json")

pytestmark = pytest.mark.skipif(not (os.path.exists(VIDEO) and os.path.exists(LAYOUT)),
                                reason="real video / layout not available")

import replay_video  # noqa: E402

SLACK_BEFORE, SLACK_AFTER = 0.3, 1.2     # a commit may lag the sheet (stable run, ID hold, track end)


def truth_sheet(sheets, t):
    c = [s for s in sheets if s["t"][0] - SLACK_BEFORE <= t <= s["t"][1] + SLACK_AFTER]
    return min(c, key=lambda s: abs(t - sum(s["t"]) / 2)) if c else None


def check_commit(c, sh):
    assert c["student_id"] in (None, sh["student_id"]), (sh["name"], c["student_id"])
    for n, st, marked, amb in c["states"]:
        if n >= 6:
            assert st == "blank" and not marked, (sh["name"], n, st, marked)       # Q6-45 blank (ambiguous blank allowed)
        else:
            want = sorted(sh["q1_5"][str(n)])
            got = sorted(str(m + 1) for m in marked)
            assert amb or got == want, (sh["name"], n, got, want)
            if st == "multi":
                assert amb or c["needs_review"] or len(want) > 1


def run(max_side):
    return replay_video.main([VIDEO, "--layout", LAYOUT, "--fps", "8", "--max-side", str(max_side)])


@pytest.mark.parametrize("max_side", [0, 1280, 960])
def test_real_video_against_truth(max_side):
    truth = json.load(open(TRUTH))
    sheets = truth["sheets"]
    s = run(max_side)
    assert s["frames"] > 80 and len(s["commits"]) >= 1, s["status"]
    by_sheet = {}
    for c in s["commits"]:
        sh = truth_sheet(sheets, c["t"])
        assert sh is not None, ("commit outside every truth sheet", c["t"], c["student_id"])
        check_commit(c, sh)
        by_sheet.setdefault(sh["name"], []).append(c)
    for name, cs in by_sheet.items():              # no sheet committed twice with different content
        content = {(c["student_id"], tuple((n, st, tuple(m)) for n, st, m, _ in c["states"])) for c in cs}
        assert len(content) == 1, (name, [(c["t"], c["student_id"]) for c in cs])
    print(f"\nmax_side={max_side}: {len(by_sheet)}/{len(sheets)} truth sheets committed:",
          {k: [(c['t'], c['student_id']) for c in v] for k, v in by_sheet.items()})
    assert len(by_sheet) >= 1
    assert s["status"].get("partial", 0) > 0         # partial frames are "sheet present"
