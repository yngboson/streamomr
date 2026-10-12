#!/usr/bin/env python
"""Real-video check of the lighting / ink alternatives: replay realasset/IMG_4425.MOV through the live pipeline
(recognize + StreamSession) with each method hooked in, and score the commits against tests/data/IMG_4425.truth.json
with the same rules as tests/test_real_video.py (counted instead of asserted).

    python scripts/illum_video.py --methods E,E.f45 --sides 0,1280,960
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for sub in ("", "tests", "scripts"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import illum_bench as ib  # noqa: E402
import replay_video  # noqa: E402
import test_real_video as trv  # noqa: E402


def score(commits, sheets):
    wrong_id, wrong_q, outside, by = 0, 0, 0, {}
    for c in commits:
        sh = trv.truth_sheet(sheets, c["t"])
        if sh is None:
            outside += 1
            continue
        by.setdefault(sh["name"], []).append(c)
        if c["student_id"] not in (None, sh["student_id"]):
            wrong_id += 1
        for n, st, marked, amb in c["states"]:
            if n >= 6:
                wrong_q += int(not (st == "blank" and not marked))
            else:
                want, got = sorted(sh["q1_5"][str(n)]), sorted(str(m + 1) for m in marked)
                wrong_q += int(not (amb or got == want))
    ids = sum(1 for cs in by.values() if any(c["student_id"] for c in cs))
    return {"sheets": len(by), "of": len(sheets), "commits": len(commits), "wrong_id": wrong_id, "wrong_q": wrong_q,
            "outside": outside, "with_id": ids}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="E")
    ap.add_argument("--sides", default="0,1280,960")
    o = ap.parse_args(argv)
    sheets = json.load(open(trv.TRUTH))["sheets"]
    print(f"{'method':<8}{'max side':>9}{'sheets':>8}{'commits':>8}{'wrong ID':>9}{'wrong Q':>8}{'outside':>8}{'sheets w/ ID':>13}")
    for m in o.methods.split(","):
        for side in [int(x) for x in o.sides.split(",")]:
            ib.install(m)
            s = trv.run(side)
            r = score(s["commits"], sheets)
            print(f"{m:<8}{side:>9}{r['sheets']:>5}/{r['of']:<2}{r['commits']:>8}{r['wrong_id']:>9}{r['wrong_q']:>8}{r['outside']:>8}{r['with_id']:>13}", flush=True)


if __name__ == "__main__":
    main()
