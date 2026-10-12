#!/usr/bin/env python
"""Relative-mark rule benchmark: faint graphite (15-25 % darker than paper) without / with a glare patch
(synth `faint`, `faint_glare`), rule off (REL_MARGIN huge) vs on, same pixels.

    python scripts/faint_bench.py --n 60 --jobs 8
    python scripts/faint_bench.py --conds faint_glare,clean,glare,k_light --n 40    # any synth condition (ordinary ones use make_frame)
"""
import argparse
import multiprocessing as mp
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import synth  # noqa: E402
from omr import recognize as R  # noqa: E402

for _kv in filter(None, os.environ.get("RECOG_PARAMS", "").split(",")):   # e.g. RECOG_PARAMS=REL_SURE_DIGIT=0.2
    setattr(R, _kv.split("=")[0], float(_kv.split("=")[1]))
REL = R.REL_MARGIN
MODES = {"off": 9.0, "on": REL}


def run_job(args):
    assets, name, cond_name, seed = args
    L, pdf = assets[name]
    rng = np.random.default_rng(seed)
    serial = 1 + seed % synth.mk_serial_max()
    mk = (synth.make_frame_faint if cond_name in synth.FAINT_CONDS else
          synth.make_frame_mixed if cond_name in synth.MIXED_CONDS else synth.make_frame)
    frame, truth = mk(L, pdf, synth.CONDS[cond_name], rng, mode="sheet", serial=serial)
    out = {"cond": cond_name, "res": {}}
    for m, v in MODES.items():
        R.REL_MARGIN = v
        res = R.recognize(frame)
        clean = tot = 0     # clear marks read right AND not flagged (what a user sees as "counted as marked")
        if res.status == "ok":
            for tq, a in zip(truth["questions"], res.answers):
                if tq["kind"] == "clear":
                    tot += 1
                    clean += int(a.state == "single" and a.marked == [tq["answer"]] and not a.ambiguous)
        out["res"][m] = {"status": res.status, "ev": synth.evaluate(res, truth), "clean": clean, "clear_n": tot}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="faint,faint_glare")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    o = ap.parse_args(argv)
    conds = o.conds.split(",")
    assets = synth.prepare(os.path.join(tempfile.gettempdir(), "streamomr_faint_bench"))
    names = list(assets)
    jobs = [(assets, names[i % len(names)], cn, o.seed * 1_000_003 + ci * 10_007 + i)
            for ci, cn in enumerate(conds) for i in range(o.n)]
    with mp.Pool(o.jobs) as pool:
        out = pool.map(run_job, jobs, chunksize=2)
    print(f"{'cond':<13}{'rule':<5}{'frames':>6}{'ok%':>5}{'clr read%':>10}{'clr CLEAN%':>11}{'clr flg%':>9}{'silent':>7}{'/qs':>6}"
          f"{'hard ok%':>9}{'ID read%':>9}{'ID wrong':>9}{'subj wrong':>11}")
    for cn in conds:
        rows = [r for r in out if r["cond"] == cn]
        for m in MODES:
            ok = [r["res"][m] for r in rows if r["res"][m]["status"] == "ok"]
            t = {}
            for r in ok:
                synth.merge(t, r["ev"])
            g = lambda k: t.get(k, 0)
            d = lambda a, b: 100.0 * a / b if b else float("nan")
            print(f"{cn:<13}{m:<5}{len(rows):>6}{d(len(ok), len(rows)):>5.0f}{d(g('q_clear_ok'), g('q_clear')):>10.2f}{d(sum(r['clean'] for r in [x['res'][m] for x in rows]), sum(r['clear_n'] for r in [x['res'][m] for x in rows])):>11.2f}"
                  f"{d(g('q_clear_flagged'), g('q_clear')):>9.2f}{g('q_silent'):>7}{g('q_total'):>6}{d(g('q_hard_ok'), g('q_hard')):>9.1f}"
                  f"{d(g('id_ok'), g('id_readable')):>9.1f}{g('id_wrong'):>9}{g('subject_wrong'):>11}  {synth.guess_line(t)}")
    for r in out:
        for m in MODES:
            ev = r["res"][m]["ev"]
            if m == "on" and (ev["id_wrong"] or ev["subject_wrong"]):
                print("WRONG ID/SUBJECT", r["cond"])
    return out


if __name__ == "__main__":
    main()
