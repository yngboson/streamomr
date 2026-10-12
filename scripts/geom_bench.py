#!/usr/bin/env python
"""Geometry bench: answer accuracy / ID read / silent errors per paper-shape condition (fold, curl, raking, ...).

    python scripts/geom_bench.py [--n 40] [--jobs 8] [--conds fold,fold_curl,curl,clean] [--tracks 1|0]

--tracks 0 prints the same forms without timing tracks (format v1) for a before/after comparison on the same
frames' paper shapes (the random streams differ because the sheets differ).  --root points at another checkout
(e.g. a snapshot of the code before the change).
"""
import argparse
import multiprocessing as mp
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_ap = argparse.ArgumentParser()
_ap.add_argument("--root", default=os.path.join(HERE, ".."))
_ap.add_argument("--n", type=int, default=40)
_ap.add_argument("--seed", type=int, default=1)
_ap.add_argument("--jobs", type=int, default=8)
_ap.add_argument("--conds", default="clean,curl,raking,h_all,fold,fold_curl,fold_hand,fold_big")
_ap.add_argument("--tracks", type=int, default=1)
_ap.add_argument("--templates", default="q45,q100_2p,q100_3p,q30_abcd")
_ap.add_argument("--list-silent", action="store_true")
O = _ap.parse_args()
ROOT = os.path.abspath(O.root)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import dataclasses  # noqa: E402

import illum_bench  # noqa: E402,F401  (registers the h_* conditions)
import synth  # noqa: E402

_W = {}


def _init(workdir, tracks):
    tpl = {k: synth.STANDARD_TEMPLATES[k] for k in O.templates.split(",")}
    if "tracks" in {f.name for f in dataclasses.fields(synth.lay.Template)}:
        tpl = {k: dataclasses.replace(t, tracks=bool(tracks)) for k, t in tpl.items()}
    _W["assets"] = synth.prepare(workdir, tpl)


def job(a):
    name, cond, seed = a
    return synth.run_one((_W["assets"], name, cond, "sheet", seed, None))


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


if __name__ == "__main__":
    conds = O.conds.split(",")
    names = O.templates.split(",")
    tmp = tempfile.mkdtemp()
    jobs = [(names[i % len(names)], c, O.seed * 1_000_003 + ci * 10_007 + i) for ci, c in enumerate(conds) for i in range(O.n)]
    t0 = time.time()
    with mp.Pool(O.jobs, _init, (tmp, O.tracks)) as pool:
        out = pool.map(job, jobs, chunksize=2)
    print(f"{len(jobs)} frames in {time.time() - t0:.0f}s (tracks={O.tracks})\n")
    hdr = f"{'cond':10s}{'n':>4s}{'read%':>7s}{'clear%':>8s}{'hard ok%':>9s}{'silent':>7s}{'silClr':>7s}{'/qs':>6s}{'IDread%':>8s}{'IDwrong':>8s}{'subjW':>6s}{'t50':>6s}{'t95':>6s}"
    print(hdr)
    div = lambda a, b: 100.0 * a / b if b else float("nan")
    for cn in conds + ["ALL"]:
        rows = [r for r in out if cn == "ALL" or r["cond"] == cn]
        ok = [r for r in rows if r["status"] == "ok"]
        tot = {}
        for r in ok:
            synth.merge(tot, r["ev"])
            if O.list_silent:
                for f in r["ev"]["failures"]:
                    if not f[6]:
                        print("silent", r["cond"], r["seed"], r["name"], f[:7])
        g = lambda k: tot.get(k, 0)
        ts = [r["ms"] for r in ok]
        print(f"{cn:10s}{len(rows):4d}{div(len(ok), len(rows)):7.1f}{div(g('q_clear_ok'), g('q_clear')):8.2f}{div(g('q_hard_ok'), g('q_hard')):9.1f}"
              f"{g('q_silent'):7d}{g('q_silent_clear'):7d}{g('q_total'):6d}{div(g('id_ok'), g('id_readable')):8.1f}{g('id_wrong'):8d}{g('subject_wrong'):6d}{pct(ts, 50):6.0f}{pct(ts, 95):6.0f}")
    st = {}
    for r in out:
        st.setdefault(r["cond"], {}).setdefault(r["status"], 0)
        st[r["cond"]][r["status"]] += 1
    print("\nstatus:", {c: v for c, v in st.items()})
