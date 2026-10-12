#!/usr/bin/env python
"""Accuracy / timing table over many synthetic camera frames.

    python scripts/bench.py [--n 40] [--seed 1] [--jobs 8] [--ppm 6]

Timing columns are measured per recognize() call on a 1080p-class frame; with --jobs > 1 the
workers compete for cores, so use --jobs 1 for clean timing numbers.
"""
import argparse
import multiprocessing as mp
import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import synth  # noqa: E402


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="frames per condition")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--ppm", type=float, default=None, help="canonical canvas px/mm")
    ap.add_argument("--conds", default=",".join(synth.CONDS))
    ap.add_argument("--list-silent", action="store_true", help="print every silent error")
    o = ap.parse_args()
    work = os.path.join(tempfile.gettempdir(), "streamomr_bench")
    assets = synth.prepare(work)
    names = list(assets)
    jobs = []
    conds = o.conds.split(",")
    for ci, cn in enumerate(conds):
        for i in range(o.n):
            jobs.append((assets, names[i % len(names)], cn, "sheet", o.seed * 1_000_003 + ci * 10_007 + i, o.ppm))
    neg_n = max(8, o.n // 2)
    for mode in ("none", "thumb2", "cut"):
        for i in range(neg_n):
            cn = conds[i % len(conds)] if mode != "none" else "tilted" if i % 2 else "phone"
            jobs.append((assets, names[i % len(names)], cn, mode, o.seed * 1_000_003 + 900_000 + {'none': 1, 'thumb2': 2, 'cut': 3}[mode] * 10_000 + i, o.ppm))
    t0 = time.time()
    if o.jobs > 1:
        with mp.Pool(o.jobs) as pool:
            out = pool.map(synth.run_one, jobs, chunksize=2)
    else:
        out = [synth.run_one(j) for j in jobs]
    print(f"{len(jobs)} frames in {time.time() - t0:.0f}s\n")

    hdr = (f"{'condition':<10}{'frames':>7}{'read%':>7}{'px/mm':>6}{'clear acc%':>11}{'silent':>7}{'/qs':>7}"
           f"{'hard flg%':>10}{'hard ok%':>9}{'clr flg%':>9}{'ID wrong':>9}{'ID read%*':>9}{'subj wrong':>11}{'subj read%*':>12}{'serial%':>8}{'ser bad':>8}{'t50ms':>7}{'t95ms':>7}")
    print(hdr)
    print("-" * len(hdr))
    silent_list = []
    for cn in conds + ["ALL"]:
        rows = [r for r in out if r["mode"] == "sheet" and (cn == "ALL" or r["cond"] == cn)]
        ok = [r for r in rows if r["status"] == "ok"]
        tot = {}
        for r in ok:
            synth.merge(tot, r["ev"])
            for f in r["ev"]["failures"]:
                if not f[6]:
                    silent_list.append((r["cond"], r["seed"], r["name"], f))
        g = lambda k: tot.get(k, 0)
        div = lambda a, b: 100.0 * a / b if b else float("nan")
        ppm = np.mean([r["px_per_mm"] for r in rows]) if rows else float("nan")
        ts = [r["ms"] for r in ok]
        print(f"{cn:<10}{len(rows):>7}{div(len(ok), len(rows)):>7.1f}{ppm:>6.2f}{div(g('q_clear_ok'), g('q_clear')):>11.2f}"
              f"{g('q_silent'):>7}{g('q_total'):>7}{div(g('q_hard_flagged'), g('q_hard')):>10.1f}"
              f"{div(g('q_hard_ok'), g('q_hard')):>9.1f}{div(g('q_clear_flagged'), g('q_clear')):>9.2f}"
              f"{g('id_wrong'):>9}{div(g('id_ok'), g('id_readable')):>9.1f}{g('subject_wrong'):>11}{div(g('subject_ok'), g('subject_total') - g('subject_none') + g('subject_missed')):>12.1f}{div(g('serial_ok'), g('serial_total')):>8.1f}{g('serial_wrong'):>8}{pct(ts, 50):>7.0f}{pct(ts, 95):>7.0f}")
    print("  (* ID read% = share of readable IDs that were read; sheets with a deliberately bad ID digit must stay unread;\n   same for the subject code: 'subj wrong' must be 0, 'subj read%' = share of readable subject codes that were read)")
    # statuses of sheet frames that were not read
    from collections import Counter
    lost = Counter((r["cond"], r["status"]) for r in out if r["mode"] == "sheet" and r["status"] != "ok")
    if lost:
        print("\nsheet frames not read (condition, status): ", dict(lost))
    # per-kind
    kinds = {}
    for r in out:
        if r["mode"] == "sheet" and r["status"] == "ok":
            for k, v in r["ev"]["kinds"].items():
                t = kinds.setdefault(k, [0, 0, 0, 0])
                for i in range(4):
                    t[i] += v[i]
    print("\nper question kind        n   correct%  flagged%  silent")
    for k, (n, c, f, s) in sorted(kinds.items()):
        print(f"  {k:<18}{n:>8}{100 * c / n:>10.1f}{100 * f / n:>10.1f}{s:>8}")
    # negatives
    print("\nnegative frames (must never yield a result):")
    for mode in ("none", "thumb2", "cut"):
        rows = [r for r in out if r["mode"] == mode]
        # a frame with >= 3 corners in view may legitimately be read; every other positive is a failure
        bad = [r for r in rows if r["status"] == "ok" and not (mode == "cut" and r["truth"].get("n_inside", 0) >= 3)]
        okcut = [r for r in rows if r["status"] == "ok" and mode == "cut" and r["truth"].get("n_inside", 0) >= 3]
        st = Counter(r["status"] for r in rows)
        ts = [r["ms"] for r in rows]
        print(f"  {mode:<6} n={len(rows):<4} results={len(bad)}  statuses={dict(st)}  t50={pct(ts, 50):.0f}ms t95={pct(ts, 95):.0f}ms"
              + (f"  (+{len(okcut)} legit 3-corner reads: silent={sum(r['ev']['q_silent'] for r in okcut)}, wrong ID={sum(r['ev']['id_wrong'] for r in okcut)})" if mode == "cut" else ""))
    # timing breakdown on sheet frames
    ok = [r for r in out if r["mode"] == "sheet" and r["status"] == "ok"]
    if ok:
        print("\ntiming breakdown, sheet frames, median ms: " +
              ", ".join(f"{k}={np.median([r['timing'].get(k, 0) for r in ok]):.1f}" for k in ok[0]["timing"]))
    nos = [r for r in out if r["mode"] == "none"]
    if nos:
        print("timing breakdown, no-sheet frames, median ms: " +
              ", ".join(f"{k}={np.median([r['timing'].get(k, 0) for r in nos]):.1f}" for k in nos[0]["timing"]))
    if o.list_silent:
        print("\nsilent errors:")
        for cn, seed, name, f in silent_list:
            print(" ", cn, seed, name, f[:7], np.round(f[7], 2))


if __name__ == "__main__":
    main()
