#!/usr/bin/env python
"""How long after a sheet becomes readable does the stream commit it?  Simulated on a cached video recognition (scripts/video_cache.py).

    python scripts/latency_bench.py --cache out/devlog_cache.pkl --truth out/omr_test_truth.csv --period 8
    python scripts/latency_bench.py --cache out/devlog_cache.pkl --truth out/omr_test_truth.csv --period 8 --k 2

Ground truth = the printed test sheets (scripts/make_test_pdf.py truth CSV: student id + 45 answers).  For every sheet the video shows:
  avail   first video time of a run of >= MIN_RUN consecutive source frames that read the sheet exactly right (what any recogniser
          running on every frame could have committed);
  commit  the time StreamSession commits it when fed every `--period`-th frame (the live client manages ~3.7 fps = period 8 at 30 fps),
          averaged over all phases of the sampling grid;
  latency commit - avail.
Also counted: wrong commits (id/answers not matching a truth row), duplicate commits, missed sheets."""
import argparse
import csv
import os
import pickle
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from omr.stream import StreamSession  # noqa: E402

MIN_RUN = 3


def labels(r):
    return "|".join(a.labels[i] for a in r.answers for i in a.marked[:1]) if False else tuple("|".join(a.labels[i] for i in a.marked) for a in r.answers)


def load_truth(path):
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))[1:]
    return {r[1]: tuple(r[3:]) for r in rows}


def availability(res, fps, truth):
    """sheet id -> first time of a run >= MIN_RUN source frames reading it exactly right (ids that never occur are absent)."""
    run, first = {}, {}
    for i, r in enumerate(res):
        k = r.student_id if r is not None and r.status == "ok" else None
        good = k in truth and labels(r) == truth[k] and r.subject is not None
        for sid in list(run):
            if not (good and k == sid):
                run[sid] = 0
        if good:
            run[k] = run.get(k, 0) + 1
            if run[k] >= MIN_RUN and k not in first:
                first[k] = (i - MIN_RUN + 1) / fps
    return first


def simulate(res, fps, period, phase, **kw):
    s = StreamSession(resolve_layout=None, **kw)
    commits = []
    for i in range(phase, len(res), period):
        r = res[i]
        if r is None:
            continue
        ev = s.feed(r, i / fps)
        if "commit" in ev:
            c = ev["commit"]
            commits.append((i / fps, c["result"]))
    return commits


def simulate_phone(res, fps, cost_ms, factor, commit_extra_ms=0.0, **kw):
    """Back-to-back processing like the live client: the worker takes the newest camera frame, needs `cost_ms[i] * factor` ms for it,
    then takes the newest frame again.  Returns [(result time in s, result)] (the commit is known when the frame's processing ends)."""
    s = StreamSession(resolve_layout=None, **kw)
    commits, T, n = [], 0.0, len(res)
    while True:
        i = int(T * fps)
        if i >= n:
            break
        r = res[i]
        d = cost_ms[i] * factor / 1000.0
        T += d
        if r is None:
            continue
        ev = s.feed(r, T)
        if "commit" in ev:
            T += commit_extra_ms / 1000.0
            commits.append((T, ev["commit"]["result"]))
    return commits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--period", type=int, default=8, help="process every N-th source frame (8 = ~3.7 fps at 30 fps)")
    ap.add_argument("--k", type=int, help="StreamSession k (default: the module default)")
    ap.add_argument("--fast-k", type=int, help="StreamSession fast_k: frames needed when the result is fully confident")
    ap.add_argument("--phases", type=int, default=4)
    ap.add_argument("--phone", type=float, help="simulate the phone instead of a fixed period: per-frame cost = cache 'ms' x this factor (1.8 ~ iPhone vs a desktop "
                                                  "single-process run; the cache must come from --jobs 1 for honest timings)")
    ap.add_argument("--commit-extra", type=float, default=0.0, help="extra ms the commit frame takes (detail JPEG etc.), with --phone")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    c = pickle.load(open(a.cache, "rb"))
    res, fps = c["results"], c["fps"]
    truth = load_truth(a.truth)
    avail = availability(res, fps, truth)
    kw = {} if a.k is None else {"k": a.k}
    if a.fast_k is not None:
        kw["fast_k"] = a.fast_k
    lat, wrong, dup, missed = [], 0, 0, 0
    per = {}
    phases = range(0, a.period, max(a.period // a.phases, 1)) if not a.phone else range(0, 8)
    for ph in phases:
        if a.phone:     # phase = a start offset of 0..7 x 1/(8 fps) so the grid of frames is not always the same
            commits = simulate_phone(res[ph:], fps, c["ms"][ph:], a.phone, a.commit_extra, **kw)
            commits = [(t + ph / fps, r) for t, r in commits]
        else:
            commits = simulate(res, fps, a.period, ph, **kw)
        seen = set()
        for t, r in commits:
            ok = r.student_id in truth and labels(r) == truth[r.student_id]
            if not ok:
                wrong += 1
            elif r.student_id in seen:
                dup += 1
            else:
                seen.add(r.student_id)
                if r.student_id in avail:
                    lat.append(t - avail[r.student_id])
                    per.setdefault(r.student_id, []).append(t - avail[r.student_id])
        missed += len(set(avail) - seen)
    n_ph = len(phases)
    print((f"phone model x{a.phone}" if a.phone else f"period {a.period} (~{fps / a.period:.1f} fps)") + f", {n_ph} phases, {len(avail)} sheets readable in the video")
    print(f"latency commit - first correct read: mean {st.mean(lat):.2f}s  median {st.median(lat):.2f}s  p90 {sorted(lat)[int(len(lat) * .9)]:.2f}s  max {max(lat):.2f}s  (n={len(lat)})")
    print(f"wrong commits {wrong}  duplicate commits {dup}  missed sheets {missed}  (summed over phases)")
    if a.verbose:
        for sid, v in sorted(per.items(), key=lambda kv: avail[kv[0]]):
            print(f"  {sid} avail {avail[sid]:6.2f}s  latency " + " ".join(f"{x:.2f}" for x in v))


if __name__ == "__main__":
    main()
