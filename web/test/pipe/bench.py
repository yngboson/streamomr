"""JS-only timing per frame class (video frames, 8 per class, one jsc process, best of 3, after two warm-up frames):  .venv/bin/python web/test/pipe/bench.py [n_per_class] [reps]"""
import collections, json, sys
from common import *  # noqa
import numpy as np
n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
man = json.load(open(f"{FIX}/video.manifest.json")); py = json.load(open(f"{FIX}/video.py.json"))
by = collections.defaultdict(list)
for i, m in enumerate(man): by[py[m["name"]]["status"]].append(i)
pick = sorted(sum((v[::max(1, len(v) // n)][:n] for v in by.values()), []))
WARM = [pick[0], pick[-1]]       # two throw-away frames first: the JIT is cold at the start of a process (the first frames are ~2x slower)
pick = WARM + pick
p = run_jsc(["--", FIX, "video", ",".join(map(str, pick)), str(reps)], files=src_files() + [HERE + "/ser.js", HERE + "/drv.js"])
rows = collections.defaultdict(list)
for n_, l in enumerate([l for l in p.stdout.splitlines() if l.startswith("@")]):
    if n_ < len(WARM): continue
    j = json.loads(l[1:]); rows[py[j["name"]]["status"]].append(j["ms"])
for st, q in sorted(rows.items()): print(f"{st:12s} n={len(q):2d} mean {np.mean(q):6.0f}  max {np.max(q):6.0f}")
