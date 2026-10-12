#!/usr/bin/env python
"""How often does the self-describing header decode, per lighting condition?

    python scripts/header_bench.py --n 40 --jobs 8
    python scripts/header_bench.py --conds shadow,h_idedge --n 80

Reuses the synthetic conditions of scripts/illum_bench.py (h_* = harder lighting).  A frame counts as
"header ok" when recognize() got as far as a form_key / matching page; otherwise the status says where it failed
(no_sheet / partial / bad_geometry = markers, bad_meta = header cells).  Only frames whose four markers were found
and warped are the header's responsibility, so both rates are printed.
"""
import argparse
import collections
import multiprocessing as mp
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import illum_bench  # noqa: E402,F401  (registers the h_* / k_* conditions in synth.CONDS)
import synth  # noqa: E402
from omr.recognize import recognize  # noqa: E402

DEFAULT = ["clean", "dark", "curl", "glare", "shadow", "raking", "h_multi", "h_idedge", "h_dimgrad", "h_glare", "h_all"]
_W = {}


def _init(d):
    _W["assets"] = synth.prepare(d, {"q30": synth.STANDARD_TEMPLATES["q30_abcd"]})


def one(args):
    cond, seed = args
    L, pdf = _W["assets"]["q30"]
    frame, truth = synth.make_frame(L, pdf, synth.CONDS[cond], np.random.default_rng(seed), mode="sheet")
    r = recognize(frame)
    return cond, r.status, bool(r.form_key), (r.quality or {}).get("meta_contrast"), (r.quality or {}).get("meta_min_margin")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default=",".join(DEFAULT))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    conds = a.conds.split(",")
    tmp = tempfile.mkdtemp()
    with mp.Pool(a.jobs, _init, (tmp,)) as pool:
        out = pool.map(one, [(c, 1000 + s) for c in conds for s in range(a.n)], chunksize=2)
    print(f"{'cond':12s} {'n':>4s} {'hdr_ok':>7s} {'of_warped':>10s}  failures")
    for c in conds:
        rows = [o for o in out if o[0] == c]
        st = collections.Counter(o[1] for o in rows)
        warped = len(rows) - st["no_sheet"] - st["partial"] - st["bad_geometry"] - st["inconsistent"]
        ok = sum(o[2] for o in rows)
        fails = ", ".join(f"{k}:{v}" for k, v in st.items() if k != "ok")
        print(f"{c:12s} {len(rows):4d} {ok / len(rows):7.0%} {ok / max(warped, 1):10.0%}  {fails}")
